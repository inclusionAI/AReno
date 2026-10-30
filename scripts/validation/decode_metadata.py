"""Real-model decode scheduler comparison against an isolated baseline module.

Fixed-token runs retain model forward, KV/recurrent state, graph replay,
selected-token logprobs and EOS handling. Only token choice is scripted so
both schedulers see identical completions, refill and shrinking batch tails.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from areno.engine import inference as candidate
from areno.engine.config import EngineConfig, ModelConfig, RuntimeConfig
from areno.engine.data import SamplingParams
from areno.engine.data.rollout_state import InferenceBatchState
from areno.engine.modeling import build_model_on_device
from areno.engine.parallel.context import TPContext, set_tp_context
from areno.engine.runtime.decode_graph import DecodeGraph
from areno.models.registry import config_from_hf, load_model_weights


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("baseline_decode_inference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def runner_type(module):
    class Runner(module.InferenceManager):
        def plan(self, lengths, slots):
            position = lengths.long() - self.prompt_lens_by_slot[slots] + 1
            ids = self.row_by_slot[slots]
            tokens = (ids * 13 + position) % (self.config.model.vocab_size - 4) + 3
            limits = self.limits_by_slot[slots]
            return torch.where((position + 1 >= limits) & (limits < self.output_tokens), 2, tokens)

        def _infer_next_token_tensor(self, payload):
            t = time.perf_counter()
            indices = payload.sample_indices.tolist()
            ends = payload.raw["cu_seqlens"][1:].tolist()
            complete = [i for i, end in enumerate(ends) if end - 1 in indices]
            rows = [payload.raw["prefill_seq_ids"][i] for i in complete]
            slots = payload.raw["recurrent_slots"][complete].to(self.device)
            lengths = torch.tensor([len(self.prompts[row]) for row in rows], device=self.device)
            self.prompt_lens_by_slot[slots] = lengths
            self.limits_by_slot[slots] = torch.tensor([self.target_lengths[row] for row in rows], device=self.device)
            self.row_by_slot[slots] = torch.tensor(rows, device=self.device)
            self.planned = self.plan(lengths - 1, slots)
            result = super()._infer_next_token_tensor(payload)
            self.enqueue_seconds += time.perf_counter() - t
            return result

        def _run_prefill_payload(self, payload):
            t = time.perf_counter()
            result = super()._run_prefill_payload(payload)
            self.enqueue_seconds += time.perf_counter() - t
            return result

        def _infer_decode_next_token_tensor(self, *args, **kwargs):
            t = time.perf_counter()
            self.decode_iterations += 1
            if self.cancel_flags is not None and kwargs["sample_step"] == 3:
                self.cancel_flags[1] = True
            if self.first_decode is None:
                self.first_decode = t
            self.planned = self.plan(args[2], args[4])
            with torch.cuda.nvtx.range("model_and_sample"):
                result = super()._infer_decode_next_token_tensor(*args, **kwargs)
            self.enqueue_seconds += time.perf_counter() - t
            return result

    return Runner


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8, 64])
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--prompt-tokens", type=int, default=257)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--modes", nargs="+", choices=["eager", "graph"], default=["eager", "graph"])
    p.add_argument("--profile", choices=["baseline", "candidate"])
    p.add_argument(
        "--scenarios",
        nargs="+",
        choices=["steady", "tail_refill", "cancel_refill", "continuous_refill"],
        default=["steady", "tail_refill"],
    )
    p.add_argument("--greedy", action="store_true", help="Use real greedy sampling instead of fixed-token choice")
    args = p.parse_args()
    rank, size, local = (int(os.environ.get(k, d)) for k, d in [("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if size > 1:
        dist.init_process_group("nccl")
    set_tp_context(
        TPContext(
            rank=rank,
            world_size=size,
            device=device,
            group=dist.group.WORLD if size > 1 else None,
            global_rank=rank,
            global_world_size=size,
            role="rollout",
            train_tp_size=size,
        )
    )
    baseline = load_baseline(args.baseline)
    torch.manual_seed(633)
    if args.tiny:
        config = ModelConfig(
            model_type="qwen3",
            vocab_size=1024,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=32,
            dtype=torch.bfloat16,
            max_position_embeddings=2048,
        )
    else:
        if args.model is None:
            p.error("--model or --tiny is required")
        config = config_from_hf(args.model)
    engine = EngineConfig(
        config,
        model_path=str(args.model) if not args.tiny else None,
        devices=list(range(size)),
        tp_size=size,
        role="rollout",
        runtime=RuntimeConfig(compile_model=False, eager_decode=True),
    )
    model = build_model_on_device(engine, device).eval()
    if not args.tiny:
        load_model_weights(model, config, args.model)
    prepare = getattr(model, "prepare_infer_weights", None)
    if prepare:
        prepare()
    max_batch = max(args.batches)
    max_length = args.prompt_tokens + args.tokens + 16
    columns = (max_length + 255) // 256
    blocks = max_batch * columns
    caches = model.allocate_kv_caches(blocks + 1, 256, device)
    options = {"num_slots": max_batch + 1} if "num_slots" in inspect.signature(model.set_kv_caches).parameters else {}
    model.set_kv_caches(caches, **options)
    graphs = {}
    if "graph" in args.modes:
        pool = torch.cuda.graph_pool_handle()
        buckets = sorted(set([1, 2, 4, 8] + args.batches))
        for bucket in buckets:
            if bucket > max_batch:
                continue
            graph = DecodeGraph(model, bucket, columns, blocks, max_batch, device, graph_pool=pool)
            graph.warmup(iterations=2)
            graph.capture()
            graphs[bucket] = graph
            if rank == 0:
                print(json.dumps({"captured_bucket": bucket}), flush=True)

    def close():
        # Captured NCCL collectives retain communicator references. Destroy
        # CUDA graphs before shutting down the distributed process group.
        torch.cuda.synchronize()
        for captured in graphs.values():
            captured.graph.reset()
        graphs.clear()
        if size > 1:
            dist.barrier()
            dist.destroy_process_group()

    runners = {}
    for name, module in [("baseline", baseline), ("candidate", candidate)]:
        worker = SimpleNamespace(
            device=device,
            model=model,
            config=engine,
            _decode_graphs=graphs,
            prompt_lens_by_slot=torch.empty(max_batch, device=device, dtype=torch.long),
            row_by_slot=torch.empty(max_batch, device=device, dtype=torch.long),
            limits_by_slot=torch.empty(max_batch, device=device, dtype=torch.long),
            output_tokens=args.tokens,
        )
        runner = runner_type(module)(worker)
        runners[name] = runner
        if not args.greedy:
            module._sample_greedy_sharded = lambda *a, _runner=runner, **kw: _runner.planned

    def run(name, batch, scenario, mode):
        engine.runtime.eager_decode = mode == "eager"
        manager = runners[name]
        count = batch if scenario == "steady" else batch * 2
        manager.prompts = [[3 + (row % 7)] * (args.prompt_tokens - row % 4) for row in range(count)]
        manager.target_lengths = [
            args.tokens if scenario == "steady" or row % 4 == 3 else min(args.tokens, [4, 8, 16][row % 4])
            for row in range(count)
        ]
        state = InferenceBatchState(
            manager.prompts[:batch] if scenario == "continuous_refill" else manager.prompts,
            args.tokens,
            max_running_seqs=batch,
            max_cache_len=max_length,
            max_prefill_tokens=batch * args.prompt_tokens,
            kv_block_size=256,
            num_cache_blocks=blocks,
        )
        manager.cancel_flags = torch.zeros(count, dtype=torch.bool) if scenario == "cancel_refill" else None
        appended = False

        def refill(state):
            nonlocal appended
            if not appended and state._pending_seq_id == len(state.prompts) and len(state._seq_to_blocks) < batch:
                appended = True
                return state.append_prompts(manager.prompts[batch:])
            return []

        manager.enqueue_seconds = 0.0
        manager.decode_iterations = 0
        manager.first_decode = None
        model.reset_kv_caches()
        torch.cuda.synchronize()
        dist.barrier() if size > 1 else None
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.cuda.nvtx.range(name + ":rollout"):
            manager._generate_rollout_tokens_no_sync(
                state,
                SamplingParams(temperature=0.0),
                2 if scenario != "steady" else None,
                cancel_flags=manager.cancel_flags,
                cancel_indices=list(range(count)) if manager.cancel_flags is not None else None,
                refill_callback=refill if scenario == "continuous_refill" else None,
            )
        torch.cuda.synchronize()
        end = time.perf_counter()
        elapsed = end - start
        decode_ms = (end - (manager.first_decode or start)) * 1000
        row = {
            "wall_ms": elapsed * 1000,
            "decode_ms": decode_ms,
            "tpot_ms": decode_ms / max(manager.decode_iterations, 1),
            "decode_iterations": manager.decode_iterations,
            "tokens_per_s": sum(map(len, state.generated)) / elapsed,
            "host_outside_model_ms": (elapsed - manager.enqueue_seconds) * 1000,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "decode_scheduled_tokens": state.metrics["decode_scheduled_tokens"],
        }
        return state, row

    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "tp_size": size,
        "model": "seeded Qwen3 4x256" if args.tiny else str(args.model),
        "tokens": args.tokens,
        "prompt_tokens": args.prompt_tokens,
        "warmups_per_case": 1,
        "repeats": args.repeats,
        "sampling": "greedy" if args.greedy else "fixed token plan; normal EOS and length handling",
        "host_timing": "outside model/sample enqueue, including completion waits and final delivery",
        "cases": [],
    }
    for mode in args.modes:
        for batch in args.batches:
            for scenario in args.scenarios:
                before, _ = run("baseline", batch, scenario, mode)
                after, _ = run("candidate", batch, scenario, mode)
                assert before.generated == after.generated, "fixed-token outputs differ"
                assert before.finish_reason == after.finish_reason, "finish reasons differ"
                a = torch.tensor([v for row in before.logprobs for v in row])
                b = torch.tensor([v for row in after.logprobs for v in row])
                delta = (a - b).abs()
                torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
                samples = {"baseline": [], "candidate": []}
                if args.profile:
                    torch.cuda.cudart().cudaProfilerStart()
                    run(args.profile, batch, scenario, mode)
                    torch.cuda.cudart().cudaProfilerStop()
                    close()
                    return
                for repeat in range(args.repeats):
                    for name in ["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]:
                        _, row = run(name, batch, scenario, mode)
                        samples[name].append(row)
                medians = {
                    name: {k: statistics.median(row[k] for row in rows) for k in rows[0]}
                    for name, rows in samples.items()
                }
                case = {
                    "mode": mode,
                    "batch": batch,
                    "scenario": scenario,
                    "logprob_abs_diff_mean": float(delta.mean()),
                    "logprob_abs_diff_max": float(delta.max()),
                    "tokens_equal": True,
                    "finish_reasons_equal": True,
                    "medians": medians,
                    "samples": samples,
                    "speedup": medians["baseline"]["wall_ms"] / medians["candidate"]["wall_ms"],
                }
                report["cases"].append(case)
                if rank == 0:
                    args.output.write_text(json.dumps(report, indent=2))
                    print(json.dumps({k: v for k, v in case.items() if k != "samples"}), flush=True)
    close()


if __name__ == "__main__":
    main()
