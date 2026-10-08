"""Compare CCE with a materialized head; emits reproducible CUDA JSON metrics."""

import argparse
import json
import time

import torch
import torch.nn.functional as F

from areno.accel.cce import cut_logprobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--vocab", type=int, default=157184)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = getattr(torch, args.dtype)
    x = torch.randn(args.tokens, args.hidden, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(args.vocab, args.hidden, device="cuda", dtype=dtype) * 0.02
    labels = torch.randint(args.vocab, (args.tokens,), device="cuda")
    scale = torch.randn(args.tokens, device="cuda") / args.tokens

    def reference():
        return F.linear(x, w).float().log_softmax(-1).gather(1, labels[:, None]).flatten()

    result = {"shape": vars(args), "torch": torch.__version__, "device": torch.cuda.get_device_name()}
    outputs = []
    for name, fn in (("materialized", reference), ("cce", lambda: cut_logprobs(x, w, labels))):
        for _ in range(2):
            out = fn()
            (out * scale).sum().backward()
            x.grad = None
            del out
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        for _ in range(3):
            out = fn()
            (out * scale).sum().backward()
            outputs.append((out.detach().cpu(), x.grad.detach().cpu())) if _ == 2 else None
            x.grad = None
            del out
        torch.cuda.synchronize()
        result[name] = {
            "seconds_per_step": (time.perf_counter() - begin) / 3,
            "peak_extra_mib": (torch.cuda.max_memory_allocated() - base) / 2**20,
        }
    result["logprob_max_abs"] = (outputs[0][0] - outputs[1][0]).abs().max().item()
    result["gradient_max_abs"] = (outputs[0][1] - outputs[1][1]).abs().max().item()
    tolerance = 2e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], atol=tolerance, rtol=tolerance)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
