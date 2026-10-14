"""Complete agent histories survive partial writes, replay and pagination."""

import json
from concurrent.futures import ThreadPoolExecutor

from areno.api.agent_events import AgentEvents
from areno.dashboard.agent_events import EventIndex


def test_concurrent_samples_have_stable_unique_ids_and_complete_histories(tmp_path):
    recorder = AgentEvents(tmp_path)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(
                pool.map(
                    lambda i: recorder.record(
                        "sample", step=2, prompt_idx=i // 4, sample_idx=i % 4, status="queued", prompt=f"task {i}"
                    ),
                    range(120),
                )
            )
        index = EventIndex(tmp_path)
        first = index.overview()
        second = index.overview(offset=100)
        assert first["total"] == 120
        assert len(first["samples"]) == 100
        assert len(second["samples"]) == 20
        assert not {s["id"] for s in first["samples"]} & {s["id"] for s in second["samples"]}
        events = index.read(limit=1000)
        assert len(events) == len({e["id"] for e in events}) == 120
        selected = first["samples"][0]["id"]
        assert len(index.read(sample=selected)) == 1
        assert index.read(after=events[-1]["cursor"]) == []
    finally:
        recorder.close()


def test_partial_live_record_is_not_consumed_and_replay_is_idempotent(tmp_path):
    path = tmp_path / "agent_events.42.jsonl"
    event = dict(
        id="42:1", pid=42, step=0, prompt_idx=0, sample_idx=0, kind="sample", time=1, data={"status": "running"}
    )
    text = json.dumps(event)
    path.write_text(text[:20])
    index = EventIndex(tmp_path, 42)
    assert index.read() == []
    with path.open("a") as f:
        f.write(text[20:] + "\n")
    assert len(index.read()) == 1
    path.write_text(text + "\n" + text + "\n")
    assert len(index.read()) == 1
    assert len(EventIndex(tmp_path, 42).read()) == 1


def test_sample_summary_merges_status_and_reward_without_tool_output(tmp_path):
    recorder = AgentEvents(tmp_path)
    try:
        recorder.record("sample", step=1, prompt_idx=0, sample_idx=0, prompt="repair", status="queued")
        recorder.record("tool", step=1, prompt_idx=0, sample_idx=0, result="large tool output")
        recorder.record("sample", step=1, prompt_idx=0, sample_idx=0, status="grading", reason="turn_limit")
        recorder.record("reward", step=1, prompt_idx=0, sample_idx=0, status="completed", reward=0.0)
        index = EventIndex(tmp_path)
        sample = index.overview()["samples"][0]
        assert sample["reward"] == 0.0
        assert sample["reason"] == "turn_limit"
        assert sample["prompt"] == "repair"
        assert "result" not in sample
        assert index.overview(step=0)["total"] == 0
    finally:
        recorder.close()


def test_owner_index_includes_worker_progress_and_statistics_cover_all_pages(tmp_path):
    owner = AgentEvents(tmp_path)
    worker = AgentEvents(tmp_path, owner_pid=owner.owner_pid)
    try:
        for i in range(120):
            owner.record("sample", step=1, prompt_idx=i, sample_idx=0, status="completed")
            owner.record("reward", step=1, prompt_idx=i, sample_idx=0, reward=float(i >= 100))
        worker.record("progress", step=1, phase="mst", chunk_size=1536)
        index = EventIndex(tmp_path, owner.owner_pid)
        overview = index.overview()
        assert overview["total"] == 120
        assert overview["stats"]["scored"] == 120
        assert overview["stats"]["perfect"] == 20
        assert overview["stats"]["mean_reward"] == 20 / 120
        assert overview["progress"][0]["data"]["chunk_size"] == 1536
        assert sum(overview["stats"]["histogram"]) == 120
        reopened = EventIndex(tmp_path, owner.owner_pid)
        assert reopened.offsets
        assert reopened.overview()["stats"] == overview["stats"]
    finally:
        worker.close()
        owner.close()


def test_ordinary_rollout_history_is_incremental_paged_and_preserves_media(tmp_path):
    path = tmp_path / "rollout_samples.42.jsonl"
    rows = [
        dict(
            pid=42,
            step=i,
            prompt_idx=0,
            sample_idx=0,
            kind="rollout",
            prompt=f"prompt {i}",
            completion=f"answer {i}",
            source_record={"audio_path": "/dataset/sample.wav"},
            prompt_tokens=[1, 2],
            response_tokens=[3],
        )
        for i in range(120)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    index = EventIndex(tmp_path, 42, rollout_files=[path])
    overview = index.overview()
    assert overview["total"] == 120
    assert len(overview["samples"]) == 100
    assert len(index.overview(offset=100)["samples"]) == 20
    sample = index.overview(step=0)["samples"][0]
    assert sample["capture"]["source_record"]["audio_path"] == "/dataset/sample.wav"
    assert sample["mode"] == "rollout"
    events = index.read(sample=sample["id"])
    assert events[1]["data"]["response"]["content"] == "answer 0"
    # The existing token previews are truncated, not exact token counts.
    assert events[1]["data"]["input_tokens"] is None
    assert overview["stats"]["mean_reward"] is None
    tail = json.dumps(dict(rows[-1], step=120))
    with path.open("a") as f:
        f.write(tail[:-2])
    assert index.overview()["total"] == 120
    with path.open("a") as f:
        f.write(tail[-2:] + "\n")
    assert index.overview()["total"] == 121
    reopened = EventIndex(tmp_path, 42, rollout_files=[path])
    assert reopened.overview()["total"] == 121
    assert len(reopened.read(limit=1000)) == 242


def test_harness_trace_takes_precedence_over_completion_capture(tmp_path):
    path = tmp_path / "rollout_samples.42.jsonl"
    path.write_text(
        json.dumps(
            dict(
                pid=42, kind="agentic", step=0, prompt_idx=0, sample_idx=0, prompt="legacy", completion="legacy answer"
            )
        )
        + "\n"
    )
    event = dict(
        id="rich",
        pid=42,
        step=0,
        prompt_idx=0,
        sample_idx=0,
        kind="sample",
        time=1,
        data=dict(prompt="live", harness="pi", status="running"),
    )
    (tmp_path / "agent_events.42.jsonl").write_text(json.dumps(event) + "\n")
    other = dict(pid=99, step=1, prompt_idx=0, sample_idx=0, prompt="another job")
    with path.open("a") as f:
        f.write(json.dumps(other) + "\n")
    index = EventIndex(tmp_path, 42, rollout_files=[path])
    assert index.overview()["total"] == 1
    assert index.overview()["samples"][0]["prompt"] == "live"
    assert len(index.read()) == 1


def test_timeline_is_uniformly_sampled_and_searches_unsampled_history():
    from areno.dashboard.agent_events import timeline_overview

    native = [dict(step=step, segments=[dict(name="train", seconds=step + 1)]) for step in range(200)]
    overview = timeline_overview(native, [])
    assert overview["total"] == 200
    steps = [row["step"] for row in overview["rows"]]
    assert len(steps) == 10
    assert steps[0] == 0 and steps[-1] == 199
    gaps = [right - left for left, right in zip(steps, steps[1:])]
    assert max(gaps) - min(gaps) <= 1
    assert 7 not in steps
    assert timeline_overview(native, [], step=7)["rows"] == [native[7]]
    assert timeline_overview(native, [], step=999)["rows"] == []
    assert timeline_overview(native, [], step=999)["total"] == 0
    average = timeline_overview(native, [], step=7)["average"]
    assert average["count"] == 200
    assert average["segments"] == [dict(name="train", seconds=100.5)]


def test_timeline_native_durations_replace_phase_fallback_without_duplicate_steps():
    from areno.dashboard.agent_events import timeline_overview

    stages = [
        dict(step=step, time=time, data=dict(stage=phase))
        for step, time, phase in [
            (0, 1, "train_start"),
            (0, 8, "train_end"),
            (1, 10, "rollout_start"),
            (1, 14, "rollout_end"),
        ]
    ]
    native = [dict(step=0, segments=[dict(name="train", seconds=6)])]
    assert timeline_overview(native, stages)["rows"] == [
        native[0],
        dict(step=1, segments=[dict(name="rollout", seconds=4)]),
    ]


def test_batch_statistics_cover_full_history_and_preserve_pending_training(tmp_path):
    recorder = AgentEvents(tmp_path)
    try:
        recorder.record(
            "batch",
            step=0,
            requested=8,
            returned=8,
            scored=6,
            accepted=6,
            learning=dict(zero=1, one=0, varying=1, flat_other=0, insufficient=0),
            sources=[dict(label="repo-a", samples=6, reward_sum=3, train_rows=12, train_tokens=400)],
        )
        index = EventIndex(tmp_path)
        overview = index.overview()
        assert overview["funnel"] == dict(requested=8, returned=8, scored=6, accepted=6, trained=None)
        assert overview["learning"]["varying"] == 1
        assert overview["sources"][0]["reward"] == 0.5
        recorder.record("stage", step=0, stage="train_end", status="running", trained_samples=6)
        recorder.record(
            "batch",
            step=1,
            requested=4,
            returned=4,
            scored=4,
            accepted=4,
            learning=dict(zero=0, one=1, varying=0, flat_other=0, insufficient=0),
            sources=[dict(label="repo-a", samples=4, reward_sum=4, train_rows=8, train_tokens=300)],
        )
        recorder.record("stage", step=1, stage="train_skip", status="running", trained_samples=0)
        overview = index.overview()
        assert overview["funnel"]["requested"] == 12
        assert overview["funnel"]["trained"] == 6
        assert overview["sources"][0]["samples"] == 10
        assert overview["sources"][0]["train_tokens"] == 700
        assert overview["sources"][0]["reward"] == 0.7
        assert overview["learning"]["one"] == 1
        assert index.overview(step=0)["funnel"]["requested"] == 8
        assert index.overview(step=1)["funnel"]["trained"] == 0
        assert len(overview["notices"]) == 1
        assert overview["notices"][0]["data"]["stage"] == "train_skip"
        assert EventIndex(tmp_path).overview()["funnel"] == overview["funnel"]
    finally:
        recorder.close()


def test_learning_summary_uses_real_group_variation_and_nonbinary_rewards():
    from areno.api.agent_events import learning_summary, source_label

    assert learning_summary([[0, 0], [1, 1], [0, 1], [0.3, 0.3], [1], []]) == dict(
        zero=1, one=1, varying=1, flat_other=1, insufficient=2
    )
    assert source_label({"source": {"dataset": "dataset-v1", "repo": "repo"}}) == "dataset-v1"
    assert source_label({"repo": "small/repo"}) == "small/repo"
    assert source_label({"source": True}) == "Unlabeled source"


def test_policy_learning_overview_reports_episode_counts_separately_from_rows():
    from types import SimpleNamespace

    from areno.api.trainers.policy_only import PolicyOnlyTrainer

    events = []
    trainer = PolicyOnlyTrainer.__new__(PolicyOnlyTrainer)
    trainer._dashboard_step = 4
    trainer.areno = SimpleNamespace(record_agent_event=lambda kind, **data: events.append((kind, data)))
    record = {"source": {"dataset": "coding"}}
    trainer._record_learning_overview(
        [[0, 1]],
        [(record, [1, 2]), (record, [1, 2, 3]), (record, [1, 2, 3, 4])],
        [(record, 0), (record, 1)],
        requested=2,
        returned=2,
        accepted=2,
    )
    kind, data = events[0]
    assert kind == "batch" and data["step"] == 4
    assert data["accepted"] == 2
    assert data["learning"]["varying"] == 1
    assert data["sources"] == [dict(label="coding", samples=2, reward_sum=1, train_rows=3, train_tokens=9)]


def test_agentic_batch_telemetry_preserves_native_training_rows_and_rewards():
    from types import SimpleNamespace

    from areno.api.agentic import AgentTrainBatch
    from areno.api.rewards import RewardRecord
    from areno.api.trainers.policy_only import PolicyOnlyTrainer

    batch = AgentTrainBatch(
        token_rows=[[10, 11], [10, 12]],
        response_masks=[[False, True], [False, True]],
        loss_masks=[[False, True], [False, True]],
        rollout_logprobs=[[0, -0.1], [0, -0.2]],
        features=[None, None],
        rewards=[0, 1],
        records=[{}, {}],
        reward_records=[
            RewardRecord(
                prompt="p",
                completion="a",
                source_record={"repo": "small/repo"},
                metadata={"prompt_index": 0, "sample_index": index},
            )
            for index in range(2)
        ],
    )
    trainer = PolicyOnlyTrainer.__new__(PolicyOnlyTrainer)
    trainer.areno = None
    tokenizer = SimpleNamespace(eos_token_id=2)
    prompts = SimpleNamespace(items=[object()])
    original = trainer._materialize_agentic_train_batch(tokenizer, prompts, batch)
    events = []
    trainer.areno = SimpleNamespace(record_agent_event=lambda kind, **data: events.append(data))
    trainer.config = SimpleNamespace(n_samples=2)
    observed = trainer._materialize_agentic_train_batch(tokenizer, prompts, batch)
    assert [row.model_dump() for row in original[0]] == [row.model_dump() for row in observed[0]]
    assert original[1] == observed[1] == [0, 1]
    assert events[0]["learning"]["varying"] == 1
    assert events[0]["requested"] == 2
    assert events[0]["sources"][0]["train_tokens"] == 4


def test_matrix_uniform_steps_preserves_full_statistics_and_exact_step(tmp_path):
    recorder = AgentEvents(tmp_path)
    try:
        for step in range(200):
            for sample in range(4):
                recorder.record("reward", step=step, prompt_idx=0, sample_idx=sample, reward=step / 199)
        index = EventIndex(tmp_path)
        view = index.overview(step_limit=10)
        assert {row["step"] for row in view["samples"]} == {round(i * 199 / 9) for i in range(10)}
        assert view["total"] == 40
        assert view["matched_total"] == 800
        assert view["stats"]["scored"] == 800
        exact = index.overview(step=7, step_limit=10)
        assert exact["total"] == 4
        assert {row["step"] for row in exact["samples"]} == {7}
        assert len(index.overview()["samples"]) == 100
    finally:
        recorder.close()


def test_live_turns_emit_before_harness_finishes_and_context_resets():
    import asyncio

    from areno.api.agent_events import capture_agent_turns
    from areno.api.agentic import AgentItem, AgentTrajectoryTurn

    item = AgentItem(record={}, prompt="task", input_tokens=[1], prompt_index=0, sample_index=0)
    recorded = []

    def recorder(kind, **data):
        recorded.append((kind, data))

    async def run():
        with capture_agent_turns(recorder):

            async def task(item):
                for value in range(2):
                    turn = AgentTrajectoryTurn(
                        item=item,
                        messages=[{"role": "user", "content": "task"}],
                        response={
                            "choices": [{"message": {"role": "assistant", "content": str(value)}}],
                            "areno": {"input_tokens": [1], "response_tokens": [2], "response_logprobs": [-0.1]},
                        },
                    )
                    assert turn.response_tokens == [2]
                    assert turn.response_logprobs == [-0.1]
                    assert any(
                        kind == "turn" and data["turn"] == value and data["sample_idx"] == item.sample_index
                        for kind, data in recorded
                    )
                    await asyncio.sleep(0)

            other = AgentItem(record={}, prompt="task", input_tokens=[1], prompt_index=0, sample_index=1)
            await asyncio.gather(task(item), task(other))

    asyncio.run(run())
    assert [(data["sample_idx"], data["turn"]) for kind, data in recorded if kind == "turn"] == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    ]
    count = len(recorded)
    AgentTrajectoryTurn(item=item, messages=[], input_tokens=[1], response_tokens=[2], response_logprobs=[-0.1])
    assert len(recorded) == count


def test_mst_reporter_survives_autograd_thread_context_and_emits_first_chunk(tmp_path):
    import torch

    from areno.engine.runtime.mst import mini_sequence_forward
    from areno.engine.runtime.observability import capture_training_events, event_reporter

    class Step:
        @capture_training_events
        def forward(self, module, values):
            return mini_sequence_forward(module, values, chunk_size=2)

    recorder = AgentEvents(tmp_path)
    try:
        step = Step()
        step._event_reporter = event_reporter(recorder, 3)
        module = torch.nn.Linear(4, 4)
        values = torch.randn(6, 4, requires_grad=True)
        actual = step.forward(module, values)
        # The reporting context has already exited before backward, just as
        # it does on autograd's worker threads.
        actual.sum().backward()
        progress = EventIndex(tmp_path).overview()["progress"]
        assert any(e["data"]["phase"] == "mst" and e["data"]["chunk_size"] == 2 for e in progress)
        assert all("microbatch" not in e["data"] for e in progress)
        expected_values = values.detach().clone().requires_grad_()
        module.zero_grad()
        module(expected_values).sum().backward()
        torch.testing.assert_close(values.grad, expected_values.grad)
    finally:
        recorder.close()
