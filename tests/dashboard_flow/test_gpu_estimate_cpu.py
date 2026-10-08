import copy

import pytest

from areno.dashboard.flow.gpu_estimate import recommend
from areno.dashboard.flow.pricing import Pricing


def workflow(checkpoint="org/model-7B", **params):
    return {"model": {"checkpoint": checkpoint}, "stages": [{"algo": "sft", "params": params}]}


def estimate(request):
    return recommend(request, Pricing().cache)


def test_adam4_state_is_smaller_than_adam8_and_fp32_but_keeps_master_weights():
    values = [estimate(workflow(**params)) for params in ({}, {"adam_8bit": True}, {"adam_4bit": False})]
    assert values[0]["required_gib_per_gpu"] < values[1]["required_gib_per_gpu"] < values[2]["required_gib_per_gpu"]
    state = values[0]["stages"][0]
    assert state["optimizer"] == "adam_4bit"
    assert state["components_gib"]["master_weights"] > 0
    assert state["components_gib"]["weights"] > state["components_gib"]["optimizer"]
    assert values[0]["recommended"]["memory_gib_per_gpu"] >= values[0]["required_gib_per_gpu"]
    assert values[0]["recommended"]["gpu_hourly_cost"] == min(
        item["gpu_hourly_cost"] for item in values[0]["candidates"]
    )


def test_sequence_microbatch_and_rollout_concurrency_increase_estimate():
    base = estimate(workflow())
    assert estimate(workflow(max_context_len=8192, mini_bs=4))["required_gib_per_gpu"] > base["required_gib_per_gpu"]
    request = workflow(max_running_prompts=1)
    request["stages"][0]["algo"] = "grpo"
    one = estimate(request)
    request["stages"][0]["params"]["max_running_prompts"] = 32
    assert estimate(request)["required_gib_per_gpu"] > one["required_gib_per_gpu"]


def test_tensor_parallel_shards_memory_but_data_parallel_does_not():
    base = estimate(workflow())
    dp = estimate(workflow(world_size=2))
    tp = estimate(workflow(world_size=2, tp_size=2))
    assert dp["required_gib_per_gpu"] == base["required_gib_per_gpu"]
    assert tp["required_gib_per_gpu"] < base["required_gib_per_gpu"]
    assert tp["recommended"]["count"] == 2


@pytest.mark.parametrize(
    "name,billions", [("Qwen3-0.6B", 0.6), ("Qwen3-30B-A3B", 30), ("Mixtral-8x7B", 56), ("model-500M", 0.5)]
)
def test_name_uses_total_parameters_including_experts(name, billions):
    assert estimate(workflow(name))["parameters_billion"] == billions


def test_unknown_size_and_insufficient_capacity_are_explicit():
    request = workflow("custom/model")
    assert not estimate(request)["available"]
    request["model"]["parameters_billion"] = 7
    assert estimate(request)["available"]
    assert not estimate(workflow("model-1000B"))["available"]


def test_serving_excludes_optimizer_and_workflow_uses_peak_stage():
    request = workflow()
    serve = {"kind": "deployment", "model": request["model"], "serve": {"max_running_requests": 1}}
    assert estimate(serve)["stages"][0]["optimizer"] == "none"
    assert "optimizer" not in estimate(serve)["stages"][0]["components_gib"]
    request["stages"].append({"algo": "grpo", "params": {"max_running_prompts": 32}})
    before = copy.deepcopy(request)
    result = estimate(request)
    assert result["required_gib_per_gpu"] == round(max(s["required_gib_per_gpu"] for s in result["stages"]), 2)
    assert request == before


@pytest.mark.parametrize("params", [{"tp_size": 1.5}, {"world_size": 9}, {"max_context_len": -1}, {"mini_bs": "nan"}])
def test_invalid_memory_inputs_rejected(params):
    with pytest.raises(ValueError):
        estimate(workflow(**params))
