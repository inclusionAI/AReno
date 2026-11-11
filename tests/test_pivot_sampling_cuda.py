"""Differential support/probability tests; no cross-algorithm RNG equality."""

import pytest
import torch

from areno.engine.data import SamplingParams, sampling

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference(p, k, top_p):
    values, ids = p.sort(descending=True, stable=True)
    if k > 0:
        values[:, k:] = 0
    if top_p < 1:
        # FP64 exclusive sums avoid cancellation in cumsum(values)-values.
        prefix = values.double().cumsum(-1) - values.double()
        values[prefix > top_p] = 0
    values /= values.sum(-1, keepdim=True)
    return torch.zeros_like(p).scatter(1, ids, values)


@pytest.mark.parametrize("vocab", [32, 32768, 131072, 151936])
@pytest.mark.parametrize("k,p", [(1, 1.0), (50, 0.9), (0, 0.9), (0, 0.0), (999999, 1.0), (0, 1e-6)])
def test_probabilities(vocab, k, p):
    from areno.accel.kernels.sampling import truncate_probs

    torch.manual_seed(634)
    probs = torch.randn(4, vocab, device="cuda").softmax(-1)
    probs[1].fill_(1 / vocab)
    probs[2].zero_()
    probs[2, 0] = 1
    expected = reference(probs, k, p)
    actual = truncate_probs(probs, k, p)
    assert torch.equal(actual > 0, expected > 0)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_seed_graph_suppression_and_invalid(dtype):
    logits = torch.randn(8, 32768, device="cuda", dtype=dtype)
    logits[0].fill_(float("nan"))
    logits[1].fill_(-float("inf"))
    logits[2, 2] = float("inf")
    params = SamplingParams(temperature=0.7, top_k=50, top_p=0.9, suppress_token_ids=(2, 3), min_new_tokens=2, seed=634)

    def run(generator):
        return sampling._sample(logits, params, logits.device, generator=generator, eos_token_id=4, sample_step=0)

    a = run(sampling._make_sample_generator(params, logits.device))
    b = run(sampling._make_sample_generator(params, logits.device))
    assert torch.equal(a, b)
    assert not torch.isin(a, torch.tensor([2, 3, 4], device="cuda")).any()
    for _ in range(3):
        run(None)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run(None)
    graph.replay()
    torch.cuda.synchronize()
    assert (out >= 0).all() and (out < logits.shape[-1]).all()


def test_distribution():
    from areno.accel.kernels.sampling import truncate_probs

    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]], device="cuda")
    filtered = truncate_probs(probs, 3, 0.5)
    draws = torch.multinomial(filtered, 100000, replacement=True)
    freq = torch.bincount(draws[0], minlength=4) / 100000
    torch.testing.assert_close(freq, reference(probs, 3, 0.5)[0], atol=0.008, rtol=0)


@pytest.mark.parametrize("temperature", [0.01, 0.7, 2.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_processed_distribution(temperature, dtype, monkeypatch):
    logits = torch.randn(4, 32768, device="cuda", dtype=dtype)
    logits[0].fill_(float("nan"))
    logits[1, 0] = float("inf")
    logits[2].fill_(0)
    params = SamplingParams(temperature=temperature, top_k=50, top_p=0.9, suppress_token_ids=(0, 1), min_new_tokens=2)
    scores = torch.nan_to_num(
        logits.float(), nan=-float("inf"), posinf=torch.finfo(torch.float32).max, neginf=-float("inf")
    )
    scores[:, :3] = -float("inf")
    p = torch.softmax(scores / temperature, -1)
    p = torch.nan_to_num(p, nan=0.0)
    total = p.sum(-1, keepdim=True)
    fallback = torch.nn.functional.one_hot(scores.argmax(-1), p.shape[-1]).float()
    p = torch.where(total > 0, p / total.clamp_min(torch.finfo(p.dtype).tiny), fallback)
    expected = reference(p, 50, 0.9)
    seen = []

    def draw(probabilities, *args, **kwargs):
        seen.append(probabilities)
        return probabilities.argmax(-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", draw)
    sampling._sample(logits, params, logits.device, eos_token_id=2)
    assert torch.equal(seen[0] > 0, expected > 0)
    torch.testing.assert_close(seen[0], expected, atol=2e-7, rtol=2e-5)
