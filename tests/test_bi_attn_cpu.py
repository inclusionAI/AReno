"""Bidirectional label attention references, packing, and leakage regressions."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from areno.api.jev.labels.attention import resolve_bi_attn
from areno.api.jev.labels.cuda import score_prompt_labels
from areno.api.jev.labels.protocol import save_protocol
from areno.engine.layers.attention_backend.train import FlashAttnTrainAttentionBackend
from areno.engine.runtime.metadata import TrainMeta


@pytest.mark.parametrize("length", [4, 515])
def test_bi_attn_packed_reference_and_gradients_cpu(length):
    torch.manual_seed(11)
    q = torch.randn(1, length + 3, 4, 4, requires_grad=True)
    k = torch.randn(1, length + 3, 2, 4, requires_grad=True)
    v = torch.randn(1, length + 3, 2, 4, requires_grad=True)
    meta = TrainMeta(
        cu_seqlens=torch.tensor([0, length, length + 3], dtype=torch.int32),
        max_seqlen=length,
        packed=True,
        bi_attn=True,
    )
    actual = FlashAttnTrainAttentionBackend("native")(q, k, v, meta)
    expected = []
    for a, b in [(0, length), (length, length + 3)]:
        qi, ki, vi = [x[:, a:b].transpose(1, 2) for x in [q, k, v]]
        expected.append(
            F.scaled_dot_product_attention(
                qi, ki.repeat_interleave(2, 1), vi.repeat_interleave(2, 1), is_causal=False
            ).transpose(1, 2)
        )
    expected = torch.cat(expected, dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    ga = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=1e-4, rtol=1e-4)


def test_future_visible_only_inside_own_prompt_cpu():
    q = k = torch.zeros(1, 5, 1, 2)
    v = torch.tensor([1.0, 3.0, 8.0, 100.0, 200.0]).view(1, 5, 1, 1).expand(1, 5, 1, 2)
    backend = FlashAttnTrainAttentionBackend("native")
    meta = TrainMeta(cu_seqlens=torch.tensor([0, 3, 5], dtype=torch.int32), max_seqlen=3)
    causal = backend(q, k, v, meta)
    meta.bi_attn = True
    both = backend(q, k, v, meta)
    assert causal[0, 0, 0, 0] == 1
    torch.testing.assert_close(both[0, 0, 0, 0], torch.tensor(4.0))
    torch.testing.assert_close(both[0, 3, 0, 0], torch.tensor(150.0))
    # A sliding-window layer remains causal despite the full-attention flag.
    local = backend(q, k, v, meta, window_size=(2, 0))
    assert local[0, 0, 0, 0] == 1


def test_label_scoring_never_appends_answer_cpu():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(20, 4)
            self.lm_head = torch.nn.Linear(4, 20, bias=False)
            self.seen = []

        def forward(self, input_ids, position_ids, train_meta, defer_lm_head):
            self.seen.append(input_ids.tolist())
            assert train_meta.bi_attn and defer_lm_head
            return SimpleNamespace(hidden_states=self.embedding(input_ids))

    m = Model()
    p = score_prompt_labels(m, [[2, 3, 4]], [[7, 8]], device="cpu", bi_attn=True)
    assert m.seen == [[[2, 3, 4]]]
    expected = m.lm_head(m.embedding(torch.tensor(4))).log_softmax(-1)[[7, 8]]
    torch.testing.assert_close(torch.tensor(p[0]), expected)


def test_protocol_attention_roundtrip_cpu(tmp_path):
    tok = SimpleNamespace(chat_template="template")
    save_protocol(tmp_path, tok, temperature=1, max_seq_len=4096, bi_attn=True)
    assert resolve_bi_attn(tmp_path) is True
    assert resolve_bi_attn(tmp_path, True) is True
    with pytest.raises(ValueError, match="causal"):
        resolve_bi_attn(tmp_path, False)
    file = tmp_path / "label_jev_config.json"
    a = json.loads(file.read_text())
    a.pop("bi_attn")
    file.write_text(json.dumps(a))
    assert resolve_bi_attn(tmp_path) is False
    assert resolve_bi_attn(tmp_path, True) is True
    a["bi_attn"] = "false"
    file.write_text(json.dumps(a))
    with pytest.raises(ValueError, match="boolean"):
        resolve_bi_attn(tmp_path)


@pytest.mark.parametrize("backend", ["cuda", "mlx"])
def test_bi_attn_cli_config_and_checkpoint_reload_cpu(tmp_path, backend):
    import click

    from areno.api.jev.labels.config import LabelJevTrainerConfig
    from areno.cli.train import _trainer_config_from_options, train_command

    args = [
        "--sgl-jev",
        "--bi-attn",
        "--backend",
        backend,
        "--ckpt",
        "unused",
        "--dataset-path",
        "unused",
        "--save-path",
        str(tmp_path),
        "--tp-size",
        "1",
        "--world-size",
        "1",
    ]
    with train_command.make_context("train", args) as ctx:
        config = _trainer_config_from_options(**ctx.params)
    assert isinstance(config, LabelJevTrainerConfig) and config.bi_attn is True
    save_protocol(tmp_path, SimpleNamespace(chat_template="t"), temperature=1, max_seq_len=64, bi_attn=True)
    config = LabelJevTrainerConfig(
        algo="label_jev",
        ckpt=str(tmp_path),
        dataset_path="unused",
        save_path=str(tmp_path / "out"),
        backend=backend,
        world_size=1,
        tp_size=1,
    )
    assert config.bi_attn is True
    with pytest.raises(click.UsageError, match="requires"):
        with train_command.make_context("train", ["--bi-attn", "--ckpt", "unused", "--dataset-path", "unused"]) as ctx:
            _trainer_config_from_options(**ctx.params)


@pytest.mark.parametrize("nested", [False, True])
def test_full_checkpoint_attention_config_cpu(tmp_path, nested):
    file = tmp_path / "config.json"
    file.write_text(json.dumps({"text_config": {}} if nested else {}))
    save_protocol(tmp_path, SimpleNamespace(chat_template="t"), temperature=1, max_seq_len=64, bi_attn=True)
    config = json.loads(file.read_text())
    assert config.get("text_config", config)["is_causal"] is False
    assert resolve_bi_attn(tmp_path) is True
    (tmp_path / "label_jev_config.json").unlink()
    assert resolve_bi_attn(tmp_path) is True
