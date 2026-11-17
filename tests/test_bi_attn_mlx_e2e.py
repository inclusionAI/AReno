"""Public SDK train -> full checkpoint -> native label serving on MLX."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")


def test_bi_attn_mlx_sdk_checkpoint_serve(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from areno import Trainer
    from areno.api.algorithms import label_jev_loss_fn
    from areno.api.jev.labels.config import LabelJevMlxConfig
    from areno.api.jev.labels.mlx import prompt_logits
    from areno.api.jev.labels.protocol import encode_question, save_protocol, training_row
    from areno.api.jev.labels.serve import LabelDecisionScorer
    from areno.api.models import BackendType
    from tests.test_jev_mlx_full_e2e import make_source

    source = tmp_path / "source"
    make_source(source)
    vocab = ["[UNK]", "[PAD]", "[EOS]", "state", "A", "B", "yes", "no", "ASSISTANT", ":"]
    tok = Tokenizer(models.WordLevel(dict(zip(vocab, range(len(vocab)))), unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    tokenizer.chat_template = "{{ messages[-1]['content'] }}\nASSISTANT:"
    tokenizer.save_pretrained(source)
    config = LabelJevMlxConfig(gradient_checkpointing=False, optimizer={"lr": 1e-3})
    trainer = Trainer(1, str(source), backend_type=BackendType.MLX, custom_config=config)
    scorer = None
    try:
        trainer.init()
        question = {"type": "choice", "instructions": "Pick.", "criteria": {"left": "left", "right": "right"}}
        encoded = encode_question("state", question, tokenizer, target={"left": 0.3, "right": 0.7})
        metrics = trainer.train([training_row(encoded, bi_attn=True)], label_jev_loss_fn, mini_bs=1)
        assert np.isfinite(metrics["loss"])
        checkpoint = tmp_path / "checkpoint"
        trainer.save_checkpoint(str(checkpoint))
        save_protocol(checkpoint, tokenizer, temperature=1, max_seq_len=4096, bi_attn=True)
        ids = mx.array([encoded["prompt_token_ids"]])
        logits = prompt_logits(trainer._backend.model, ids, [ids.shape[1]], bi_attn=True)[0, -1].astype(mx.float32)
        expected = (logits[mx.array(encoded["label_token_ids"])] - mx.logsumexp(logits)).tolist()
        scorer = LabelDecisionScorer(str(checkpoint), backend_type=BackendType.MLX)
        assert scorer.bi_attn is True
        np.testing.assert_allclose(scorer.label_logprobs(encoded), expected, atol=1e-5)
    finally:
        if scorer is not None:
            scorer.close()
        trainer.close()
