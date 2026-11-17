"""Answer-position LM readout preserving the native head's dtype and layout."""


def answer_logits(hidden_states, positions, lm_head, *, logit_softcap=None):
    states = hidden_states.reshape(-1, hidden_states.shape[-1])[positions]
    # The Bailing head may remain FP32 while the backbone is BF16. Keep the
    # native 3D training layout, as in packed_next_token_logprobs_from_hidden.
    logits = lm_head(states.to(lm_head.weight.dtype).unsqueeze(0)).squeeze(0)
    if logit_softcap:
        import torch

        logits = float(logit_softcap) * torch.tanh(logits / float(logit_softcap))
    return logits


def score_prompt_labels(model, token_rows, label_rows, *, device, bi_attn=False):
    """No candidate answer tokens enter the backbone, including in bi-attn mode."""
    import torch

    from areno.engine.runtime.metadata import TrainMeta

    results = []
    for tokens, labels in zip(token_rows, label_rows, strict=True):
        if not tokens:
            raise ValueError("label scoring requires a nonempty prompt")
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        meta = TrainMeta(
            cu_seqlens=torch.tensor([0, len(tokens)], dtype=torch.int32, device=device),
            max_seqlen=len(tokens),
            packed=True,
            bi_attn=bi_attn,
        )
        output = model(
            input_ids=ids,
            position_ids=torch.arange(len(tokens), device=device)[None, :],
            train_meta=meta,
            defer_lm_head=True,
        )
        logits = answer_logits(
            output.hidden_states,
            torch.tensor([len(tokens) - 1], device=device),
            model.lm_head,
            logit_softcap=getattr(model, "final_logit_softcapping", None),
        )
        results.append(torch.log_softmax(logits.float(), dim=-1)[0, labels].tolist())
    return results
