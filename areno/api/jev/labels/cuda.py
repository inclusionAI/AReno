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
