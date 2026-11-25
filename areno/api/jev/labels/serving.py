"""Optional HTTP scoring of identical prompt IDs on a native SGLang CausalLM."""

import json
import math
import urllib.request


def score_sglang(endpoint, encoded, *, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    payload = {
        "input_ids": encoded["prompt_token_ids"],
        "token_ids_logprob": encoded["label_token_ids"],
        "sampling_params": {"max_new_tokens": 0},
        "return_logprob": True,
        "logprob_start_len": -1,
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    values = result["meta_info"]["output_token_ids_logprobs"][0]
    lookup = {int(item[1]): float(item[0]) for item in values}
    logprobs = [lookup[token] for token in encoded["label_token_ids"]]
    if not all(math.isfinite(value) for value in logprobs):
        raise ValueError("nonfinite SGLang label logprobs")
    maximum = max(logprobs)
    weights = [math.exp((value - maximum) / temperature) for value in logprobs]
    total = math.fsum(weights)
    return {
        "logprobs": logprobs,
        "probabilities": [value / total for value in weights],
        "label_mass": math.fsum(math.exp(value) for value in logprobs),
        "prompt_token_ids": encoded["prompt_token_ids"],
        "label_token_ids": encoded["label_token_ids"],
    }
