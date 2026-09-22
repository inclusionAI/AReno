"""Normalize self-contained pi coding tasks for AReno training."""


def load_training_dataset(dataset_path, *, default_loader, **kwargs):
    records = []
    for row in default_loader(dataset_path):
        record = dict(row)
        prompt = record.get("prompt") or record.get("problem_statement")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("pi task requires prompt or problem_statement")
        if not isinstance(record.get("files"), dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in record["files"].items()
        ):
            raise ValueError("pi task files must map relative paths to text")
        if not isinstance(record.get("verify"), str) or not record["verify"].strip():
            raise ValueError("pi task requires trusted Python verify code")
        record["prompt"] = prompt
        records.append(record)
    return records
