"""Load generated SWE-bench/DinD tasks; test metadata stays outside the agent."""


def load_training_dataset(dataset_path, *, default_loader, **kwargs):
    rows = list(default_loader(dataset_path))
    for row in rows:
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError("SWE task requires prompt")
        instance = row.get("swebench")
        if not isinstance(instance, dict) or not instance.get("instance_id") or not instance.get("FAIL_TO_PASS"):
            raise ValueError("SWE task requires instance metadata and FAIL_TO_PASS tests")
    return rows
