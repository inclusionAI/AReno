"""Separate controller process running official SWE-bench grading in DinD."""

import json
import sys
from pathlib import Path

from swe_images import proxy_test_spec


def main():
    import docker
    from swebench.harness.run_evaluation import run_instance
    from swebench.harness.test_spec.test_spec import make_test_spec

    job = json.loads(Path(sys.argv[1]).read_text())
    spec = proxy_test_spec(make_test_spec(job["instance"], namespace=job["namespace"], arch="x86_64"))
    with docker.from_env() as client:
        result = run_instance(
            spec,
            job["prediction"],
            rm_image=False,
            force_rebuild=False,
            client=client,
            run_id=job["run_id"],
            timeout=job["timeout"],
        )
    Path(sys.argv[2]).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
