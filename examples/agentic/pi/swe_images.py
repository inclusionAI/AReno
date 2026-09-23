"""Apply the image proxy to actual build references in the SWE-bench example."""

import os
import re
from dataclasses import asdict


def docker_image(image):
    proxy = os.environ.get("ARENO_PI_DOCKER_PROXY", "").strip().rstrip("/")
    if "://" in proxy or any(char.isspace() for char in proxy):
        raise ValueError("ARENO_PI_DOCKER_PROXY must be an image prefix such as v4.gh-proxy.org/docker")
    return f"{proxy}/{image}" if proxy else image


def proxy_test_spec(spec):
    from swebench.harness.test_spec.test_spec import TestSpec

    if docker_image("ubuntu") == "ubuntu":
        return spec

    class ProxyTestSpec(TestSpec):
        @property
        def base_dockerfile(self):
            dockerfile, count = re.subn(
                r"(?m)^(FROM\s+(?:--platform=\S+\s+)?)(ubuntu:\S+)",
                lambda match: match[1] + docker_image(match[2]),
                super().base_dockerfile,
            )
            if count != 1:
                raise ValueError("Expected one Ubuntu FROM in the pinned SWE-bench base Dockerfile")
            return dockerfile

    # Preserve the harness's scripts, cache keys and isinstance(TestSpec) contract.
    return ProxyTestSpec(**asdict(spec))
