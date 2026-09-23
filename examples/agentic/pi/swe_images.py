"""Apply the image proxy to actual build references in the SWE-bench example."""

import os
import re
from dataclasses import asdict


def docker_platform(arch):
    return {"x86_64": "linux/amd64", "arm64": "linux/arm64/v8"}[arch]


def docker_image(image):
    proxy = os.environ.get("ARENO_PI_DOCKER_PROXY", "").strip().rstrip("/")
    if "://" in proxy or any(char.isspace() for char in proxy):
        raise ValueError("ARENO_PI_DOCKER_PROXY must be an image prefix such as v4.gh-proxy.org/docker")
    return f"{proxy}/{image}" if proxy else image


def ensure_native_image(client, name, arch):
    """Replace a tag cached for another CPU before the classic builder uses it."""
    import docker

    platform = docker_platform(arch)
    expected_arch = {"x86_64": "amd64", "arm64": "arm64"}[arch]

    def matches(image):
        return image.attrs.get("Os") == "linux" and image.attrs.get("Architecture") == expected_arch

    try:
        cached = client.images.get(name)
    except docker.errors.ImageNotFound:
        cached = None
    if cached is not None and matches(cached):
        return cached

    print(f"[pi/DinD] Pulling {name} for {platform}", flush=True)
    client.images.pull(name, platform=platform)
    # Reinspect the tag the builder will resolve, not just the pull response.
    pulled = client.images.get(name)
    if not matches(pulled):
        actual = f"{pulled.attrs.get('Os')}/{pulled.attrs.get('Architecture')}"
        raise RuntimeError(
            f"Image {name} is {actual} after an explicit {platform} pull. "
            "The registry/proxy or Docker image store did not provide the requested architecture; "
            "refusing to build with this image."
        )
    return pulled


def proxy_test_spec(spec):
    from swebench.harness.test_spec.test_spec import TestSpec

    use_proxy = docker_image("ubuntu") != "ubuntu"

    class ProxyTestSpec(TestSpec):
        @property
        def platform(self):
            return docker_platform(self.arch)

        @property
        def base_dockerfile(self):
            if not use_proxy:
                return super().base_dockerfile
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
