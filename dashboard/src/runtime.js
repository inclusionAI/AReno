export function runtimeSummary(report = {}) {
  if ("metal" in report) {
    const available = report.metal.available;
    return {
      label: "MLX / Metal",
      value: `${report.dependencies?.mlx?.version || "n/a"} / Metal`,
      detail: available ? "Metal runtime available" : report.metal.error || "Metal runtime unavailable",
      tone: available ? "ok" : "warn",
    };
  }
  const torch = report.torch || {};
  return {
    label: "PyTorch / CUDA",
    value: `${torch.version || "n/a"} / ${torch.cuda_runtime || torch.cuda_build || "n/a"}`,
    detail: torch.cuda_available ? "Compatible runtime detected" : "CUDA runtime unavailable",
    tone: torch.cuda_available ? "ok" : "warn",
  };
}

export function runtimeFacts(report = {}) {
  const torch = report.torch || {};
  const cuda = report.cuda || {};
  const deps = report.dependencies || {};
  return [
    ["AReno", report.areno?.version],
    ["Python", report.python?.version],
    ...("metal" in report ? [
      ["MLX", deps.mlx?.version],
      ["MLX-LM", deps.mlx_lm?.version],
      ["MLX-VLM", deps.mlx_vlm?.version],
      ["Metal available", report.metal.available],
      ["Metal error", report.metal.error],
    ] : [
      ["PyTorch", torch.version],
      ["CUDA build", torch.cuda_build],
      ["CUDA runtime", torch.cuda_runtime],
      ["CUDA available", torch.cuda_available],
      ["Visible GPUs", torch.device_count],
      ["NVCC", cuda.nvcc?.version || cuda.nvcc?.path],
      ["NVIDIA driver", cuda.driver?.driver_version],
      ["Driver CUDA", cuda.driver?.cuda_version],
    ]),
    ["Platform", report.platform?.platform],
  ].filter(([, value]) => value !== undefined && value !== null && value !== "");
}
