// Translate the shared launcher fields into the validated Modal workflow schema.
export function modalWorkflow(mode, config, catalog, settings) {
  const schema = mode === "train" ? catalog.train : catalog.serve;
  const aliases = { metrics_dir: "metrics_log_dir", save_dir: "save_path" };
  const params = {};
  if (String(config.extra_args || "").trim()) throw new Error("For Modal, enter options in the launcher fields instead of Extra args.");
  for (const [key, value] of Object.entries(config)) {
    const name = aliases[key] || key;
    const option = schema.find(item => item.name === name);
    if (!option || value === "" || value == null) continue;
    if (mode === "train" && option.algorithms && !option.algorithms.includes(config.algo)) continue;
    if (option.type === "bool") {
      if (typeof value === "boolean") params[name] = value;
      else if (["true", "false"].includes(String(value).toLowerCase())) params[name] = String(value).toLowerCase() === "true";
      else throw new Error(`${name} must be true or false.`);
    } else params[name] = value;
  }
  for (const name of ["save_path", "metrics_log_dir"]) {
    if (params[name] && !String(params[name]).startsWith("/")) params[name] = `/artifacts/${params[name]}`;
  }
  return {
    kind: mode === "train" ? "training" : "deployment",
    name: `${mode} ${mode === "train" ? config.ckpt : config.model_path}`,
    model: { adapter: settings.adapter, checkpoint: mode === "train" ? config.ckpt : config.model_path, ...(settings.parameters_billion ? { parameters_billion: Number(settings.parameters_billion) } : {}) },
    resources: { ...(settings.auto_gpu ? { auto_gpu: true } : {}), gpu: settings.gpu, count: settings.count, cpu: settings.cpu, memory_gib: settings.memory_gib, timeout_seconds: Math.round(settings.duration_hours * 3600) },
    estimate_hours: settings.duration_hours,
    ...(mode === "train" ? { stages: [{ algo: config.algo, params }] } : { serve: params }),
  };
}
