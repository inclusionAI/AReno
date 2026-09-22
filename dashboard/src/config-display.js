const items = (value) => Object.entries(value || {})
  .filter(([, entry]) => entry !== undefined && entry !== null && entry !== "")
  .map(([key, entry]) => ({ key, value: entry }));

export function normalizeConfigSections(settings) {
  if (Array.isArray(settings?.sections)) {
    return settings.sections
      .map(section => ({ title: section.title || "Config", items: (section.items || []).filter(({ value }) => value !== undefined && value !== null && value !== "") }))
      .filter(section => section.items.length > 0);
  }
  if (Array.isArray(settings?.stages)) {
    const { stages, model, resources, serve, ...general } = settings;
    return [
      { title: "Launch", items: items(general) },
      { title: "Model", items: items(model) },
      { title: "Resources", items: items(resources) },
      ...stages.map((stage, index) => {
        const { params, ...metadata } = stage || {};
        return { title: `Stage ${index + 1}${stage?.algo ? ` · ${stage.algo.toUpperCase()}` : ""}`, items: items({ ...metadata, ...params }) };
      }),
      { title: "Serving", items: items(serve) },
    ].filter(section => section.items.length > 0);
  }
  const entries = items(settings);
  return entries.length ? [{ title: "Launch", items: entries }] : [];
}

export function formatConfigValue(value) {
  if (Array.isArray(value)) return value.map(formatConfigValue).join(" · ");
  if (value !== null && typeof value === "object") return JSON.stringify(value);
  return String(value);
}
