import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Activity,
  ArrowLeft,
  Bot,
  Box,
  CircleStop,
  Cpu,
  Database,
  FileText,
  Film,
  History,
  Image as ImageIcon,
  LayoutDashboard,
  Languages,
  Layers,
  MessageSquare,
  Moon,
  Plus,
  Play,
  RefreshCw,
  Send,
  Server,
  Settings2,
  Sun,
  TerminalSquare,
  Timer,
  Volume2,
  Wrench,
} from "lucide-react";
import "./styles.css";

function dashboardBasePath() {
  const moduleScript = document.querySelector('script[type="module"][src]');
  if (moduleScript?.src) return new URL("../", moduleScript.src).pathname;
  const pathname = window.location.pathname;
  return pathname.endsWith("/") ? pathname : `${pathname}/`;
}

const API_BASE = dashboardBasePath();
const UI_LANGUAGE_STORAGE_KEY = "areno-dashboard-language";
const ZH_UI = {
  "Overview": "概览",
  "Jobs": "任务",
  "Runtime": "运行环境",
  "Launcher": "任务启动",
  "Agent": "智能助手",
  "Operations Overview": "运行概览",
  "Runtime health, active work, and the signals that need attention.": "查看运行环境、活跃任务和需要关注的信号。",
  "Runtime Environment": "运行环境",
  "Review areno check, areno env, dependencies, GPU state, and repository context.": "检查 AReno 环境、依赖、GPU 状态和仓库上下文。",
  "Task Launcher": "任务启动",
  "Start low-intrusion AReno train or serve subprocesses from explicit configs.": "通过明确配置启动 AReno 训练或服务进程。",
  "Agent Console": "智能助手",
  "Chat with an operations agent using the selected job context.": "基于所选任务上下文与运维助手对话。",
  "Running Job Summary": "运行任务摘要",
  "Current health, route, and stage progression for the latest job.": "最新任务的健康状态、运行路径和阶段进度。",
  "Metrics": "指标",
  "Switch between reward and loss for the latest job.": "切换查看最新任务的奖励、损失及训练指标。",
  "Runtime attention": "运行环境提醒",
  "Highest-priority environment finding.": "当前最高优先级的环境问题。",
  "Quick actions": "快捷操作",
  "Short paths into common workflows.": "快速执行常用工作流。",
  "Latest health": "最新健康指标",
  "Active jobs": "活跃任务",
  "No signal": "暂无信号",
  "Waiting for metrics": "等待指标上报",
  "Reward": "奖励",
  "Loss": "损失",
  "Grad Norm": "梯度范数",
  "Sequence Length": "序列长度",
  "Job Detail: Overview": "任务详情：概览",
  "Rollout Sample": "采样样例",
  "Environment Checks": "环境检查",
  "Runtime requirements, compatibility, and actionable diagnostics.": "运行要求、兼容性和可执行诊断。",
  "GPU Cards": "GPU 状态",
  "Memory pressure and utilization before launch.": "启动前的显存压力和利用率。",
  "Details": "详情",
  "Run Check": "运行检查",
  "Fix": "修复",
  "Starting...": "正在启动…",
  "Installing...": "正在安装…",
  "Installed": "已安装",
  "Retry fix": "重试修复",
  "Ready": "就绪",
  "Needs attention": "需要关注",
  "Checking": "检查中",
  "Dependency Risk": "依赖风险",
  "No checks reported": "暂无检查结果",
  "No GPUs reported": "暂无 GPU 信息",
  "No metrics available": "暂无指标",
  "No rollout sample captured yet.": "尚未记录采样样例。",
  "No TensorBoard scalar points loaded yet.": "尚未加载 TensorBoard 标量数据。",
  "Loading selected metric...": "正在加载所选指标…",
  "Open": "打开",
  "Back to Jobs": "返回任务列表",
  "Stop job": "停止任务",
  "Stop": "停止",
  "Prev": "上一页",
  "Next": "下一页",
  "All": "全部",
  "Running": "运行中",
  "Failed": "失败",
  "Stopped": "已停止",
  "Config": "配置",
  "Logs": "日志",
  "Command Preview": "命令预览",
  "Preflight": "预检",
  "Start Train": "启动训练",
  "Start Serve": "启动服务",
  "Send": "发送",
  "New Chat": "新对话",
  "Settings": "设置",
  "Chat": "对话",
  "History": "历史",
  "Error Recovery": "错误恢复",
  "Suggested Follow-ups": "建议追问",
  "Done": "完成",
  "Toggle theme": "切换主题",
  "Translate to Chinese": "切换为中文",
  "Translate to English": "切换为英文",
  "Ask about this job, its metrics, or recent logs...": "询问此任务的指标、运行状态或最近日志…",
  "Ask about the runtime or describe a task to launch...": "询问运行环境，或描述要启动的任务…",
  "Message the operations agent": "向运维助手发送消息",
};
const EN_UI = Object.fromEntries(Object.entries(ZH_UI).map(([english, chinese]) => [chinese, english]));

async function api(path, options) {
  const response = await fetch(`${API_BASE}${path.replace(/^\//, "")}`, {
    headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
    ...options,
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(data.error || response.statusText);
  }
  return data;
}

function classNames(...items) {
  return items.filter(Boolean).join(" ");
}

function translateDashboard(root, language) {
  if (!root) return;
  const dictionary = language === "zh" ? ZH_UI : EN_UI;
  const excluded = "pre, code, .mono, .chatMessages, .runtimeCommandResult, .commandPreview";
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let node = walker.nextNode();
  while (node) {
    const parent = node.parentElement;
    if (parent && !parent.closest(excluded)) {
      const trimmed = node.nodeValue.trim();
      const translated = dictionary[trimmed];
      if (translated) node.nodeValue = node.nodeValue.replace(trimmed, translated);
    }
    node = walker.nextNode();
  }
  for (const element of root.querySelectorAll("[title], [placeholder], [aria-label]")) {
    if (element.closest(excluded)) continue;
    for (const attribute of ["title", "placeholder", "aria-label"]) {
      const value = element.getAttribute(attribute);
      if (value && dictionary[value]) element.setAttribute(attribute, dictionary[value]);
    }
  }
}

function usePolling(loader, delay = 2500, deps = []) {
  const [data, setData] = useState(null);
  const refresh = async () => {
    try {
      const value = await loader();
      setData(value);
    } catch (err) {
      // Polling can race dashboard restarts or proxy reconnects. Keep the last
      // good snapshot instead of surfacing noisy transient fetch failures.
    }
  };
  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, delay);
    return () => clearInterval(timer);
  }, deps);
  return { data, refresh };
}

const defaultAgentMessages = [
  { role: "assistant", content: "Select a job, then ask about metrics, runtime, logs, or how to start the next AReno task." },
];
const zhAgentMessages = [
  { role: "assistant", content: "选择一个任务，然后询问指标、运行环境、日志，或如何启动下一个 AReno 任务。" },
];

const AGENT_CHAT_STORAGE_KEY = "areno-dashboard-agent-chat";
const AGENT_SESSIONS_STORAGE_KEY = "areno-dashboard-agent-chat-sessions";
const AGENT_ACTIVE_SESSION_STORAGE_KEY = "areno-dashboard-agent-active-chat";
const AGENT_DRAFT_STORAGE_KEY = "areno-dashboard-agent-draft";
const AGENT_FAILURE_STORAGE_KEY = "areno-dashboard-agent-failure";

function loadAgentMessages() {
  try {
    const parsed = JSON.parse(localStorage.getItem(AGENT_CHAT_STORAGE_KEY) || "[]");
    return Array.isArray(parsed) && parsed.length ? parsed : defaultAgentMessages;
  } catch {
    return defaultAgentMessages;
  }
}

function createAgentSession(messages = defaultAgentMessages, title = "New chat") {
  const now = Date.now();
  return {
    id: `chat-${now}-${Math.random().toString(16).slice(2)}`,
    title,
    createdAt: now,
    updatedAt: now,
    messages,
    followUps: [],
  };
}

function loadAgentSessions() {
  try {
    const parsed = JSON.parse(localStorage.getItem(AGENT_SESSIONS_STORAGE_KEY) || "[]");
    if (Array.isArray(parsed) && parsed.length) {
      return parsed.map((session) => ({ ...session, messages: session.messages?.length ? session.messages : defaultAgentMessages, followUps: Array.isArray(session.followUps) ? session.followUps : [] }));
    }
  } catch {
    // Fall through to migrate the legacy single-chat storage.
  }
  return [createAgentSession(loadAgentMessages(), "Default chat")];
}

function inferAgentSessionTitle(messages, fallback = "New chat") {
  const firstUser = messages.find((message) => message.role === "user" && message.content);
  if (!firstUser) return fallback;
  const title = firstUser.content.replace(/\s+/g, " ").trim();
  return title.length > 42 ? `${title.slice(0, 42)}...` : title;
}

const defaultTrainConfig = {
  ckpt: "",
  dataset_path: "",
  dataset_loader_fn: "",
  reward_fn_path: "",
  ref_ckpt: "",
  reward_ckpt: "",
  critic_ckpt: "",
  agent_fn: "",
  algo: "sft",
  model_hub: "modelscope",
  epochs: 10,
  max_steps: 5,
  world_size: 1,
  tp_size: 1,
  attn_backend: "flash",
  activation_checkpointing: true,
  drop_rollout_state: false,
  eager_decode: false,
  disable_thinking: false,
  batch_size: 8,
  n_samples: 8,
  mini_bs: 1,
  score_micro_bs: 8,
  gradient_accumulation_steps: "",
  max_prompt_tokens: 1024,
  max_context_len: 2048,
  max_new_tokens: 1024,
  max_running_prompts: "",
  temperature: 1,
  top_k: -1,
  top_p: 1,
  greedy: false,
  agent_timeout_s: 300,
  train_tool_results: false,
  lr: 1.0e-6,
  min_lr: 1.0e-7,
  lr_decay_steps: 1000,
  lr_decay_style: "cosine",
  adam_beta1: 0.9,
  adam_beta2: 0.999,
  adam_8bit: false,
  unfreeze_multimodal_tower: false,
  unfreeze_multimodal_projector: false,
  multimodal_tower_lr: "",
  multimodal_tower_min_lr: "",
  multimodal_tower_lr_decay_steps: "",
  multimodal_tower_lr_decay_style: "",
  multimodal_projector_lr: "",
  multimodal_projector_min_lr: "",
  multimodal_projector_lr_decay_steps: "",
  multimodal_projector_lr_decay_style: "",
  weight_decay: 1.0e-2,
  grad_clip_norm: 1,
  gspo_clip_eps: 3.0e-4,
  grpo_clip_eps: 0.2,
  dpo_beta: 0.1,
  critic_warmup_steps: 20,
  critic_lr: 1.0e-5,
  use_kl_loss: true,
  kl_loss_coef: 0.001,
  kl_loss_type: "low_var_kl",
  clip_eps: 0.2,
  clip_ratio_c: 3,
  value_clip_eps: 0.5,
  value_loss_coef: 0.5,
  gamma: 1,
  lam: 0.95,
  tune_params: false,
  mem_frac: 0.9,
  tune_max_samples: 256,
  save_path: "outputs/dashboard-run",
  save_interval: 100,
  metrics_dir: "outputs/dashboard-run/metrics",
  extra_args: "",
};

const defaultServeConfig = {
  model_path: "",
  model_hub: "modelscope",
  host: "0.0.0.0",
  port: 8000,
  world_size: 1,
  tp_size: 1,
  max_running_prompts: 16,
  default_max_tokens: 1024,
  decode_progress_interval_s: 0,
  attn_backend: "flash",
  eager_decode: false,
  disable_thinking: false,
  extra_args: "",
};

function App() {
  const [selectedJobId, setSelectedJobId] = useState(null);
  const [trainConfig, setTrainConfig] = useState(defaultTrainConfig);
  const [serveConfig, setServeConfig] = useState(defaultServeConfig);
  const [agentPrompt, setAgentPrompt] = useState(() => localStorage.getItem(AGENT_DRAFT_STORAGE_KEY) || "");
  const [agentSessions, setAgentSessions] = useState(() => loadAgentSessions());
  const [activeAgentSessionId, setActiveAgentSessionId] = useState(() => localStorage.getItem(AGENT_ACTIVE_SESSION_STORAGE_KEY) || "");
  const [agentChatTab, setAgentChatTab] = useState("chat");
  const [agentProvider, setAgentProvider] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("areno-dashboard-agent-provider") || "{}");
    } catch {
      return {};
    }
  });
  const [activePage, setActivePage] = useState("overview");
  const [jobFilter, setJobFilter] = useState("all");
  const [launcherMode, setLauncherMode] = useState("train");
  const [theme, setTheme] = useState(() => localStorage.getItem("areno-dashboard-theme-v2") || "light");
  const [language, setLanguage] = useState(() => localStorage.getItem(UI_LANGUAGE_STORAGE_KEY) || "en");
  const [busy, setBusy] = useState("");
  const [agentSettingsOpen, setAgentSettingsOpen] = useState(false);
  const [jobPage, setJobPage] = useState(1);
  const [jobDetailOpen, setJobDetailOpen] = useState(false);
  const [refreshNonce, setRefreshNonce] = useState(0);
  const [followUpsLoading, setFollowUpsLoading] = useState(false);
  const [agentFailure, setAgentFailure] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(AGENT_FAILURE_STORAGE_KEY) || "null");
    } catch {
      return null;
    }
  });
  const [agentRecovering, setAgentRecovering] = useState(false);
  const [runtimeCheckResult, setRuntimeCheckResult] = useState(null);
  const [runtimeRepair, setRuntimeRepair] = useState(null);
  const chatMessagesRef = useRef(null);
  const env = usePolling(() => api("/api/env"), 5000);
  const jobs = usePolling(() => api("/api/jobs"), 2000);
  const jobDetail = usePolling(() => selectedJobId ? api(`/api/jobs/${selectedJobId}`) : Promise.resolve(null), 3000, [selectedJobId]);
  const runtimeAttention = usePolling(() => api("/api/runtime/attention"), 5000);
  const runtimeRepairDetail = usePolling(
    () => runtimeRepair?.jobId ? api(`/api/jobs/${runtimeRepair.jobId}`) : Promise.resolve(null),
    1000,
    [runtimeRepair?.jobId],
  );
  const quickActions = usePolling(() => api("/api/quick-actions"), 30000);
  const launcherPresets = usePolling(() => api("/api/launcher/presets"), 30000);
  const agentRecoveryState = usePolling(
    () => api(`/api/agent/recovery${selectedJobId ? `?job_id=${encodeURIComponent(selectedJobId)}` : ""}`),
    3000,
    [selectedJobId],
  );

  const jobList = (jobs.data?.jobs || []).filter((job) => job.kind !== "runtime-repair");
  const runtimeRepairJob = runtimeRepairDetail.data?.job || runtimeRepair?.job || null;
  const filteredJobs = jobFilter === "all" ? jobList : jobList.filter((job) => job.status === jobFilter);
  const jobPageSize = 3;
  const jobPageCount = Math.max(1, Math.ceil(filteredJobs.length / jobPageSize));
  const currentJobPage = Math.min(jobPage, jobPageCount);
  const pagedJobs = filteredJobs.slice((currentJobPage - 1) * jobPageSize, currentJobPage * jobPageSize);
  const selectedJob = jobDetail.data?.job || (selectedJobId ? jobList.find((job) => job.id === selectedJobId) : null) || null;
  const latestJob = newestJob(jobList);
  const latestJobPreviewDetail = usePolling(
    () => latestJob ? api(`/api/jobs/${latestJob.id}`) : Promise.resolve(null),
    3000,
    [latestJob?.id],
  );
  const previewJob = latestJob && latestJobPreviewDetail.data?.job?.id === latestJob.id
    ? latestJobPreviewDetail.data.job
    : latestJob;
  const activeAgentSession = useMemo(() => {
    return agentSessions.find((session) => session.id === activeAgentSessionId) || agentSessions[0] || createAgentSession();
  }, [agentSessions, activeAgentSessionId]);
  const agentMessages = activeAgentSession.messages || defaultAgentMessages;
  const agentFollowUps = activeAgentSession.followUps || [];

  useEffect(() => {
    if (selectedJobId && jobList.length && !jobList.some((job) => job.id === selectedJobId)) {
      setSelectedJobId(null);
    }
  }, [jobList.length, selectedJobId]);

  useEffect(() => {
    if (jobPage > jobPageCount) setJobPage(jobPageCount);
  }, [jobPage, jobPageCount]);

  useEffect(() => {
    const status = String(runtimeRepairJob?.status || "").toLowerCase();
    if (!runtimeRepair || runtimeRepair.refreshed || !["succeeded", "failed", "stopped"].includes(status)) return;
    setRuntimeRepair((current) => current ? { ...current, refreshed: true } : current);
    void (async () => {
      try {
        await api("/api/runtime/refresh", { method: "POST", body: "{}" });
        await Promise.all([env.refresh(), runtimeAttention.refresh(), jobs.refresh()]);
      } catch (err) {
        setBusy(err.message || String(err));
      }
    })();
  }, [runtimeRepair, runtimeRepairJob?.status]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("areno-dashboard-theme-v2", theme);
  }, [theme]);

  useEffect(() => {
    const root = document.getElementById("root");
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
    localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, language);
    translateDashboard(root, language);
    const observer = new MutationObserver(() => translateDashboard(root, language));
    if (root) observer.observe(root, { childList: true, subtree: true, characterData: true });
    return () => observer.disconnect();
  }, [language]);

  useEffect(() => {
    localStorage.setItem("areno-dashboard-agent-provider", JSON.stringify(agentProvider));
  }, [agentProvider]);

  useEffect(() => {
    localStorage.setItem(AGENT_SESSIONS_STORAGE_KEY, JSON.stringify(agentSessions.slice(-40)));
  }, [agentSessions]);

  useEffect(() => {
    if (!agentSessions.length) {
      const session = createAgentSession(language === "zh" ? zhAgentMessages : defaultAgentMessages, language === "zh" ? "新对话" : "New chat");
      setAgentSessions([session]);
      setActiveAgentSessionId(session.id);
      return;
    }
    if (!agentSessions.some((session) => session.id === activeAgentSessionId)) {
      setActiveAgentSessionId(agentSessions[0].id);
    }
  }, [agentSessions, activeAgentSessionId, language]);

  useEffect(() => {
    if (activeAgentSessionId) {
      localStorage.setItem(AGENT_ACTIVE_SESSION_STORAGE_KEY, activeAgentSessionId);
    }
  }, [activeAgentSessionId]);

  useEffect(() => {
    localStorage.setItem(AGENT_DRAFT_STORAGE_KEY, agentPrompt);
  }, [agentPrompt]);

  useEffect(() => {
    if (agentFailure) localStorage.setItem(AGENT_FAILURE_STORAGE_KEY, JSON.stringify(agentFailure));
    else localStorage.removeItem(AGENT_FAILURE_STORAGE_KEY);
  }, [agentFailure]);

  useEffect(() => {
    const node = chatMessagesRef.current;
    if (node) {
      node.scrollTop = node.scrollHeight;
    }
  }, [agentMessages, agentChatTab]);

  const pages = [
    { id: "overview", label: "Overview", icon: <LayoutDashboard size={17} /> },
    { id: "jobs", label: "Jobs", icon: <Activity size={16} /> },
    { id: "runtime", label: "Runtime", icon: <Server size={16} /> },
    { id: "launcher", label: "Launcher", icon: <Play size={16} /> },
    { id: "agent", label: "Agent", icon: <Bot size={16} /> },
  ];
  const pageCopy = {
    overview: ["Operations Overview", "Runtime health, active work, and the signals that need attention."],
    jobs: jobDetailOpen && selectedJob
      ? [selectedJob.name, `${selectedJob.kind} · ${selectedJob.status} · step ${selectedJob.step ?? 0}`]
      : ["Jobs", "Open an AReno train or serve task to inspect metrics, samples, config, and logs."],
    runtime: ["Runtime Environment", "Review areno check, areno env, dependencies, GPU state, and repository context."],
    launcher: ["Task Launcher", "Start low-intrusion AReno train or serve subprocesses from explicit configs."],
    agent: ["Agent Console", "Chat with an operations agent using the selected job context."],
  };

  async function startTrain() {
    setBusy("Starting train job...");
    try {
      const result = await api("/api/jobs/train", { method: "POST", body: JSON.stringify(trainConfig) });
      setSelectedJobId(result.job.id);
      await jobs.refresh();
    } finally {
      setBusy("");
    }
  }

  async function startServe() {
    setBusy("Starting serve job...");
    try {
      const result = await api("/api/jobs/serve", { method: "POST", body: JSON.stringify(serveConfig) });
      setSelectedJobId(result.job.id);
      await jobs.refresh();
    } finally {
      setBusy("");
    }
  }

  async function executeAgentPlan(plan) {
    setBusy("Executing plan...");
    try {
      const result = await api("/api/agent/tools/run", {
        method: "POST",
        body: JSON.stringify({
          tool: plan.tool || inferPlanRunTool(plan),
          parameters: plan.parameters || {},
        }),
      });
      if (result.job?.id) {
        setSelectedJobId(result.job.id);
        await jobs.refresh();
      }
      return result;
    } finally {
      setBusy("");
    }
  }

  async function stopJob(id) {
    setBusy("Stopping job...");
    try {
      await api(`/api/jobs/${id}/stop`, { method: "POST", body: "{}" });
      await jobs.refresh();
    } finally {
      setBusy("");
    }
  }

  async function refreshRuntime() {
    setBusy("Running environment checks...");
    try {
      await api("/api/runtime/refresh", { method: "POST", body: "{}" });
      await Promise.all([env.refresh(), runtimeAttention.refresh()]);
    } finally {
      setBusy("");
    }
  }

  async function repairRuntime(action) {
    setBusy(`Fixing ${action.package || "runtime dependency"}...`);
    try {
      const result = await api("/api/runtime/repair", {
        method: "POST",
        body: JSON.stringify({ action_id: action.id }),
      });
      if (result.job?.id) {
        setRuntimeRepair({ actionId: action.id, jobId: result.job.id, job: result.job, refreshed: false });
        await jobs.refresh();
      }
      await runtimeAttention.refresh();
      setBusy("");
    } catch (err) {
      setBusy(err.message || String(err));
    }
  }

  async function executeOverviewQuickAction(action, overviewJob = null) {
    if (action.kind === "agent_prompt") {
      const jobContext = overviewJob ? `\n\nTrack this overview job: ${overviewJob.name} (${overviewJob.id}).` : "";
      setSelectedJobId(overviewJob?.id || null);
      setActivePage("agent");
      await runAgent(`${action.prompt || "Track the latest job and summarize its health."}${jobContext}`, false, overviewJob?.id || null);
      return;
    }
    setBusy(`Running ${action.label}...`);
    try {
      const result = await api("/api/quick-actions/run", {
        method: "POST",
        body: JSON.stringify({ action_id: action.id, config: trainConfig }),
      });
      if (result.job?.id) {
        setSelectedJobId(result.job.id);
        await jobs.refresh();
        setBusy(`${result.job.name || "Job"} started.`);
        window.setTimeout(() => setBusy(""), 2400);
      } else if (result.env) {
        await Promise.all([env.refresh(), runtimeAttention.refresh()]);
        setRuntimeCheckResult(result);
        setBusy("");
      }
    } catch (err) {
      setBusy(err.message || String(err));
    }
  }

  async function generateAgentFollowUps(historyOverride = null) {
    setFollowUpsLoading(true);
    try {
      const result = await api("/api/agent/follow-ups", {
        method: "POST",
        body: JSON.stringify({
          job_id: selectedJob?.id || null,
          provider: agentProvider,
          history: historyOverride || compactAgentHistory(agentMessages),
          language,
        }),
      });
      setAgentFollowUps(result.follow_ups || []);
    } catch (err) {
      setBusy(err.message || String(err));
    } finally {
      setFollowUpsLoading(false);
    }
  }

  async function runAgent(promptOverride = null, recovery = false, jobIdOverride = null) {
    const prompt = String(promptOverride ?? agentPrompt).trim();
    if (!prompt) return;
    if (promptOverride == null) setAgentPrompt("");
    setAgentFollowUps([]);
    if (recovery) setAgentRecovering(true);
    const assistantId = `assistant-${Date.now()}`;
    setAgentMessages((messages) => [
      ...messages,
      { role: "user", content: prompt },
      { id: assistantId, role: "assistant", content: "", events: [], streaming: true },
    ]);
    setBusy("Agent analyzing...");
    let streamedAssistantText = "";
    let agentFailed = false;
    let failureMessage = "";
    try {
      await streamAgentResponse({
        prompt,
        job_id: jobIdOverride ?? selectedJob?.id ?? null,
        provider: agentProvider,
        history: compactAgentHistory(agentMessages),
        language,
        onEvent: (event) => {
          if (event.type === "content_delta") streamedAssistantText += event.content || "";
          if (event.type === "error") {
            agentFailed = true;
            failureMessage = event.content || "The agent stream reported an error.";
          }
          applyAgentEvent(assistantId, event);
        },
      });
    } catch (err) {
      agentFailed = true;
      failureMessage = err.message || String(err);
      applyAgentEvent(assistantId, { type: "error", content: `Agent request failed: ${failureMessage}` });
    } finally {
      applyAgentEvent(assistantId, { type: "done" });
      setBusy("");
      setAgentRecovering(false);
      if (agentFailed) {
        setAgentFailure((previous) => ({
          prompt,
          error: failureMessage || "Agent request failed before producing a complete response.",
          jobId: jobIdOverride ?? selectedJob?.id ?? null,
          attempts: recovery ? Number(previous?.attempts || 1) + 1 : 1,
          failedAt: Date.now(),
        }));
        agentRecoveryState.refresh();
      } else {
        setAgentFailure(null);
        api("/api/agent/recovery/clear", { method: "POST", body: JSON.stringify({ job_id: jobIdOverride ?? selectedJob?.id ?? null }) }).catch(() => {});
        agentRecoveryState.refresh();
      }
      if (!agentFailed && streamedAssistantText.trim()) {
        await generateAgentFollowUps([
          ...compactAgentHistory(agentMessages),
          { role: "user", content: prompt },
          { role: "assistant", content: streamedAssistantText.trim() },
        ]);
      }
    }
  }

  async function dismissAgentFailure() {
    setAgentFailure(null);
    await api("/api/agent/recovery/clear", { method: "POST", body: JSON.stringify({ job_id: selectedJob?.id || null }) }).catch(() => {});
    agentRecoveryState.refresh();
  }

  function setAgentMessages(updater) {
    const sessionId = activeAgentSession.id;
    setAgentSessions((sessions) =>
      sessions.map((session) => {
        if (session.id !== sessionId) return session;
        const currentMessages = session.messages || defaultAgentMessages;
        const nextMessages = typeof updater === "function" ? updater(currentMessages) : updater;
        return {
          ...session,
          messages: nextMessages,
          updatedAt: Date.now(),
          title: inferAgentSessionTitle(nextMessages, session.title),
        };
      })
    );
  }

  function setAgentFollowUps(updater) {
    const sessionId = activeAgentSession.id;
    setAgentSessions((sessions) => sessions.map((session) => {
      if (session.id !== sessionId) return session;
      const current = session.followUps || [];
      return { ...session, followUps: typeof updater === "function" ? updater(current) : updater, updatedAt: Date.now() };
    }));
  }

  function newAgentChat() {
    const session = createAgentSession(language === "zh" ? zhAgentMessages : defaultAgentMessages, language === "zh" ? "新对话" : "New chat");
    setAgentSessions((sessions) => [session, ...sessions]);
    setActiveAgentSessionId(session.id);
    setAgentPrompt("");
    setAgentFailure(null);
    api("/api/agent/recovery/clear", { method: "POST", body: "{}" }).catch(() => {});
    setAgentChatTab("chat");
  }

  function openAgentSession(sessionId) {
    setActiveAgentSessionId(sessionId);
    setAgentChatTab("chat");
  }

  function compactAgentHistory(messages) {
    return messages
      .filter((message) => (message.role === "user" || message.role === "assistant") && !message.streaming)
      .map((message) => ({ role: message.role, content: agentMessageText(message) }))
      .filter((message) => message.content)
      .slice(-10);
  }

  function agentMessageText(message) {
    if (message.content) return message.content;
    return (message.events || [])
      .filter((event) => event.type === "content" || event.type === "reasoning")
      .map((event) => event.text || "")
      .join("\n")
      .trim();
  }

  async function streamAgentResponse({ prompt, job_id, provider, history, language: responseLanguage, onEvent }) {
    const response = await fetch(`${API_BASE}api/agent/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, job_id, provider, history, language: responseLanguage }),
      });
    if (!response.ok || !response.body) {
      const text = await response.text();
      throw new Error(text || response.statusText);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        onEvent(JSON.parse(line));
      }
    }
    if (buffer.trim()) onEvent(JSON.parse(buffer));
  }

  function applyAgentEvent(messageId, event) {
    setAgentMessages((messages) =>
      messages.map((message) => {
        if (message.id !== messageId) return message;
        if (event.type === "content_delta") {
          return appendAgentEventText(message, "content", event.content || "");
        }
        if (event.type === "reasoning_delta") {
          return appendAgentEventText(message, "reasoning", event.content || "");
        }
        if (event.type === "tool_calls") {
          return mergeAgentToolCalls(message, event.tool_calls || []);
        }
        if (event.type === "tool_call_delta") {
          return upsertAgentToolCall(message, event.tool_call, true);
        }
        if (event.type === "tool_result") {
          return { ...message, events: [...(message.events || []), { type: "tool_result", result: event.tool_result }] };
        }
        if (event.type === "error") {
          return { ...appendAgentEventText(message, "content", event.content || ""), streaming: false };
        }
        if (event.type === "done") {
          return { ...message, streaming: false };
        }
        return message;
      })
    );
  }

  function appendAgentEventText(message, type, delta) {
    if (!delta) return message;
    const events = [...(message.events || [])];
    const last = events[events.length - 1];
    if (last?.type === type) {
      events[events.length - 1] = { ...last, text: `${last.text || ""}${delta}` };
    } else {
      events.push({ type, text: delta });
    }
    const content = type === "content" ? `${message.content || ""}${delta}` : message.content;
    return { ...message, content, events };
  }

  function upsertAgentToolCall(message, toolCall, live = false) {
    if (!toolCall) return message;
    const events = [...(message.events || [])];
    const matchIndex = events.findIndex((event) => event.type === "tool_call" && sameToolCall(event.call, toolCall));
    if (matchIndex >= 0) {
      events[matchIndex] = { ...events[matchIndex], call: toolCall, live };
    } else {
      events.push({ type: "tool_call", call: toolCall, live });
    }
    return { ...message, events };
  }

  function sameToolCall(left, right) {
    if (!left || !right) return false;
    if (left.id && right.id && left.id === right.id) return true;
    if (
      left.round !== undefined &&
      right.round !== undefined &&
      left.index !== undefined &&
      right.index !== undefined &&
      left.round === right.round &&
      left.index === right.index
    ) {
      return true;
    }
    return false;
  }

  function mergeAgentToolCalls(message, toolCalls) {
    let next = message;
    for (const toolCall of toolCalls) {
      next = upsertAgentToolCall(next, toolCall, false);
    }
    return next;
  }

  function renderPage() {
    if (activePage === "overview") {
      return (
        <OverviewPage
          env={env.data}
          jobs={jobList}
          runtimeAttention={runtimeAttention.data}
          quickActions={quickActions.data?.actions || []}
          onQuickAction={executeOverviewQuickAction}
          onRuntimeRepair={repairRuntime}
          runtimeRepair={runtimeRepair ? { ...runtimeRepair, job: runtimeRepairJob } : null}
        />
      );
    }
    if (activePage === "runtime") {
      return <RuntimePrdPage env={env.data} onRefresh={refreshRuntime} />;
    }
    if (activePage === "launcher") {
      return (
        <LauncherPrdPage
          mode={launcherMode}
          setMode={setLauncherMode}
          trainConfig={trainConfig}
          setTrainConfig={setTrainConfig}
          serveConfig={serveConfig}
          setServeConfig={setServeConfig}
          onStartTrain={startTrain}
          onStartServe={startServe}
          env={env.data}
          presets={launcherPresets.data?.presets || []}
        />
      );
    }
    if (activePage === "agent") {
      const serverRecovery = agentRecoveryState.data?.recovery || {};
      const recovery = agentFailure || (serverRecovery.active ? { error: serverRecovery.error, attempts: 1 } : null);
      return (
        <div className="agentPrdLayout">
        <section className="panel chatPanel">
          <div className="panelHeader">
            <div>
              <h2>Agent Console</h2>
              <p>Natural-language operations with explicit runtime and job context.</p>
            </div>
            <div className="agentHeaderActions">
              <StatusBadge status={env.data?.ready ? "ok" : "warn"} />
              <button className="secondaryButton" onClick={newAgentChat}><Plus size={15} /> New Chat</button>
              <button className="secondaryButton" onClick={() => setAgentSettingsOpen(true)}><Settings2 size={15} /> Settings</button>
            </div>
          </div>
          <div className="pillRow agentContextPills">
            <span>repo: {env.data?.repo?.branch || "unknown"}</span>
            <span>job: {selectedJob?.id || "none selected"}</span>
            <span>GPU: {env.data?.gpus?.length || 0} visible</span>
          </div>
          <div className="agentTabs">
            <button className={classNames(agentChatTab === "chat" && "active")} onClick={() => setAgentChatTab("chat")}>
              <MessageSquare size={15} /> Chat
            </button>
            <button className={classNames(agentChatTab === "history" && "active")} onClick={() => setAgentChatTab("history")}>
              <History size={15} /> History
            </button>
          </div>
          {agentChatTab === "history" ? (
            <AgentHistory sessions={agentSessions} activeId={activeAgentSession.id} onOpen={openAgentSession} onNew={newAgentChat} />
          ) : (
            <>
              <div className="chatMessages" ref={chatMessagesRef}>
                {agentMessages.map((message, index) => (
                  <div key={`${message.id || message.role}-${index}`} className={classNames("chatBubble", message.role)}>
                    <span>{message.role}</span>
                    {message.events?.length ? <AgentEventList events={message.events} onPlanConfirm={executeAgentPlan} /> : <MarkdownBlock text={message.content} />}
                  </div>
                ))}
              </div>
              <div className="chatComposer">
                <label className="chatInputField">
                  <textarea
                    aria-label="Message the operations agent"
                    placeholder={selectedJob ? "Ask about this job, its metrics, or recent logs..." : "Ask about the runtime or describe a task to launch..."}
                    value={agentPrompt}
                    onChange={(event) => setAgentPrompt(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        runAgent();
                      }
                    }}
                  />
                </label>
                <button className="primaryButton chatSendButton" disabled={!agentPrompt.trim()} onClick={() => runAgent()}><Send size={16} /> Send</button>
              </div>
            </>
          )}
          {agentSettingsOpen && (
            <Modal title="Agent Settings" onClose={() => setAgentSettingsOpen(false)}>
              <AgentProviderForm provider={agentProvider} setProvider={setAgentProvider} />
            </Modal>
          )}
        </section>
        <aside className="agentSideRail">
          <section className="panel agentRecoveryCard">
            <div className="panelHeader"><div><h2>Error Recovery</h2><p>Detect and retry failed agent requests.</p></div><StatusBadge status={recovery ? "failed" : "ok"} /></div>
            {recovery ? (
              <div className="attentionItem"><StatusBadge status="failed" /><div><strong>Agent request failed</strong><p>{recovery.error}</p>{recovery.attempts > 1 && <small>{recovery.attempts} recovery attempts</small>}</div></div>
            ) : (
              <div className="attentionItem"><StatusBadge status="ok" /><div><strong>No active agent errors</strong><p>The current conversation has no failed request.</p></div></div>
            )}
            {agentFailure && <div className="recoveryButtons"><button className="primaryButton fullButton recoveryAction" disabled={agentRecovering} onClick={() => runAgent(agentFailure.prompt, true)}>{agentRecovering ? "Recovering..." : "Retry failed request"}</button><button className="secondaryButton fullButton recoveryAction" disabled={agentRecovering} onClick={dismissAgentFailure}>Dismiss</button></div>}
          </section>
          <section className="panel agentFollowupsCard">
            <div className="panelHeader"><div><h2>Suggested Follow-ups</h2><p>LLM-generated actions grounded in the current context.</p></div></div>
            {followUpsLoading && <div className="followupsLoading">Generating follow-ups...</div>}
            {!followUpsLoading && agentFollowUps.length === 0 && <p>Follow-ups appear after the current response completes.</p>}
            {agentFollowUps.map((followUp) => <button key={followUp.id} className="secondaryButton" onClick={() => { setAgentChatTab("chat"); runAgent(followUp.prompt); }}>{followUp.label}</button>)}
          </section>
        </aside>
        </div>
      );
    }
    if (jobDetailOpen && selectedJob) {
      return <JobFullDetailPage job={selectedJob} refreshNonce={refreshNonce} onBack={() => setJobDetailOpen(false)} onStop={() => stopJob(selectedJob.id)} />;
    }
    return (
      <div className="jobsPageStack">
        <section className="panel jobListPage">
          <div className="panelHeader">
            <div>
              <h2>Jobs</h2>
              <p>Registered AReno train and serve processes. Open a job to inspect its full detail page.</p>
            </div>
            <div className="jobsToolbar">
              <div className="tabs compactTabs">
                {["all", "running", "failed", "stopped"].map((status) => (
                  <button key={status} className={classNames(jobFilter === status && "active")} onClick={() => { setJobFilter(status); setJobPage(1); }}>
                    {status[0].toUpperCase() + status.slice(1)}
                  </button>
                ))}
              </div>
              <div className="pagerControls">
              <button className="secondaryButton" disabled={currentJobPage <= 1} onClick={() => setJobPage((page) => Math.max(1, page - 1))}>Prev</button>
              <span>{currentJobPage} / {jobPageCount}</span>
              <button className="secondaryButton" disabled={currentJobPage >= jobPageCount} onClick={() => setJobPage((page) => Math.min(jobPageCount, page + 1))}>Next</button>
              </div>
            </div>
          </div>
          <div className="jobTableWrap">
            {filteredJobs.length === 0 && <EmptyState title="No matching jobs" text="Start a task from Launcher or select another status." />}
            {filteredJobs.length > 0 && <table className="jobTable">
              <thead><tr><th>Job</th><th>Status</th><th>Stage</th><th>Metric</th><th>Elapsed</th><th>Action</th></tr></thead>
              <tbody>
            {pagedJobs.map((job) => (
              <tr key={job.id}>
                <td><strong>{job.name}</strong><span className="mono subline">{job.id} · {job.kind}</span></td>
                <td><StatusBadge status={job.status} /></td>
                <td>{job.stage || "unknown"} · step {job.step ?? 0}</td>
                <td>{latestPerfSignal(job)}</td>
                <td>{formatElapsed(job)}</td>
                <td><button className="secondaryButton tableAction" onClick={() => { setSelectedJobId(job.id); setJobDetailOpen(true); }}>Open</button></td>
              </tr>
            ))}
              </tbody>
            </table>}
          </div>
          {filteredJobs.length > jobPageSize && (
            <div className="listFooter">
              Showing {(currentJobPage - 1) * jobPageSize + 1}-{Math.min(currentJobPage * jobPageSize, filteredJobs.length)} of {filteredJobs.length} jobs
            </div>
          )}
        </section>
        {previewJob ? (
          <JobsSelectedDetail
            job={previewJob}
            env={env.data}
            onStop={() => stopJob(previewJob.id)}
          />
        ) : <div className="jobDetailPreviewGrid"><section className="panel"><EmptyState title="No job overview" text="Launch a task to populate the job overview." /></section><section className="panel"><EmptyState title="No rollout sample" text="Samples appear after a rollout completes." /></section></div>}
      </div>
    );
  }

  const [pageTitle, pageDescription] = pageCopy[activePage] || pageCopy.jobs;

  return (
    <div className="shell">
      <aside className="rail">
        <div className="brand">
          <img className="brandMark" src="https://mdn.alipayobjects.com/huamei_fz8c8n/afts/img/6aFwRZclmL8AAAAAQyAAAAgADpuRAQJr/original" alt="AReno logo" />
          <div>
            <div className="brandName">AReno Ops</div>
            <div className="brandMeta">runtime workbench</div>
          </div>
        </div>
        <nav className="nav">
          {pages.map((page) => (
            <button key={page.id} className={classNames("navItem", activePage === page.id && "active")} onClick={() => { if (page.id === "jobs") setJobDetailOpen(false); setActivePage(page.id); }}>
              {page.icon} {page.label}
            </button>
          ))}
        </nav>
      </aside>

      <main className="main">
        <div className="mobileNav">
          <strong>AReno Ops</strong>
          <select value={activePage} onChange={(event) => { if (event.target.value === "jobs") setJobDetailOpen(false); setActivePage(event.target.value); }}>
            {pages.map((page) => <option key={page.id} value={page.id}>{page.label}</option>)}
          </select>
        </div>
        <header className="topbar">
          <div>
            <h1>{pageTitle}</h1>
            <p>{pageDescription}</p>
          </div>
          <div className="topActions">
            <button className="iconButton" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} title="Toggle theme">
              {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
            </button>
            <button
              className="iconButton languageButton"
              onClick={() => setLanguage(language === "en" ? "zh" : "en")}
              title={language === "en" ? "Translate to Chinese" : "Translate to English"}
            >
              <Languages size={16} />
              <span>{language === "en" ? "中文" : "EN"}</span>
            </button>
          </div>
        </header>

        {busy && <div className="notice">{busy}</div>}

        {renderPage()}
        {runtimeCheckResult && <RuntimeCheckResultModal result={runtimeCheckResult} onClose={() => setRuntimeCheckResult(null)} />}
      </main>
    </div>
  );
}

function OverviewPage({ env, jobs, runtimeAttention, quickActions, onQuickAction, onRuntimeRepair, runtimeRepair }) {
  const activeJobs = jobs.filter(isActiveJob);
  const failedJobs = jobs.filter((job) => job.status === "failed");
  const latestJobSummary = newestJob(activeJobs.length ? activeJobs : jobs);
  const latestJobDetail = usePolling(
    () => latestJobSummary ? api(`/api/jobs/${latestJobSummary.id}`) : Promise.resolve(null),
    2000,
    [latestJobSummary?.id],
  );
  const detailedJob = latestJobDetail.data?.job;
  const focusJob = detailedJob?.id === latestJobSummary?.id ? detailedJob : latestJobSummary;
  const gpus = env?.gpus || [];
  const checks = env?.checks || [];
  const warning = runtimeAttention?.attention || checks.find((check) => ["warn", "fail"].includes(String(check.status).toLowerCase()));
  const repairMatchesWarning = warning?.repair?.id && runtimeRepair?.actionId === warning.repair.id;
  const repairStatus = repairMatchesWarning ? String(runtimeRepair?.job?.status || "created").toLowerCase() : "";
  const repairInProgress = ["created", "running"].includes(repairStatus);
  const algo = String(configValue(focusJob, "algo") || "").toLowerCase();
  const rewardBearing = ["gspo", "grpo", "ppo"].includes(algo);
  const health = rewardBearing
    ? findJobMetric(focusJob, ["rollout/rewards_mean", "reward_mean", "reward"])
    : findJobMetric(focusJob, ["train/loss", "loss", "policy_loss"]);

  return (
    <div className="overviewPage">
      <section className="summaryGrid">
        <SummaryCard label="Runtime" value={env?.ready ? "Ready" : env ? "Needs attention" : "Checking"} detail={`${env?.check_counts?.ok ?? 0} OK · ${env?.check_counts?.warn ?? 0} WARN`} tone={env?.ready ? "ok" : "warn"} />
        <SummaryCard label="GPU" value={gpus.length ? `${gpus.length} available` : "No GPU data"} detail={gpus[0]?.name || "Reported by runtime API"} />
        <SummaryCard label="Active jobs" value={String(activeJobs.length)} detail={failedJobs.length ? `${failedJobs.length} failed job${failedJobs.length === 1 ? "" : "s"}` : "No failed jobs"} tone={failedJobs.length ? "warn" : "info"} />
        <SummaryCard label="Latest health" value={health == null ? "No signal" : formatMetric(health)} detail={health == null ? "Waiting for metrics" : `Latest ${rewardBearing ? "reward" : "loss"} signal`} />
      </section>

      <div className="overviewLayout">
        <div className="overviewPrimary">
          {focusJob ? <RunningJobSummary job={focusJob} /> : (
            <section className="panel"><EmptyState title="No jobs yet" text="Launch a train or serve task to populate live operations data." /></section>
          )}

          <section className="panel overviewSignals">
            <div className="panelHeader"><div><h2>Metrics</h2><p>Switch between reward and loss for the latest job.</p></div><span className="sourceBadge">Live API</span></div>
            {focusJob ? <OverviewRewardLossChart job={focusJob} /> : <EmptyState title="No metrics available" text="Metrics appear after a job starts reporting scalar data." />}
          </section>
        </div>

        <aside className="overviewAside">
          <section className="panel attentionCard">
            <div className="panelHeader"><div><h2>Runtime attention</h2><p>Highest-priority environment finding.</p></div></div>
            {warning ? (
              <div className="attentionItem">
                <StatusBadge status={warning.status} />
                <div className="attentionItemBody">
                  <strong>{warning.name || warning.label || "Runtime warning"}</strong>
                  <p>{warning.detail || warning.message || "Review the runtime check details."}</p>
                  {warning.repair?.kind === "install_package" && (
                    <div className="runtimeRepairControl">
                      <button
                        className="secondaryButton runtimeFixButton"
                        disabled={repairInProgress || repairStatus === "succeeded"}
                        onClick={() => onRuntimeRepair(warning.repair)}
                      >
                        {repairInProgress ? <RefreshCw className="spinIcon" size={14} /> : <Wrench size={14} />}
                        {repairStatus === "running" ? "Installing..." : repairStatus === "created" ? "Starting..." : repairStatus === "succeeded" ? "Installed" : repairStatus === "failed" ? "Retry fix" : warning.repair.label || "Fix"}
                      </button>
                      {repairMatchesWarning && <RuntimeRepairProgress job={runtimeRepair.job} />}
                    </div>
                  )}
                </div>
              </div>
            ) : <div className="attentionItem"><StatusBadge status="ok" /><div><strong>No blocking checks</strong><p>The current runtime report has no warning or failure.</p></div></div>}
          </section>
          <section className="panel quickActions">
            <div className="panelHeader"><div><h2>Quick actions</h2><p>Short paths into common workflows.</p></div></div>
            {quickActions.map((action, index) => <button key={action.id} className={index === 0 ? "primaryButton" : "secondaryButton"} onClick={() => onQuickAction(action, focusJob)}>{quickActionIcon(action.kind)} {action.label}</button>)}
          </section>
        </aside>
      </div>
    </div>
  );
}

function RuntimeRepairProgress({ job }) {
  if (!job) return null;
  const status = String(job.status || "created").toLowerCase();
  const logs = job.logs || [];
  const latestLog = logs.length ? logs[logs.length - 1] : "Preparing package installer...";
  return (
    <div className={classNames("runtimeRepairProgress", status)} aria-live="polite">
      {["created", "running"].includes(status) && <div className="runtimeRepairTrack"><i /></div>}
      <small>{status === "failed" ? "Installation failed" : status === "succeeded" ? "Installation complete" : latestLog}</small>
    </div>
  );
}

function quickActionIcon(kind) {
  if (kind === "launcher_preset") return <Play size={15} />;
  if (kind === "runtime_refresh") return <RefreshCw size={15} />;
  return <Bot size={15} />;
}

function OverviewRewardLossChart({ job }) {
  const algo = String(configValue(job, "algo") || "").toLowerCase();
  const metricKinds = ["gspo", "grpo", "ppo"].includes(algo)
    ? ["reward", "loss", "gradnorm", "seqlen"]
    : ["loss", "gradnorm", "seqlen"];
  const [series, setSeries] = useState({ reward: [], loss: [], gradnorm: [], seqlen: [] });
  const [names, setNames] = useState({ reward: "", loss: "", gradnorm: "", seqlen: "" });
  const [activeMetric, setActiveMetric] = useState(metricKinds[0]);
  const [hoveredPoint, setHoveredPoint] = useState(null);

  useEffect(() => {
    if (!metricKinds.includes(activeMetric)) setActiveMetric(metricKinds[0]);
  }, [algo, activeMetric]);

  useEffect(() => {
    let cancelled = false;
    let timer;
    const load = async () => {
      try {
        const data = await api(`/api/jobs/${job.id}/metrics`);
        const metricNames = metricNamesFrom(data.metrics || []);
        const nextNames = Object.fromEntries(metricKinds.map((kind) => [kind, selectOverviewMetric(metricNames, kind)]));
        const metricData = await Promise.all(metricKinds.map((kind) => (
          nextNames[kind]
            ? api(`/api/jobs/${job.id}/metric?name=${encodeURIComponent(nextNames[kind])}`)
            : Promise.resolve({ points: [] })
        )));
        if (cancelled) return;
        setNames((current) => ({ ...current, ...nextNames }));
        setSeries((current) => ({
          ...current,
          ...Object.fromEntries(metricKinds.map((kind, index) => [kind, normalizeMetricPoints(metricData[index].points)])),
        }));
      } catch {
        if (!cancelled) setSeries({ reward: [], loss: [], gradnorm: [], seqlen: [] });
      }
    };
    load();
    timer = window.setInterval(load, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [job.id, algo]);

  const activeSeries = series[activeMetric] || [];
  const plot = buildOverviewMetricPlot(activeSeries);
  const hasPoints = activeSeries.length > 0;
  return (
    <div className="overviewMetricChart">
      <div className="overviewMetricToolbar">
        <div className="tabs compactTabs metricSwitch" aria-label="Metric plot">
          {metricKinds.map((kind) => <button key={kind} className={classNames(activeMetric === kind && "active")} onClick={() => setActiveMetric(kind)}>{overviewMetricLabel(kind)}</button>)}
        </div>
        <div className="metricLegend">
          <span className={`${activeMetric}Legend`}><i />{names[activeMetric] || overviewMetricLabel(activeMetric)}<b>{lastMetricValue(activeSeries)}</b></span>
        </div>
      </div>
      {!hasPoints ? <div className="plotEmpty">No {activeMetric} points reported yet.</div> : (
        <div className="metricPlotWrap">
          <svg className="metricPlot overviewPlot" viewBox="0 0 720 220" role="img" aria-label={`${activeMetric} metrics`}>
            <g className="plotGrid">{[0, 1, 2, 3].map((item) => <line key={item} x1="10" x2="710" y1={35 + item * 52} y2={35 + item * 52} />)}</g>
            <polyline className={`overviewMetricLine ${activeMetric}Line`} points={plot.points} />
            {activeSeries.map((point, index) => (
              <g key={`${point.step}-${index}`}>
                <circle className={`metricDataPoint ${activeMetric}Point`} cx={plot.coords[index]?.x || 0} cy={plot.coords[index]?.y || 0} r={metricPointRadius(activeSeries.length)} />
                <circle className="metricHoverTarget" cx={plot.coords[index]?.x || 0} cy={plot.coords[index]?.y || 0} r="5" onMouseEnter={() => setHoveredPoint({ point, coord: plot.coords[index] })} onMouseLeave={() => setHoveredPoint(null)} />
              </g>
            ))}
          </svg>
          {hoveredPoint && <MetricPointTooltip name={names[activeMetric] || overviewMetricLabel(activeMetric)} point={hoveredPoint.point} coord={hoveredPoint.coord} width={720} height={220} />}
        </div>
      )}
      <div className="plotFooter"><span>step {plot.stepMin} to {plot.stepMax}</span><span>{plot.minLabel} to {plot.maxLabel}</span></div>
    </div>
  );
}

function selectOverviewMetric(names, type) {
  const lowered = names.map((name) => ({ name, key: name.toLowerCase() }));
  const exactPreferences = {
    reward: ["rollout/rewards_mean"],
    loss: ["train/loss", "loss", "train/policy_loss", "policy_loss", "actor_loss"],
    gradnorm: ["train/grad_norm", "grad_norm"],
    seqlen: ["rollout/seq_len_mean"],
  }[type] || [];
  for (const preferred of exactPreferences) {
    const exact = lowered.find((item) => item.key === preferred);
    if (exact) return exact.name;
  }
  if (type === "reward" || type === "seqlen") return "";
  const fallbackNeedle = type === "gradnorm" ? "grad_norm" : type;
  return lowered.find((item) => item.key.includes(fallbackNeedle))?.name || "";
}

function overviewMetricLabel(kind) {
  return { reward: "Reward", loss: "Loss", gradnorm: "Grad Norm", seqlen: "Sequence Length" }[kind] || kind;
}

function normalizeMetricPoints(points = []) {
  return points
    .filter((point) => Number.isFinite(Number(point.value)))
    .map((point) => ({ step: Number(point.step || 0), value: Number(point.value), time: point.time }));
}

function buildOverviewMetricPlot(points) {
  if (!points.length) return { points: "", coords: [], stepMin: 0, stepMax: 0, minLabel: "n/a", maxLabel: "n/a" };
  const { min, max } = numericExtent(points, (point) => point.value);
  const flat = max === min;
  const { min: stepMin, max: stepMax } = numericExtent(points, (point) => point.step);
  const valueSpan = Math.max(max - min, 1e-9);
  const stepSpan = Math.max(stepMax - stepMin, 1);
  const coords = points.map((point) => {
    const x = ((point.step - stepMin) / stepSpan) * 680 + 20;
    const y = flat ? 113 : 200 - ((point.value - min) / valueSpan) * 174;
    return { x, y };
  });
  return { points: coords.map(({ x, y }) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" "), coords, stepMin, stepMax, minLabel: compactNumber(min), maxLabel: compactNumber(max) };
}

function MetricPointTooltip({ name, point, coord, width, height }) {
  const align = coord.x < width * 0.2 ? "start" : coord.x > width * 0.8 ? "end" : "center";
  return (
    <div className={`metricPointTooltip ${align}`} style={{ left: `${(coord.x / width) * 100}%`, top: `${(coord.y / height) * 100}%` }}>
      <strong>{name || "metric"}</strong>
      <span>Step <b>{point.step}</b></span>
      <span>Value <b>{compactNumber(point.value)}</b></span>
      {point.time && <small>{new Date(point.time).toLocaleString()}</small>}
    </div>
  );
}

function metricPointRadius(pointCount) {
  if (pointCount <= 12) return 2.75;
  if (pointCount <= 60) return 1.9;
  return 1.15;
}

function lastMetricValue(points) {
  return points.length ? compactNumber(points[points.length - 1].value) : "—";
}

function RunningJobSummary({ job }) {
  const algo = getJobConfigValue(job, "algo");
  const checkpoint = getJobConfigValue(job, "ckpt") || getJobConfigValue(job, "model_path");
  const dataset = getJobConfigValue(job, "dataset_path") || getJobConfigValue(job, "dataset");
  const latestTiming = (job.timeperf || []).slice(-1)[0];
  const segments = Object.fromEntries((latestTiming?.segments || []).map((segment) => [segment.name, segment.seconds]));
  const steps = timelineSteps(job);
  const stage = timelineStageId(job);
  const currentIndex = Math.max(0, steps.findIndex((item) => timelineItemMatches(item, stage, job)));
  const stageDuration = (item, index) => {
    if (item.id === "rollout" && Number.isFinite(Number(segments.rollout))) return `${Number(segments.rollout).toFixed(1)}s`;
    if (["train", "actor_train", "critic_train"].includes(item.id) && Number.isFinite(Number(segments.train))) return `${Number(segments.train).toFixed(1)}s`;
    if (item.id === "created" && job.created_at) return "created";
    if (timelineItemMatches(item, stage, job)) return ["succeeded", "done"].includes(job.status) ? "complete" : job.status;
    return index < currentIndex ? "complete" : "pending";
  };

  return (
    <section className="panel runningSummary">
      <div className="panelHeader">
        <div><h2>Running Job Summary</h2><p>Current health, route, and stage progression for the latest job.</p></div>
        <div className="summaryHeaderActions"><StatusBadge status={job.status} /><span className="stepBadge">step {job.step ?? 0}</span></div>
      </div>
      <div className="pillRow summaryPills">
        <span>job: {job.id}</span>
        {algo && <span>algo: {algo}</span>}
        {checkpoint && <span>model: {shortModelName(checkpoint)}</span>}
        {dataset && <span>dataset: {String(dataset)}</span>}
      </div>
      <div className="stageRow">
        {steps.map((item, index) => {
          const current = timelineItemMatches(item, stage, job);
          return <div key={item.id} className={classNames("stageCell", index <= currentIndex && "done", current && "current")}><strong>{item.label}</strong><span>{stageDuration(item, index)}</span></div>;
        })}
      </div>
    </section>
  );
}

function getJobConfigValue(job, key) {
  if (job?.config?.[key] !== undefined && job.config[key] !== null && job.config[key] !== "") return job.config[key];
  for (const section of job?.config?.sections || []) {
    const item = (section.items || []).find((entry) => entry.key === key);
    if (item?.value !== undefined && item.value !== null && item.value !== "") return item.value;
  }
  if (job?.launch?.[key] !== undefined && job.launch[key] !== null && job.launch[key] !== "") return job.launch[key];
  return null;
}

function newestJob(jobs = []) {
  return [...jobs].sort((left, right) => Date.parse(right.created_at || 0) - Date.parse(left.created_at || 0))[0] || null;
}

function isActiveJob(job) {
  return String(job?.status || "").toLowerCase() === "running";
}

function shortModelName(value) {
  return String(value).replace(/\/$/, "").split("/").pop() || value;
}

function JobsSelectedDetail({ job, env, onStop }) {
  const reward = findJobMetric(job, ["rollout/rewards_mean", "reward_mean", "reward"]);
  const latestTiming = (job.timeperf || []).slice(-1)[0];
  const gpu = env?.gpus?.[0];
  return (
    <div className="selectedJobArea">
      <div className="jobDetailPreviewGrid">
        <section className="panel jobDetailOverviewCard">
          <div className="panelHeader">
            <div><h2>Job Detail: Overview</h2><p>Current health and recent progress for the selected job.</p></div>
            <div className="detailActions"><StatusBadge status={job.status} />{job.status === "running" && <button className="dangerButton" onClick={onStop}><CircleStop size={16} /> Stop</button>}</div>
          </div>
          <div className="jobIdentity"><strong>{job.name}</strong><span className="mono subline">{job.id}</span></div>
          <p className="healthSummary"><strong>Health summary:</strong> {jobHealthSummary(job)}</p>
          <div className="detailMetricGrid">
            <div><span>Reward</span><strong>{reward == null ? "No data" : formatMetric(reward)}</strong></div>
            <div><span>Step time</span><strong>{latestTiming?.total_s ? `${Number(latestTiming.total_s).toFixed(1)}s` : "No data"}</strong></div>
            <div><span>GPU memory</span><strong>{gpu ? `${gpu.memory_used_mb ?? 0} MB` : "No data"}</strong></div>
          </div>
        </section>
        <section className="panel rolloutSamplePanel">
          <div className="panelHeader"><div><h2>Rollout Sample</h2><p>Inspect prompt and completion pairs by step and sample.</p></div></div>
          <SampleView samples={job.samples || []} jobId={job.id} hideTitle />
        </section>
      </div>
    </div>
  );
}

function JobFullDetailPage({ job, refreshNonce, onBack, onStop }) {
  const logs = job.logs || [];
  return (
    <div className="jobFullDetailPage">
      <div className="detailPageToolbar">
        <button className="secondaryButton" onClick={onBack}><ArrowLeft size={16} /> Back to Jobs</button>
        {job.status === "running" && <button className="dangerButton" onClick={onStop}><CircleStop size={16} /> Stop job</button>}
      </div>
      <JobOverview job={job} detail refreshNonce={refreshNonce} />
      <section className="panel jobDetailSection">
        <div className="panelHeader"><div><h2>Metrics</h2><p>Training quality and stage timing for this job.</p></div></div>
        <JobMetricsView job={job} refreshNonce={refreshNonce} />
      </section>
      <section className="panel jobDetailSection">
        <div className="panelHeader"><div><h2>Rollout Sample</h2><p>Prompt and completion output captured during rollout.</p></div></div>
        <SampleView samples={job.samples || []} jobId={job.id} hideTitle />
      </section>
      <div className="jobDetailDataGrid">
        <ConfigView config={job.config} launch={job.launch} />
        <LogView logs={logs} />
      </div>
    </div>
  );
}

function jobHealthSummary(job) {
  if (job.status === "running") return `The job is running at ${job.stage || "its current stage"}, step ${job.step ?? 0}, with no terminal status reported.`;
  if (job.status === "failed") return `The job failed during ${job.stage || "an unknown stage"}. Review metrics and logs below.`;
  if (["succeeded", "done"].includes(job.status)) return `The job completed successfully after ${job.step ?? 0} steps.`;
  return `The job is ${job.status || "unknown"} at ${job.stage || "an unknown stage"}, step ${job.step ?? 0}.`;
}

function formatElapsed(job) {
  const start = Date.parse(job?.created_at || "");
  const end = Date.parse(job?.status === "running" ? new Date().toISOString() : job?.updated_at || "");
  if (!Number.isFinite(start) || !Number.isFinite(end)) return "—";
  const seconds = Math.max(0, Math.round((end - start) / 1000));
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

function JobOverview({ job, compact = false, detail = false, refreshNonce = 0, onOpen, onStop }) {
  return (
    <section className={classNames("panel", "jobOverview", detail && "detailPanel")}>
      <div className="panelHeader">
        <div>
          <span className="sectionEyebrow">{compact ? "Latest job" : "Selected job"}</span>
          <h2>{detail ? "Job Detail: Overview" : job.name}</h2>
          {detail && <strong className="detailJobName">{job.name}</strong>}
          <p className="mono">{job.id} · updated {formatRelativeTime(job.updated_at)}</p>
        </div>
        <div className="detailActions"><StatusBadge status={job.status} />{compact && <button className="secondaryButton" onClick={onOpen}>Open job</button>}</div>
      </div>
      <div className="jobOverviewStats">
        <div><span>Stage</span><strong>{job.stage || "unknown"}</strong></div>
        <div><span>Step</span><strong>{job.step ?? 0}</strong></div>
        <div><span>Latest signal</span><strong>{latestPerfSignal(job)}</strong></div>
        <div><span>Process</span><strong>{job.pid ? `PID ${job.pid}` : job.kind}</strong></div>
      </div>
      <Timeline job={job} />
    </section>
  );
}

function SummaryCard({ label, value, detail, tone = "neutral" }) {
  return <div className={classNames("summaryCard", tone)}><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}

function StatusBadge({ status = "unknown" }) {
  const statusText = String(status).toLowerCase();
  const normalized = statusText === "succeeded" || statusText === "done" ? "ok" : statusText;
  return <span className={classNames("statusBadge", normalized)}><i />{status}</span>;
}

function findJobMetric(job, names) {
  if (!job?.perf) return null;
  for (const name of names) if (Number.isFinite(Number(job.perf[name]))) return Number(job.perf[name]);
  return null;
}

function formatMetric(value) {
  if (!Number.isFinite(Number(value))) return "—";
  const number = Number(value);
  return Math.abs(number) >= 100 ? number.toFixed(0) : number.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function latestPerfSignal(job) {
  const entries = Object.entries(job?.perf || {}).filter(([, value]) => Number.isFinite(Number(value)));
  if (!entries.length) return "No metrics";
  const [name, value] = entries[0];
  return `${name} ${formatMetric(value)}`;
}

function formatRelativeTime(value) {
  const time = Date.parse(value || "");
  if (!Number.isFinite(time)) return "—";
  const seconds = Math.max(0, Math.round((Date.now() - time) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

function RuntimePrdPage({ env, onRefresh }) {
  const [selectedCheck, setSelectedCheck] = useState(null);
  const report = env?.report || {};
  const torch = report.torch || {};
  const checks = env?.checks || [];
  const gpus = env?.gpus || report?.torch?.gpus || [];
  const warnCount = env?.check_counts?.warn ?? checks.filter((check) => String(check.status).toLowerCase() === "warn").length;
  const failCount = env?.check_counts?.fail ?? checks.filter((check) => String(check.status).toLowerCase() === "fail").length;
  const dependencyRisk = failCount ? `${failCount} FAIL` : warnCount ? `${warnCount} WARN` : "Clear";
  return (
    <div className="runtimePrdPage">
      <section className="runtimeSummaryGrid">
        <SummaryCard label="AReno Check" value={env?.ready ? "Ready" : env ? "Needs attention" : "Checking"} detail={`Last refreshed ${new Date().toLocaleTimeString()}`} tone={env?.ready ? "ok" : "warn"} />
        <SummaryCard label="PyTorch / CUDA" value={`${torch.version || "n/a"} / ${torch.cuda_runtime || torch.cuda_build || "n/a"}`} detail={torch.cuda_available ? "Compatible runtime detected" : "CUDA runtime unavailable"} tone={torch.cuda_available ? "ok" : "warn"} />
        <SummaryCard label="Dependency Risk" value={dependencyRisk} detail={runtimeRiskLabel(checks)} tone={failCount || warnCount ? "warn" : "ok"} />
      </section>
      <div className="runtimePrdLayout">
        <section className="panel runtimeChecksPanel">
          <div className="panelHeader"><div><h2>Environment Checks</h2><p>Runtime requirements, compatibility, and actionable diagnostics.</p></div><button className="secondaryButton" onClick={onRefresh}><RefreshCw size={15} /> Run Check</button></div>
          <div className="runtimeCheckList">
            {checks.length === 0 && <EmptyState title="No checks reported" text="Run the environment check to populate diagnostics." />}
            {checks.slice(0, 12).map((check, index) => (
              <div className="runtimeCheckRow" key={`${check.name || check.label}-${index}`}>
                <StatusBadge status={check.status || "unknown"} />
                <div><strong>{check.name || check.label || "Runtime check"}</strong><p>{check.detail || check.message || "No additional details."}</p></div>
                <button className="secondaryButton tableAction" title="Diagnostic details" onClick={() => setSelectedCheck(check)}>Details</button>
              </div>
            ))}
          </div>
        </section>
        <section className="panel runtimeGpuPanel">
          <div className="panelHeader"><div><h2>GPU Cards</h2><p>Memory pressure and utilization before launch.</p></div></div>
          <div className="runtimeGpuList">
            {gpus.length === 0 && <EmptyState title="No GPUs reported" text="GPU cards appear when CUDA devices are visible." />}
            {gpus.map((gpu, index) => <RuntimeGpuCard key={gpu.index ?? index} gpu={gpu} index={index} />)}
          </div>
        </section>
      </div>
      {selectedCheck && (
        <Modal title={`${selectedCheck.name || selectedCheck.label || "Environment Check"} Details`} onClose={() => setSelectedCheck(null)}>
          <RuntimeCheckDetails check={selectedCheck} report={report} onClose={() => setSelectedCheck(null)} />
        </Modal>
      )}
    </div>
  );
}

function RuntimeCheckDetails({ check, report, onClose }) {
  const torch = report?.torch || {};
  const cuda = report?.cuda || {};
  const facts = [
    ["AReno", report?.areno?.version],
    ["Python", report?.python?.version],
    ["PyTorch", torch.version],
    ["CUDA build", torch.cuda_build],
    ["CUDA runtime", torch.cuda_runtime],
    ["CUDA available", torch.cuda_available],
    ["Visible GPUs", torch.device_count],
    ["NVCC", cuda.nvcc?.version || cuda.nvcc?.path],
    ["NVIDIA driver", cuda.driver?.driver_version],
    ["Driver CUDA", cuda.driver?.cuda_version],
    ["Platform", report?.platform?.platform],
  ].filter(([, value]) => value !== undefined && value !== null && value !== "");
  return (
    <div className="runtimeCheckDetails">
      <div className="runtimeCheckDetailLead"><StatusBadge status={check.status || "unknown"} /><div><strong>{check.detail || check.message || "No diagnostic value reported."}</strong>{check.next_step && <p>{check.next_step}</p>}</div></div>
      <div className="runtimeVersionGrid">
        {facts.map(([label, value]) => <div key={label}><span>{label}</span><strong>{String(value)}</strong></div>)}
      </div>
      <button className="primaryButton fullButton" onClick={onClose}>Done</button>
    </div>
  );
}

function runtimeRiskLabel(checks) {
  const risk = checks.find((check) => ["fail", "warn"].includes(String(check.status).toLowerCase()));
  return risk?.name || risk?.label || "No dependency warnings";
}

function RuntimeGpuCard({ gpu, index }) {
  const used = Number(gpu.memory_used_mb ?? gpu.memory_used ?? 0);
  const total = Number(gpu.memory_total_mb ?? gpu.memory_total ?? 0);
  const util = Number(gpu.utilization ?? gpu.utilization_gpu ?? 0);
  const memoryPct = total > 0 ? Math.min(100, (used / total) * 100) : 0;
  return (
    <div className="runtimeGpuCard">
      <div><strong>GPU {gpu.index ?? index} · {gpu.name || "CUDA device"}</strong><span>{used.toFixed(0)} / {total.toFixed(0)} MB · Util {util.toFixed(0)}%</span></div>
      <div className="meterTrack"><i style={{ width: `${memoryPct}%` }} /></div>
    </div>
  );
}

function AgentProviderForm({ provider, setProvider }) {
  return (
    <div className="agentConfig modalForm">
      <Field label="Base URL" value={provider.base_url || ""} onChange={(value) => setProvider({ ...provider, base_url: value })} compact />
      <Field label="Model" value={provider.model || ""} onChange={(value) => setProvider({ ...provider, model: value })} compact />
      <label className="field compact">
        <span>API key</span>
        <input type="password" value={provider.api_key || ""} onChange={(event) => setProvider({ ...provider, api_key: event.target.value })} />
      </label>
    </div>
  );
}

function AgentHistory({ sessions, activeId, onOpen, onNew }) {
  return (
    <div className="agentHistory">
      <div className="agentHistoryHeader">
        <div>
          <h3>Chat History</h3>
          <p>{sessions.length} saved conversations in this browser.</p>
        </div>
        <button className="secondaryButton" onClick={onNew}><Plus size={15} /> New Chat</button>
      </div>
      <div className="agentHistoryList">
        {sessions.map((session) => {
          const last = [...(session.messages || [])].reverse().find((message) => message.role === "user" || message.content);
          return (
            <button
              key={session.id}
              className={classNames("agentHistoryItem", session.id === activeId && "active")}
              onClick={() => onOpen(session.id)}
            >
              <strong>{session.title || "New chat"}</strong>
              <span>{last?.content || "No messages yet."}</span>
              <small>{new Date(session.updatedAt || session.createdAt || Date.now()).toLocaleString()}</small>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function Modal({ title, children, onClose }) {
  return (
    <div className="modalOverlay" role="presentation" onMouseDown={onClose}>
      <div className="modalCard" role="dialog" aria-modal="true" aria-label={title} onMouseDown={(event) => event.stopPropagation()}>
        <div className="modalHeader">
          <div>
            <h2>{title}</h2>
            <p>Stored locally in this browser.</p>
          </div>
          <button className="iconButton" onClick={onClose}>×</button>
        </div>
        {children}
      </div>
    </div>
  );
}

function RuntimeCheckResultModal({ result, onClose }) {
  const counts = result.env?.check_counts || {};
  return (
    <Modal title="Runtime Check Result" onClose={onClose}>
      <div className="runtimeResultSummary">
        <StatusBadge status={result.env?.ready ? "ok" : "warn"} />
        <span>{counts.ok || 0} OK · {counts.warn || 0} WARN · {counts.fail || 0} FAIL</span>
      </div>
      <pre className="runtimeCommandResult">{result.output || "$ areno check\nNo output returned."}</pre>
      <button className="primaryButton fullButton" onClick={onClose}>Done</button>
    </Modal>
  );
}

function JobMetricsView({ job, refreshNonce }) {
  return (
    <div className="jobMetricsGrid">
      <div className="panel insetPanel">
        <MetricChart jobId={job?.id} metricsDir={job?.metrics_dir} refreshNonce={refreshNonce} />
      </div>
      <div className="panel insetPanel">
        <TimePerfView rows={job?.timeperf || []} job={job} />
      </div>
    </div>
  );
}

function AgentEventList({ events, onPlanConfirm }) {
  const planEvents = events.filter((event) => event.type === "tool_result" && event.result?.plan);
  const otherEvents = events.filter((event) => !(event.type === "tool_result" && event.result?.plan));
  return (
    <div className="agentEventList">
      {otherEvents.map((event, index) => {
        if (event.type === "reasoning") return <ReasoningBlock key={index} text={event.text} />;
        if (event.type === "content") return <MarkdownBlock key={index} text={event.text} />;
        if (event.type === "tool_call") return <ToolCallCard key={index} call={event.call} live={event.live} />;
        if (event.type === "tool_result") return <ToolResultCard key={index} result={event.result} />;
        return null;
      })}
      {planEvents.map((event, index) => <AgentPlanCard key={event.result.plan.id || index} plan={event.result.plan} onConfirm={onPlanConfirm} />)}
    </div>
  );
}

function AgentPlanCard({ plan, onConfirm }) {
  const [editing, setEditing] = useState(false);
  const [parameters, setParameters] = useState(plan.parameters || {});
  const [execution, setExecution] = useState(null);
  useEffect(() => setParameters(plan.parameters || {}), [plan.id]);
  const entries = Object.entries(parameters);
  const planTool = plan.tool || inferPlanRunTool(plan);
  const command = commandForPlan(planTool, parameters);
  const editedPlan = { ...plan, tool: planTool, parameters, command };
  return (
    <section className="agentPlanCard">
      <div className="agentPlanHeader"><div><span>Execution plan</span><strong>{plan.objective}</strong></div><StatusBadge status={plan.status || "proposed"} /></div>
      {plan.summary && <p className="agentPlanSummary">{plan.summary}</p>}
      {entries.length > 0 && <div className={classNames("agentPlanParams", editing && "editing")}>{entries.map(([label, value]) => <label key={label}><span>{label.replaceAll("_", " ")}</span>{editing ? <input value={String(value)} onChange={(event) => setParameters((current) => ({ ...current, [label]: event.target.value }))} /> : <strong>{String(value)}</strong>}</label>)}</div>}
      <ol className="agentPlanSteps">{(plan.steps || []).map((step, index) => <li key={step.id || index}><span>{index + 1}</span><div><strong>{step.title}</strong>{step.detail && <p>{step.detail}</p>}</div><small>{step.status || "pending"}</small></li>)}</ol>
      {command && <pre className="agentPlanCommand">{command}</pre>}
      <div className="agentPlanActions">
        <button
          className="primaryButton"
          disabled={execution?.status === "running"}
          onClick={async () => {
            setExecution({ status: "running", message: "Starting..." });
            try {
              const result = await onConfirm?.(editedPlan);
              setExecution({ status: result?.ok === false ? "failed" : "ok", message: result?.job?.id ? `Started job ${result.job.id}` : "Execution completed" });
            } catch (error) {
              setExecution({ status: "failed", message: error.message || String(error) });
            }
          }}
        >{execution?.status === "running" ? "Executing..." : "Confirm Execution"}</button>
        {entries.length > 0 && <button className="secondaryButton" onClick={() => setEditing((value) => !value)}>{editing ? "Save Parameters" : "Edit Parameters"}</button>}
        {command && <button className="secondaryButton" onClick={() => navigator.clipboard.writeText(command)}>Copy Command</button>}
      </div>
      {execution && execution.status !== "running" && <p className={classNames("agentPlanExecution", execution.status)}>{execution.message}</p>}
    </section>
  );
}

function inferPlanRunTool(plan) {
  const command = String(plan?.command || "").toLowerCase();
  if (command.includes("--smoke-train")) return "smoke_train";
  if (command.includes("--smoke-infer")) return "smoke_infer";
  if (/\bareno\s+serve\b/.test(command)) return "start_serve";
  return "start_train";
}

function commandForPlan(tool, parameters = {}) {
  const mode = tool === "start_serve" ? "serve" : "train";
  const args = [];
  const negativeFlags = new Set(["activation_checkpointing", "use_kl_loss"]);
  for (const [key, value] of Object.entries(parameters)) {
    if (key === "extra_args" || value === "" || value === null || value === undefined) continue;
    const flag = `--${key.replaceAll("_", "-")}`;
    if (isPlanBoolean(value)) {
      if (planBoolValue(value)) args.push(flag);
      else if (negativeFlags.has(key)) args.push(`--no-${key.replaceAll("_", "-")}`);
      continue;
    }
    args.push(`${flag} ${shellQuote(value)}`);
  }
  if (tool === "smoke_train") args.push("--smoke-train");
  if (tool === "smoke_infer") args.push("--smoke-infer");
  const extraArgs = String(parameters.extra_args || "").trim();
  if (extraArgs) args.push(extraArgs);
  return [`areno ${mode} \\`, ...args.map((arg, index) => `  ${arg}${index < args.length - 1 ? " \\" : ""}`)].join("\n");
}

function isPlanBoolean(value) {
  return typeof value === "boolean" || (typeof value === "string" && ["true", "false"].includes(value.trim().toLowerCase()));
}

function planBoolValue(value) {
  return value === true || String(value).trim().toLowerCase() === "true";
}

function MarkdownBlock({ text }) {
  return (
    <div className="markdownBlock">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text || ""}</ReactMarkdown>
    </div>
  );
}

function ReasoningBlock({ text }) {
  return (
    <details className="reasoningBlock">
      <summary>
        <span>✓</span>
        <strong>Completed thinking</strong>
      </summary>
      <MarkdownBlock text={text} />
    </details>
  );
}

function ToolCallList({ toolCalls, live = false }) {
  return (
    <div className="toolCallList">
      {toolCalls.map((call, index) => <ToolCallCard key={call.id || index} call={call} live={live} />)}
    </div>
  );
}

function ToolCallCard({ call, live = false }) {
  const fn = call?.function || {};
  const name = fn.name || call?.name || "tool";
  const details = summarizeToolCall(call);
  return (
    <details className={classNames("toolCallCard", "toolCallDetails", live && "live")}>
      <summary className="toolCallHead">
        <span>{live ? "calling" : "tool call"}</span>
        <b>{formatToolInvocation(name, details)}</b>
        <em>{live ? "running" : ""}</em>
      </summary>
      <pre>{details}</pre>
    </details>
  );
}

function formatToolInvocation(name, argsText) {
  const compact = compactOneLine(argsText);
  if (!compact || compact === "{}") return `${name}()`;
  return `${name}(${compact})`;
}

function summarizeToolCall(call) {
  const fn = call.function || {};
  return fn.arguments || call.arguments || JSON.stringify(call, null, 2);
}

function ToolResultList({ toolResults }) {
  return (
    <div className="toolCallList">
      {toolResults.map((result, index) => <ToolResultCard key={`${result.name || "tool"}-${index}`} result={result} />)}
    </div>
  );
}

function ToolResultCard({ result }) {
  const summary = summarizeToolResult(result);
  return (
    <details className={classNames("toolCallCard", "toolResultDetails", result.ok ? "ok" : "failed")}>
      <summary>
        <span>{result.ok ? "result" : "error"}</span>
        <b>{result.name || "unknown"} · {result.ok ? "ok" : "failed"}</b>
        <em>{compactOneLine(summary) || "No output."}</em>
      </summary>
      <pre>{summary}</pre>
    </details>
  );
}

function compactOneLine(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

function summarizeToolResult(result) {
  if (result.error) return result.error;
  if (Array.isArray(result.jobs)) {
    return result.jobs.map((job) => `${job.id} · ${job.kind} · ${job.status} · step ${job.step ?? 0} · ${job.name}`).join("\n") || "No jobs.";
  }
  if (result.job) {
    return `${result.job.id} · ${result.job.kind} · ${result.job.status} · step ${result.job.step ?? 0}\n${result.job.name}`;
  }
  if (result.env) {
    return `ready=${result.env.ready} · gpu=${result.env.gpu_summary || "n/a"} · cwd=${result.env.cwd || "n/a"}`;
  }
  return JSON.stringify(result, null, 2);
}

function RuntimeDeck({ env }) {
  const checks = env?.checks || [];
  const report = env?.report || {};
  const deps = report.dependencies || {};
  const torch = report.torch || {};
  const platform = report.platform || {};
  const visibleChecks = checks.slice(0, 12);
  return (
    <section className="runtimeDeck">
      <div className="runtimeHero">
        <div className={classNames("readinessOrb", env?.ready ? "ready" : "blocked")}>
          {env?.ready ? "OK" : "!"}
        </div>
        <div>
          <div className="tinyLabel">AReno check</div>
          <h2>{env?.ready ? "Runtime ready for AReno tasks" : "Runtime needs attention before heavy jobs"}</h2>
          <p>
            {env?.check_counts?.ok ?? 0} OK · {env?.check_counts?.warn ?? 0} WARN · {env?.check_counts?.fail ?? 0} FAIL
          </p>
        </div>
      </div>
      <div className="envFacts">
        <EnvFact icon={<FileText size={15} />} label="AReno" value={report.areno?.version || "unknown"} />
        <EnvFact icon={<Cpu size={15} />} label="PyTorch" value={torch.version || "missing"} />
        <EnvFact icon={<Server size={15} />} label="CUDA" value={torch.cuda_build || "none"} />
        <EnvFact icon={<Box size={15} />} label="Platform" value={`${platform.system || "unknown"} ${platform.machine || ""}`} />
      </div>
      <div className="checkMatrix">
        {visibleChecks.map((item) => (
          <div key={item.name} className={classNames("checkItem", item.status.toLowerCase())}>
            <span>{item.status}</span>
            <strong>{item.name}</strong>
            <small>{item.detail || item.next_step || "no detail"}</small>
          </div>
        ))}
      </div>
      <div className="dependencyStrip">
        {Object.entries(deps).map(([name, dep]) => (
          <div key={name} className={classNames("depPill", dep.imported ? "ok" : "warn")}>
            {name}: {dep.imported ? dep.version || "imported" : "missing"}
          </div>
        ))}
      </div>
    </section>
  );
}

function GpuDeck({ gpus }) {
  if (!gpus.length) {
    return (
      <section className="gpuDeck">
        <div className="panel">
          <EmptyState title="No GPU detected" text="nvidia-smi did not return device utilization." />
        </div>
      </section>
    );
  }
  return (
    <section className="gpuDeck">
      {gpus.map((gpu, index) => {
        const memPct = Math.min(100, Math.max(0, (Number(gpu.memory_used_mb || 0) / Math.max(Number(gpu.memory_total_mb || 1), 1)) * 100));
        const utilPct = Math.min(100, Math.max(0, Number(gpu.utilization || 0)));
        return (
          <div key={`${gpu.name}-${index}`} className="gpuCard">
            <div className="gpuHead">
              <strong>GPU {index}</strong>
              <span>{gpu.name}</span>
            </div>
            <div className="gpuMeter">
              <div className="gpuMeterLabel">
                <span>Memory</span>
                <b>{gpu.memory_used_mb}/{gpu.memory_total_mb} MB</b>
              </div>
              <div className="meterTrack"><i style={{ width: `${memPct}%` }} /></div>
            </div>
            <div className="gpuMeter">
              <div className="gpuMeterLabel">
                <span>Utilization</span>
                <b>{utilPct}%</b>
              </div>
              <div className="meterTrack util"><i style={{ width: `${utilPct}%` }} /></div>
            </div>
          </div>
        );
      })}
    </section>
  );
}

function EnvFact({ icon, label, value }) {
  return (
    <div className="envFact">
      <span>{icon}</span>
      <div>
        <label>{label}</label>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

function Timeline({ job }) {
  const steps = timelineSteps(job);
  const stage = timelineStageId(job);
  const activeIndex = Math.max(0, steps.findIndex((item) => timelineItemMatches(item, stage, job)));
  return (
    <div className="timeline">
      {steps.map((item, index) => (
        <div key={item.id} className={classNames("timelineItem", index <= activeIndex && "done", timelineItemMatches(item, stage, job) && "current")}>
          <span>{index + 1}</span>
          <label>{item.label}</label>
        </div>
      ))}
    </div>
  );
}

function timelineSteps(job) {
  if (job?.kind === "serve") {
    return [
      { id: "created", label: "created" },
      { id: "load", label: "load", aliases: ["registered"] },
      { id: "serve", label: "serve", aliases: ["running"] },
      { id: "exit", label: "exit", aliases: ["exited", "failed", "succeeded", "stopped"] },
    ];
  }
  const algo = String(configValue(job, "algo") || "").toLowerCase();
  if (algo === "sft") {
    return [
      { id: "created", label: "created", aliases: ["registered", "epoch_start"] },
      { id: "train", label: "train", aliases: ["train_start", "train_end", "train_skip"] },
      { id: "save", label: "save", aliases: ["save_checkpoint_start", "save_checkpoint_end"] },
      { id: "done", label: "done", aliases: ["max_steps_reached", "epoch_end", "exited", "failed", "succeeded", "stopped"] },
    ];
  }
  if (algo === "dpo") {
    return [
      { id: "created", label: "created", aliases: ["registered", "epoch_start"] },
      { id: "ref_score", label: "ref score", aliases: ["logprob_score_start", "logprob_score_end"], roles: ["ref"] },
      { id: "train", label: "train", aliases: ["train_start", "train_end", "train_skip"] },
      { id: "save", label: "save", aliases: ["save_checkpoint_start", "save_checkpoint_end"] },
      { id: "done", label: "done", aliases: ["max_steps_reached", "epoch_end", "exited", "failed", "succeeded", "stopped"] },
    ];
  }
  if (algo === "ppo") {
    return [
      { id: "created", label: "created", aliases: ["registered", "epoch_start"] },
      { id: "rollout", label: "rollout", aliases: ["rollout_start", "rollout_end"], roles: ["actor"] },
      { id: "reward", label: "reward", aliases: ["score_start", "score_end"], roles: ["reward"] },
      { id: "ref_score", label: "ref score", aliases: ["logprob_score_start", "logprob_score_end"], roles: ["ref"] },
      { id: "old_logprob", label: "old logprob", aliases: ["old_logprob_score_start", "old_logprob_score_end"], roles: ["actor"] },
      { id: "critic_value", label: "value", aliases: ["value_score_start", "value_score_end"], roles: ["critic"] },
      { id: "advantage_prepare", label: "advantage", aliases: ["advantage_start"], roles: ["critic"] },
      { id: "critic_train", label: "critic train", aliases: ["train_start", "train_end"], roles: ["critic"] },
      { id: "advantage_ready", label: "advantage ready", aliases: ["advantage_end"], roles: ["critic"] },
      { id: "actor_train", label: "actor train", aliases: ["train_start", "train_end", "train_skip"], roles: ["actor"] },
      { id: "save", label: "save", aliases: ["save_checkpoint_start", "save_checkpoint_end"] },
      { id: "done", label: "done", aliases: ["max_steps_reached", "epoch_end", "exited", "failed", "succeeded", "stopped"] },
    ];
  }
  return [
    { id: "created", label: "created", aliases: ["registered", "epoch_start"] },
    { id: "rollout", label: "rollout", aliases: ["rollout_start", "rollout_end"] },
    {
      id: "score",
      label: "score",
      aliases: [
        "score_start",
        "score_end",
        "reward_score_start",
        "reward_score_end",
        "logprob_score_start",
        "logprob_score_end",
        "old_logprob_score_start",
        "old_logprob_score_end",
        "value_score_start",
        "value_score_end",
      ],
    },
    { id: "train", label: "train", aliases: ["train_start", "train_end", "train_skip"] },
    { id: "save", label: "save", aliases: ["save_checkpoint_start", "save_checkpoint_end"] },
    { id: "done", label: "done", aliases: ["max_steps_reached", "epoch_end", "exited", "failed", "succeeded", "stopped"] },
  ];
}

function timelineStageId(job) {
  const status = String(job?.status || "").toLowerCase();
  if (["succeeded", "failed", "stopped", "exited"].includes(status)) return "done";
  const stage = String(job?.stage || "created").toLowerCase();
  const steps = timelineSteps(job);
  const match = steps.find((item) => timelineItemMatches(item, stage, job));
  return match?.id || stage;
}

function timelineItemMatches(item, stage, job) {
  const normalizedStage = String(stage || "").toLowerCase();
  const stageMatches = item.id === normalizedStage || item.aliases?.includes(normalizedStage);
  if (!stageMatches) return false;
  if (!item.roles?.length) return true;
  return item.roles.includes(String(job?.role || "").toLowerCase());
}

function configValue(job, key) {
  const config = job?.config && Object.keys(job.config).length ? job.config : job?.launch || {};
  if (config[key] !== undefined) return config[key];
  for (const section of config.sections || []) {
    const item = (section.items || []).find((entry) => entry.key === key);
    if (item) return item.value;
  }
  return undefined;
}

function metricNamesFrom(metricList) {
  return (metricList || []).map((item) => item.name).sort();
}

// Mirror the dropdown's fallback so the fetched series always matches the
// option shown to the user, even when the previous selection disappears from a
// refreshed metric list.
function resolveActiveMetricName(names, selectedName) {
  return selectedName && names.includes(selectedName) ? selectedName : names[0] || "";
}

function MetricChart({ jobId, metricsDir, refreshNonce }) {
  const [selectedName, setSelectedName] = useState("");
  const [smooth, setSmooth] = useState(0.6);
  const [metricList, setMetricList] = useState([]);
  const [points, setPoints] = useState([]);
  const [metricLoading, setMetricLoading] = useState(false);
  const [pollTick, setPollTick] = useState(0);
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [prevJobId, setPrevJobId] = useState(jobId);
  // Reset the selection during render (not in an effect) when the job changes so
  // the reset happens before any effect runs. This avoids a stale-name fetch and
  // a stuck loading state on job switch, while live polls (refreshNonce) still
  // keep the user's chosen metric and chart intact.
  if (jobId !== prevJobId) {
    setPrevJobId(jobId);
    setSelectedName("");
    setMetricList([]);
    setPoints([]);
    setMetricLoading(false);
  }
  const names = metricNamesFrom(metricList);
  const effectiveName = resolveActiveMetricName(names, selectedName);
  useEffect(() => {
    if (!jobId) return undefined;
    const timer = window.setInterval(() => setPollTick((value) => value + 1), 2500);
    return () => window.clearInterval(timer);
  }, [jobId]);
  useEffect(() => {
    let cancelled = false;
    if (!jobId) return undefined;
    api(`/api/jobs/${jobId}/metrics`)
      .then((data) => {
        if (cancelled) return;
        const list = data.metrics || [];
        setMetricList(list);
        setSelectedName((current) => current || list[0]?.name || "");
      })
      .catch(() => {
        if (!cancelled) setMetricList([]);
      });
    return () => {
      cancelled = true;
    };
  }, [jobId, refreshNonce, pollTick]);
  useEffect(() => {
    let cancelled = false;
    if (!jobId || !effectiveName) {
      setPoints([]);
      setMetricLoading(false);
      return undefined;
    }
    setMetricLoading(true);
    api(`/api/jobs/${jobId}/metric?name=${encodeURIComponent(effectiveName)}`)
      .then((data) => {
        if (cancelled) return;
        setPoints((data.points || []).filter((point) => Number.isFinite(Number(point.value))).map((point) => ({
          ...point,
          step: Number(point.step || 0),
          value: Number(point.value),
        })));
      })
      .catch(() => {
        if (!cancelled) setPoints([]);
      })
      .finally(() => {
        if (!cancelled) setMetricLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [jobId, effectiveName, refreshNonce, pollTick]);
  const activeName = effectiveName;
  const visiblePoints = points;
  const smoothed = smoothTensorboard(visiblePoints, smooth);
  const smoothingEnabled = smooth > 0;
  const displayedPoints = smoothingEnabled ? smoothed : visiblePoints;
  const plot = buildMetricPlot(displayedPoints);
  return (
    <div className="chart">
      <div className="chartHeader">
        <span><Activity size={14} /> TensorBoard scalars</span>
        <div className="chartControls">
          <select value={activeName} onChange={(event) => setSelectedName(event.target.value)}>
            {names.length === 0 ? <option value="">no metrics</option> : names.map((name) => <option key={name}>{name}</option>)}
          </select>
          <label>
            smooth {smooth.toFixed(2)}
            <input type="range" min="0" max="0.99" step="0.01" value={smooth} onChange={(event) => setSmooth(Number(event.target.value))} />
          </label>
        </div>
      </div>
      {visiblePoints.length === 0 ? (
        <div className="plotEmpty">{metricLoading ? "Loading selected metric..." : "No TensorBoard scalar points loaded yet."}</div>
      ) : (
        <div className="metricPlotWrap">
          <svg className="metricPlot" viewBox="0 0 720 180" role="img">
            <g className="plotGrid">
              {[0, 1, 2, 3].map((item) => <line key={item} x1="0" x2="720" y1={30 + item * 42} y2={30 + item * 42} />)}
            </g>
            <polyline className={smoothingEnabled ? "smoothLine" : "rawLine"} points={plot.line} />
            {!smoothingEnabled && displayedPoints.map((point, index) => (
              <g key={`${point.step}-${index}`}>
                <circle className="metricDataPoint" cx={plot.coords[index]?.x || 0} cy={plot.coords[index]?.y || 0} r={metricPointRadius(displayedPoints.length)} />
                <circle className="metricHoverTarget" cx={plot.coords[index]?.x || 0} cy={plot.coords[index]?.y || 0} r="5" onMouseEnter={() => setHoveredPoint({ point, coord: plot.coords[index] })} onMouseLeave={() => setHoveredPoint(null)} />
              </g>
            ))}
          </svg>
          {!smoothingEnabled && hoveredPoint && <MetricPointTooltip name={activeName} point={hoveredPoint.point} coord={hoveredPoint.coord} width={720} height={180} />}
        </div>
      )}
      <div className="plotFooter">
        <span>{activeName || "metric"} · {points.length} points</span>
        <span>{metricsDir || "no metrics dir"} · {plot.minLabel} to {plot.maxLabel}</span>
      </div>
    </div>
  );
}

function smoothTensorboard(points, smooth) {
  if (!points.length) return [];
  const weight = Math.min(Math.max(Number(smooth) || 0, 0), 0.999);
  let last = points[0].value;
  return points.map((point) => {
    last = last * weight + point.value * (1 - weight);
    return { ...point, value: last };
  });
}

function buildMetricPlot(points) {
  if (!points.length) return { line: "", coords: [], minLabel: "n/a", maxLabel: "n/a" };
  const { min, max } = numericExtent(points, (point) => point.value);
  const flat = max === min;
  const span = Math.max(max - min, 1e-9);
  const stepMin = points[0].step;
  const stepMax = points[points.length - 1].step;
  const stepSpan = Math.max(stepMax - stepMin, 1);
  const coord = (point) => ({
    x: ((point.step - stepMin) / stepSpan) * 700 + 10,
    y: flat ? 95 : 168 - ((point.value - min) / span) * 146,
  });
  const coords = points.map(coord);
  return {
    line: coords.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" "),
    coords,
    minLabel: compactNumber(min),
    maxLabel: compactNumber(max),
  };
}

function numericExtent(values, accessor) {
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    const numeric = Number(accessor(value));
    if (numeric < min) min = numeric;
    if (numeric > max) max = numeric;
  }
  return { min, max };
}

function compactNumber(value) {
  if (!Number.isFinite(value)) return "n/a";
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) return value.toExponential(2);
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function TimePerfView({ rows, job }) {
  const normalizedRows = normalizeTimePerfRows(rows || [], job);
  const visible = sampleTimePerfRows(normalizedRows, 7).reverse();
  const avgTotal = normalizedRows.length ? normalizedRows.reduce((sum, row) => sum + Number(row.total_s || 0), 0) / normalizedRows.length : 0;
  const maxTotal = Math.max(1, ...visible.map((row) => Number(row.total_s || 0)));
  const segmentNames = Array.from(new Set(normalizedRows.flatMap((row) => (row.segments || []).map((segment) => segment.name)))).slice(0, 8);
  return (
    <div className="timePerf">
      <div className="timePerfHeader">
        <div>
          <div className="codeTitle inline"><Timer size={14} /> Runtime timeperf</div>
        </div>
        <div className="legend">
          {segmentNames.map((name) => <span key={name}><i className={segmentClass(name)} /> {name}</span>)}
          <b>avg {avgTotal ? `${avgTotal.toFixed(1)}s` : "n/a"}</b>
        </div>
      </div>
      {visible.length === 0 ? (
        <div className="sampleEmpty">No step timing captured yet.</div>
      ) : (
        <div className="timeRows">
          {visible.map((row) => {
            const total = Number(row.total_s || 0);
            const width = Math.max(10, (total / maxTotal) * 100);
            const segments = row.segments?.length
              ? row.segments
              : [];
            return (
              <div key={`${row.step}-${row.time}`} className="timeRow">
                <label>step {row.step}</label>
                <div className="timeTrack">
                  <div className="timeStack" style={{ width: `${width}%` }}>
                    {segments.map((segment) => {
                      const seconds = Number(segment.seconds || 0);
                      const pct = (seconds / Math.max(total, 1)) * 100;
                      return (
                        <span
                          key={segment.name}
                          className={segmentClass(segment.name)}
                          style={{ width: `${pct}%` }}
                          title={`${segment.name}: ${seconds.toFixed(1)}s`}
                        >
                          {pct > 11 ? `${seconds.toFixed(1)}s` : ""}
                        </span>
                      );
                    })}
                  </div>
                </div>
                <strong>{total.toFixed(1)}s</strong>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function normalizeTimePerfRows(rows, job) {
  const algo = String(configValue(job, "algo") || "").toLowerCase();
  const valid = timePerfSegmentsForAlgorithm(algo, job?.kind);
  return rows.map((row) => {
    const kept = [];
    let other = 0;
    for (const segment of row.segments || []) {
      const seconds = Number(segment.seconds || 0);
      if (valid.has(segment.name)) kept.push({ ...segment, seconds });
      else other += seconds;
    }
    if (other > 0) {
      const existing = kept.find((segment) => segment.name === "other");
      if (existing) existing.seconds += other;
      else kept.push({ name: "other", seconds: other });
    }
    return { ...row, segments: kept };
  });
}

function timePerfSegmentsForAlgorithm(algo, kind) {
  if (kind === "serve") return new Set(["load", "prefill", "decode", "other"]);
  if (algo === "sft") return new Set(["train", "save", "other"]);
  if (algo === "dpo") return new Set(["ref log probs", "train", "save", "other"]);
  if (algo === "ppo") return new Set(["rollout", "make_sample", "reward", "old policy log probs", "actor log probs", "ref log probs", "value", "advantages", "sync weight", "train", "save", "other"]);
  return new Set(["rollout", "make_sample", "reward", "old policy log probs", "actor log probs", "ref log probs", "advantages", "sync weight", "train", "save", "other"]);
}

function sampleTimePerfRows(rows, limit) {
  if (rows.length <= limit) return rows.slice();
  if (limit <= 1) return rows.slice(-1);
  const lastIndex = rows.length - 1;
  const selected = new Set();
  for (let index = 0; index < limit; index += 1) {
    selected.add(Math.round((index * lastIndex) / (limit - 1)));
  }
  selected.add(lastIndex);
  return Array.from(selected)
    .sort((a, b) => a - b)
    .slice(-limit)
    .map((index) => rows[index]);
}

function segmentClass(name) {
  return `seg-${String(name || "other").replace(/[^a-zA-Z0-9]+/g, "-")}`;
}

function normalizeDisplayName(name) {
  return String(name || "")
    .replace(/_time_s$/, "")
    .replace(/^step_/, "")
    .replace(/_/g, " ")
    .trim();
}

function sampleOutput(sample) {
  for (const value of [sample.completion, sample.rendered_completion, sample.final_answer]) {
    if (typeof value === "string" && value.trim()) return value;
  }
  if (Array.isArray(sample.tool_calls) && sample.tool_calls.length) {
    return JSON.stringify(sample.tool_calls, null, 2);
  }
  return "No assistant output was captured.";
}

function samplePromptMessages(sample) {
  if (Array.isArray(sample.prompt_messages)) return sample.prompt_messages;
  if (!Array.isArray(sample.messages)) return [];
  const messages = [...sample.messages];
  if (messages.at(-1)?.role === "assistant") messages.pop();
  return messages;
}

function messageMediaPart(part) {
  if (!part || typeof part !== "object") return null;
  const rawType = String(part.type || "");
  if (rawType === "input_audio") {
    const input = part.input_audio;
    if (!input?.data || String(input.data).includes("<truncated>")) return null;
    const format = String(input.format || "wav").toLowerCase();
    const mime = format === "mp3" ? "audio/mpeg" : `audio/${format}`;
    const source = String(input.data).startsWith("data:") ? input.data : `data:${mime};base64,${input.data}`;
    return { type: "audio", source };
  }
  const type = rawType.replace(/_url$/, "");
  if (!["image", "audio", "video"].includes(type)) return null;
  let source = part[rawType] ?? part[type] ?? part.url;
  if (source && typeof source === "object") source = source.url;
  return typeof source === "string" && source ? { type, source } : null;
}

function sampleMedia(sample) {
  const items = [];
  const add = (item) => {
    if (item && !items.some((existing) => existing.type === item.type && existing.source === item.source)) items.push(item);
  };
  for (const message of samplePromptMessages(sample)) {
    if (!Array.isArray(message?.content)) continue;
    for (const part of message.content) add(messageMediaPart(part));
  }
  const walkRecord = (value, key = "") => {
    if (Array.isArray(value)) {
      value.forEach((item) => walkRecord(item, key));
      return;
    }
    if (value && typeof value === "object") {
      Object.entries(value).forEach(([itemKey, item]) => walkRecord(item, itemKey));
      return;
    }
    if (typeof value !== "string") return;
    const normalized = key.toLowerCase();
    const type = ["video", "audio", "image"].find((kind) => normalized.includes(kind));
    const directMediaKey = ["video", "videos", "audio", "audios", "image", "images"].includes(normalized);
    if (type && (directMediaKey || ["path", "url", "file"].some((marker) => normalized.includes(marker)))) {
      add({ type, source: value });
    }
  };
  walkRecord(sample.source_record);
  return items;
}

function sampleMediaUrl(jobId, source) {
  if (/^(data:|blob:|https?:)/.test(source)) return source;
  return `${API_BASE}api/jobs/${encodeURIComponent(jobId)}/media?path=${encodeURIComponent(source)}`;
}

function SampleMedia({ jobId, media }) {
  if (!jobId || !media.length) return null;
  const hasVideo = media.some((item) => item.type === "video");
  const hasAudio = media.some((item) => item.type === "audio");
  const label = hasVideo && hasAudio ? "Video + audio" : hasVideo ? "Video" : hasAudio ? "Audio" : "Image";
  return (
    <section className="sampleMediaSection">
      <div className="sampleSectionLabel">
        {hasVideo ? <Film size={14} /> : hasAudio ? <Volume2 size={14} /> : <ImageIcon size={14} />}
        {label}
      </div>
      <div className={classNames("sampleMediaGrid", media.length === 1 && "single")}>
        {media.map((item, index) => {
          const source = sampleMediaUrl(jobId, item.source);
          if (item.type === "video") return <video key={`${item.type}-${item.source}`} src={source} controls preload="metadata" />;
          if (item.type === "audio") return <audio key={`${item.type}-${item.source}`} src={source} controls preload="metadata" />;
          return <img key={`${item.type}-${item.source}`} src={source} alt={`Sample media ${index + 1}`} loading="lazy" />;
        })}
      </div>
    </section>
  );
}

function JsonSampleSection({ icon, label, value, emptyText }) {
  return (
    <section className="sampleDetailSection">
      <div className="sampleSectionLabel">{icon}{label}</div>
      <pre>{value == null ? emptyText : JSON.stringify(value, null, 2)}</pre>
    </section>
  );
}

function SampleView({ samples, jobId, hideTitle = false }) {
  const orderedSamples = useMemo(
    () =>
      [...(samples || [])].sort(
        (left, right) =>
          Number(left.step || 0) - Number(right.step || 0) ||
          Number(left.prompt_idx || 0) - Number(right.prompt_idx || 0) ||
          Number(left.sample_idx || 0) - Number(right.sample_idx || 0)
      ),
    [samples]
  );
  const stepOptions = useMemo(
    () => Array.from(new Set(orderedSamples.map((sample) => Number(sample.step || 0)))).sort((a, b) => b - a),
    [orderedSamples]
  );
  const [selectedStep, setSelectedStep] = useState("");
  const [selectedSampleKey, setSelectedSampleKey] = useState("");
  useEffect(() => {
    if (!stepOptions.length) {
      setSelectedStep("");
      setSelectedSampleKey("");
      return;
    }
    if (selectedStep === "" || !stepOptions.includes(Number(selectedStep))) {
      setSelectedStep(String(stepOptions[0]));
      setSelectedSampleKey("");
    }
  }, [selectedStep, stepOptions]);
  const stepSamples = selectedStep === "" ? [] : orderedSamples.filter((sample) => Number(sample.step || 0) === Number(selectedStep));
  const sampleOptions = stepSamples.map((sample, index) => ({
    key: sampleKey(sample, index),
    label: `prompt ${sample.prompt_idx ?? "?"} · sample ${sample.sample_idx ?? "?"}`,
    sample,
  }));
  const activeOption =
    sampleOptions.find((option) => option.key === selectedSampleKey) || sampleOptions[sampleOptions.length - 1] || null;
  const sample = activeOption?.sample || null;
  const media = useMemo(() => (sample ? sampleMedia(sample) : []), [sample]);
  const promptMessages = sample ? samplePromptMessages(sample) : [];
  return (
    <div className="sampleCard">
      {!hideTitle && <div className="codeTitle sampleTitle">
        <span><FileText size={14} /> Rollout sample</span>
        {!!orderedSamples.length && (
          <div className="sampleControls">
            <select value={selectedStep} onChange={(event) => { setSelectedStep(event.target.value); setSelectedSampleKey(""); }}>
              {stepOptions.map((step) => <option key={step} value={step}>step {step}</option>)}
            </select>
            <select value={activeOption?.key || ""} onChange={(event) => setSelectedSampleKey(event.target.value)}>
              {sampleOptions.map((option) => <option key={option.key} value={option.key}>{option.label}</option>)}
            </select>
          </div>
        )}
      </div>}
      {sample ? (
        <div className="sampleContent">
          <SampleMedia jobId={jobId} media={media} />
          <div className="sampleGrid">
            <div>
              <span>Prompt</span>
              <p>{sample.prompt || "No prompt text was captured."}</p>
            </div>
            <div>
              <span>Output</span>
              <p>{sampleOutput(sample)}</p>
            </div>
          </div>
          <div className="sampleDetailGrid">
            <JsonSampleSection
              icon={<FileText size={14} />}
              label="Full prompt"
              value={promptMessages.length ? promptMessages : null}
              emptyText="No structured prompt messages were captured."
            />
            <JsonSampleSection
              icon={<Database size={14} />}
              label="Data record"
              value={sample.source_record}
              emptyText="No source dataset record was captured."
            />
          </div>
        </div>
      ) : (
        <div className="sampleEmpty">No rollout sample captured yet.</div>
      )}
    </div>
  );
}

function sampleKey(sample, index) {
  return `${sample.step ?? "x"}-${sample.prompt_idx ?? "x"}-${sample.sample_idx ?? "x"}-${index}`;
}

function ConfigView({ config, launch }) {
  const settings = config && Object.keys(config).length ? config : launch || {};
  const sections = normalizeConfigSections(settings);
  return (
    <div className="codeCard">
      <div className="codeTitle"><Settings2 size={14} /> Config</div>
      {sections.length > 0 ? (
        <div className="configSections">
          {sections.map((section) => (
            <div key={section.title} className="configSection">
              <h3>{section.title}</h3>
              <div className="configGrid">
                {section.items.map(({ key, value }) => (
                  <div key={key} className="configItem">
                    <span>{key.replace(/_/g, " ")}</span>
                    <strong>{formatConfigValue(value)}</strong>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <pre>No config captured yet.</pre>
      )}
    </div>
  );
}

function normalizeConfigSections(settings) {
  if (Array.isArray(settings?.sections)) {
    return settings.sections
      .map((section) => ({
        title: section.title || "Config",
        items: (section.items || []).filter(({ value }) => value !== undefined && value !== null && value !== ""),
      }))
      .filter((section) => section.items.length > 0);
  }
  const entries = Object.entries(settings || {})
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .map(([key, value]) => ({ key, value }));
  return entries.length ? [{ title: "Launch", items: entries }] : [];
}

function formatConfigValue(value) {
  if (Array.isArray(value)) return value.join(" ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function LogView({ logs }) {
  const logRef = useRef(null);
  useEffect(() => {
    const node = logRef.current;
    if (node) {
      node.scrollTop = node.scrollHeight;
    }
  }, [logs.length]);
  return (
    <div className="codeCard" ref={logRef}>
      <div className="codeTitle"><TerminalSquare size={14} /> Logs</div>
      <pre>{logs.slice(-80).join("\n") || "No logs yet."}</pre>
    </div>
  );
}

function LauncherPrdPage({ mode, setMode, trainConfig, setTrainConfig, serveConfig, setServeConfig, onStartTrain, onStartServe, env, presets }) {
  const config = mode === "train" ? trainConfig : serveConfig;
  const [preflightResult, setPreflightResult] = useState(null);
  const [preflightBusy, setPreflightBusy] = useState("");
  const [preflightJob, setPreflightJob] = useState(null);
  const worldSize = Number(config.world_size || 0);
  const tpSize = Number(config.tp_size || 0);
  const gpuCount = env?.gpus?.length || 0;
  const smokeTrainTunable = mode === "train";
  const smokeInferTunable = smokeTrainTunable && ["gspo", "grpo", "ppo"].includes(String(config.algo || "").toLowerCase());
  useEffect(() => {
    if (!preflightJob?.job?.id || !["created", "running"].includes(preflightJob.job.status)) return undefined;
    let cancelled = false;
    const refresh = async () => {
      try {
        const result = await api(`/api/jobs/${preflightJob.job.id}`);
        if (cancelled || !result.job) return;
        const updated = { ...preflightJob, job: result.job };
        setPreflightJob(updated);
        if (!["created", "running"].includes(result.job.status)) {
          setPreflightResult({ ...updated, ok: result.job.status === "succeeded", output: (result.job.logs || []).join("\n") });
        }
      } catch (error) {
        if (!cancelled) setPreflightResult({ ok: false, check: preflightJob.check, output: error.message });
      }
    };
    refresh();
    const timer = window.setInterval(refresh, 1000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [preflightJob?.job?.id, preflightJob?.job?.status]);
  const checks = [
    {
      id: "gpu_count",
      name: "GPU count",
      status: gpuCount === 0 ? "warn" : worldSize <= gpuCount ? "ok" : "warn",
      detail: gpuCount ? `World size ${worldSize} uses ${gpuCount} visible GPU${gpuCount === 1 ? "" : "s"}.` : "No visible GPU inventory is available.",
      tunable: smokeInferTunable && gpuCount > 0,
    },
    {
      id: "tensor_parallel",
      name: "Tensor parallelism",
      status: worldSize > 0 && tpSize > 0 && worldSize % tpSize === 0 ? "ok" : "fail",
      detail: worldSize > 0 && tpSize > 0 && worldSize % tpSize === 0 ? `World size ${worldSize} is divisible by TP size ${tpSize}.` : "World size must be divisible by TP size.",
      tunable: smokeTrainTunable,
    },
    mode === "train" ? {
      id: "batch_relation",
      name: "Batch relation",
      status: Number(config.batch_size) > 0 && Number(config.mini_bs) > 0 && Number(config.batch_size) % Number(config.mini_bs) === 0 ? "ok" : "fail",
      detail: `Batch ${config.batch_size} × samples ${config.n_samples || 1}; mini batch ${config.mini_bs}.`,
      tunable: smokeTrainTunable,
    } : {
      id: "batch_relation",
      name: "Serving capacity",
      status: Number(config.max_running_prompts) > 0 ? "ok" : "warn",
      detail: `${config.max_running_prompts || 0} concurrent prompts configured.`,
      tunable: false,
    },
    ...(mode === "train" ? [{
      id: "max_new_tokens",
      name: "Max new tokens",
      status: Number(config.max_new_tokens) > 1024 ? "warn" : "ok",
      detail: Number(config.max_new_tokens) > 1024 ? `${config.max_new_tokens} may increase rollout memory.` : `${config.max_new_tokens || 0} is within the preflight target.`,
      tunable: smokeInferTunable,
    }] : []),
  ];
  const command = launcherCommand(mode, config);
  const hasFailure = checks.some((check) => check.status === "fail");
  const runPreflightAction = async (check) => {
    const action = check.status !== "ok" && check.tunable ? "tune" : "view";
    setPreflightBusy(check.id);
    try {
      const result = await api("/api/launcher/preflight", { method: "POST", body: JSON.stringify({ mode, config, check_id: check.id, action }) });
      if (action === "tune" && result.job) {
        setPreflightResult(null);
        setPreflightJob({ ...result, check, job: result.job });
      } else {
        setPreflightResult(result);
      }
    } catch (error) {
      setPreflightResult({ ok: false, check: { name: check.name, status: "fail", detail: error.message }, patch: {} });
    } finally {
      setPreflightBusy("");
    }
  };
  return (
    <>
    <div className="launcherPrdLayout">
      <section className="panel launcher launcherMainCard">
        <div className="panelHeader">
          <div><h2>Task Launcher</h2><p>Configure, validate, and review the generated command before launch.</p></div>
          <div className="tabs"><button className={classNames(mode === "train" && "active")} onClick={() => setMode("train")}>Train</button><button className={classNames(mode === "serve" && "active")} onClick={() => setMode("serve")}>Serve</button></div>
        </div>
        {mode === "train" && presets.length > 0 && <div className="launcherPresetRow">
          {presets.map((preset) => <button key={preset.id} className="presetPill" title={preset.source} onClick={() => setTrainConfig((current) => ({ ...current, ...(preset.preset || {}) }))}>{preset.label}</button>)}
        </div>}
        <div className="launcherFormScroll">
          {mode === "train" ? <TrainForm config={trainConfig} setConfig={setTrainConfig} onStart={onStartTrain} /> : <ServeForm config={serveConfig} setConfig={setServeConfig} onStart={onStartServe} />}
        </div>
      </section>
      <aside className="launcherSideRail">
        <section className="panel launcherPreflight">
          <div className="panelHeader"><div><h2>Preflight</h2><p>Resolve launch risks before allocating workers.</p></div><StatusBadge status={hasFailure ? "failed" : "ok"} /></div>
          <div className="runtimeCheckList">{checks.map((check) => {
            const launching = preflightBusy === check.id && check.status !== "ok" && check.tunable;
            const tuning = preflightJob?.check_id === check.id && ["created", "running"].includes(preflightJob.job?.status);
            const smokeLabel = preflightJob?.smoke_stage === "infer" ? "Smoke infer" : "Smoke train";
            const elapsed = tuning && preflightJob.job?.created_at ? Math.max(0, Math.floor((Date.now() - Date.parse(preflightJob.job.created_at)) / 1000)) : 0;
            const latestLog = tuning ? [...(preflightJob.job?.logs || [])].reverse().find((line) => line && !String(line).startsWith("$ ")) : "";
            const progressText = launching ? "Starting smoke tuning job..." : tuning ? `${smokeLabel}: ${preflightJob.job?.stage || "starting"} · step ${preflightJob.job?.step ?? 0} · ${elapsed}s` : check.detail;
            return <div className="runtimeCheckRow launcherCheck" key={check.id}><StatusBadge status={tuning ? "running" : check.status} /><div><strong>{check.name}</strong><p>{progressText}</p>{(launching || tuning) && <div className="preflightProgress" role="progressbar" aria-label={`Tuning ${check.name}`}><span /></div>}{tuning && <div className="preflightTuneDetails"><div>{Object.entries(preflightJob.tuning_params || {}).map(([key, value]) => <span key={key}><b>{key.replaceAll("_", " ")}</b>{String(value)}</span>)}</div>{latestLog && <code>{latestLog}</code>}</div>}</div><button className="secondaryButton tableAction" disabled={Boolean(preflightBusy) || Boolean(preflightJob && ["created", "running"].includes(preflightJob.job?.status))} onClick={() => runPreflightAction(check)}>{launching ? "Starting..." : tuning ? `${smokeLabel}...` : check.status !== "ok" && check.tunable ? "Tune" : "View"}</button></div>;
          })}</div>
        </section>
        <section className="panel commandPreviewPanel">
          <div className="panelHeader"><div><h2>Command Preview</h2><p>Review the final CLI mapping.</p></div><button className="secondaryButton" onClick={() => navigator.clipboard.writeText(command)}>Copy</button></div>
          <pre className="commandPreview">{command}</pre>
        </section>
      </aside>
    </div>
    {preflightResult && <Modal title="Preflight Result" onClose={() => setPreflightResult(null)}>
      <div className="runtimeResultSummary"><StatusBadge status={preflightResult.job?.status || preflightResult.check?.status || (preflightResult.ok ? "ok" : "failed")} /><span>{preflightResult.check?.name || "Launcher check"}</span></div>
      {Object.keys(preflightResult.tuning_params || {}).length > 0 && <div className="preflightResultParams">{Object.entries(preflightResult.tuning_params).map(([key, value]) => <span key={key}><b>{key.replaceAll("_", " ")}</b>{String(value)}</span>)}</div>}
      <pre className="runtimeCommandResult">{preflightResult.command?.length ? `$ ${preflightResult.command.map(shellQuote).join(" ")}\n\n` : ""}{preflightResult.output || preflightResult.check?.detail || "No details returned."}</pre>
      <button className="primaryButton fullButton" onClick={() => setPreflightResult(null)}>Done</button>
    </Modal>}
    </>
  );
}

function launcherCommand(mode, config) {
  const pairs = mode === "train"
    ? [["algo", config.algo], ["ckpt", config.ckpt], ["dataset-path", config.dataset_path], ["dataset-loader-fn", config.dataset_loader_fn], ["reward-fn-path", config.reward_fn_path], ["world-size", config.world_size], ["tp-size", config.tp_size], ["batch-size", config.batch_size], ["mini-bs", config.mini_bs], ["n-samples", config.n_samples], ["max-new-tokens", config.max_new_tokens]]
    : [["model-path", config.model_path], ["host", config.host], ["port", config.port], ["world-size", config.world_size], ["tp-size", config.tp_size], ["max-running-prompts", config.max_running_prompts], ["default-max-tokens", config.default_max_tokens]];
  const args = pairs.filter(([, value]) => value !== "" && value !== undefined && value !== null).map(([name, value]) => `  --${name} ${shellQuote(value)}`);
  return [`areno ${mode} \\`, ...args.map((line, index) => `${line}${index < args.length - 1 ? " \\" : ""}`)].join("\n");
}

function shellQuote(value) {
  const text = String(value);
  return /^[a-zA-Z0-9_./:@+-]+$/.test(text) ? text : `'${text.replaceAll("'", `'\\''`)}'`;
}

function TrainForm({ config, setConfig, onStart }) {
  const algo = String(config.algo || "sft").toLowerCase();
  const sections = trainLauncherSections(algo);
  const updateField = (key, value) => setConfig({ ...config, [key]: value });
  const primaryFields = [
    selectField("algo", "Algorithm", ["sft", "dpo", "gspo", "grpo", "ppo"], true),
    field("ckpt", "Checkpoint"),
    field("dataset_path", "Dataset path"),
    field("dataset_loader_fn", "Dataset loader"),
    field("reward_fn_path", "Reward function"),
    selectField("model_hub", "Model hub", ["modelscope", "hf"], true),
    field("world_size", "World size", true),
    field("tp_size", "TP size", true),
    field("batch_size", "Batch size", true),
    field("mini_bs", "Mini batch size", true),
    field("n_samples", "N samples", true),
    field("max_new_tokens", "Max new tokens", true),
  ];
  const primaryKeys = new Set(primaryFields.map((item) => item.key));
  const advancedSections = sections.map((section) => ({ ...section, fields: section.fields.filter((item) => !primaryKeys.has(item.key)) })).filter((section) => section.fields.length);
  const renderLauncherField = (item) => <Field key={item.key} label={item.label} value={config[item.key]} onChange={(value) => updateField(item.key, value)} compact={item.compact} type={item.type} options={item.options} />;
  return (
    <div className="launcherSections">
      <div className="formGrid launcherPrimaryFields">{primaryFields.map(renderLauncherField)}</div>
      <details className="launcherAdvanced">
        <summary>Advanced settings</summary>
        <div className="launcherAdvancedBody">{advancedSections.map((section) => <div className="launcherSection" key={section.title}><div className="launcherSectionHeader"><strong>{section.title}</strong>{section.note && <span>{section.note}</span>}</div><div className="formGrid">{section.fields.map(renderLauncherField)}</div></div>)}</div>
      </details>
      <button className="primaryButton launchButton wide" onClick={onStart}><Play size={16} /> Start train</button>
    </div>
  );
}

function trainLauncherSections(algo) {
  const isRollout = ["gspo", "grpo", "ppo"].includes(algo);
  const isAgentic = isRollout;
  const isDpo = algo === "dpo";
  const isPpo = algo === "ppo";
  const isGspo = algo === "gspo";
  const isGrpo = algo === "grpo";
  const sections = [
    {
      title: "Basic",
      note: "model, data, and trainer loop",
      fields: [
        selectField("algo", "Algorithm", ["sft", "dpo", "gspo", "grpo", "ppo"], true),
        field("ckpt", "Checkpoint"),
        selectField("model_hub", "Model hub", ["modelscope", "hf"], true),
        field("dataset_path", "Dataset path"),
        field("dataset_loader_fn", "Dataset loader"),
        field("epochs", "Epochs", true),
        field("max_steps", "Max steps", true),
      ],
    },
    {
      title: "Runtime",
      note: "parallelism, memory, and kernels",
      fields: [
        field("world_size", "World", true),
        field("tp_size", "TP", true),
        selectField("attn_backend", "Attention", ["flash", "native"], true),
        checkField("activation_checkpointing", "Activation ckpt"),
        checkField("drop_rollout_state", "Drop rollout state"),
        checkField("eager_decode", "Eager decode"),
        checkField("disable_thinking", "Disable thinking"),
      ],
    },
    {
      title: "Batching",
      note: "controls train and rollout memory",
      fields: [
        field("batch_size", "Batch", true),
        ...(isRollout ? [field("n_samples", "Samples", true), field("max_running_prompts", "Running prompts", true)] : []),
        field("mini_bs", "Mini BS", true),
        field("score_micro_bs", "Score micro BS", true),
        field("gradient_accumulation_steps", "Grad accum", true),
      ],
    },
    {
      title: isRollout ? "Rollout" : "Sequence",
      note: isRollout ? "generation and sampling" : "token limits for supervised data",
      fields: [
        field("max_prompt_tokens", "Prompt tokens", true),
        field("max_new_tokens", "New tokens", true),
        ...(isAgentic ? [field("max_context_len", "Context", true), field("agent_fn", "Agent fn"), field("agent_timeout_s", "Agent timeout", true), checkField("train_tool_results", "Train tool results")] : []),
        ...(isRollout ? [field("temperature", "Temp", true), field("top_k", "Top K", true), field("top_p", "Top P", true), checkField("greedy", "Greedy")] : []),
      ],
    },
    {
      title: "Optimizer",
      note: "policy optimizer settings",
      fields: [
        field("lr", "LR", true),
        field("min_lr", "Min LR", true),
        field("lr_decay_steps", "Decay steps", true),
        field("lr_decay_style", "Decay style", true),
        field("adam_beta1", "Adam beta1", true),
        field("adam_beta2", "Adam beta2", true),
        field("weight_decay", "Weight decay", true),
        field("grad_clip_norm", "Grad clip", true),
        checkField("adam_8bit", "8-bit Adam"),
        checkField("unfreeze_multimodal_tower", "Train media tower"),
        field("multimodal_tower_lr", "Tower LR", true),
        field("multimodal_tower_min_lr", "Tower min LR", true),
        field("multimodal_tower_lr_decay_steps", "Tower decay", true),
        selectField("multimodal_tower_lr_decay_style", "Tower schedule", ["", "constant", "linear", "cosine"]),
        checkField("unfreeze_multimodal_projector", "Train media projector"),
        field("multimodal_projector_lr", "Projector LR", true),
        field("multimodal_projector_min_lr", "Projector min LR", true),
        field("multimodal_projector_lr_decay_steps", "Projector decay", true),
        selectField("multimodal_projector_lr_decay_style", "Projector schedule", ["", "constant", "linear", "cosine"]),
      ],
    },
  ];

  const roleFields = [];
  if (isDpo || isPpo) roleFields.push(field("ref_ckpt", "Reference ckpt"));
  if (isRollout) roleFields.push(field("reward_fn_path", "Reward fn"));
  if (isPpo) roleFields.push(field("reward_ckpt", "Reward ckpt"), field("critic_ckpt", "Critic ckpt"), field("critic_lr", "Critic LR", true), field("critic_warmup_steps", "Critic warmup", true));
  if (isGspo) roleFields.push(field("gspo_clip_eps", "GSPO clip", true));
  if (isGrpo) roleFields.push(field("grpo_clip_eps", "GRPO clip", true));
  if (isDpo) roleFields.push(field("dpo_beta", "DPO beta", true));
  if (isPpo) {
    roleFields.push(
      checkField("use_kl_loss", "Use KL loss"),
      field("kl_loss_coef", "KL coef", true),
      field("kl_loss_type", "KL type", true),
      field("clip_eps", "Clip eps", true),
      field("clip_ratio_c", "Clip ratio C", true),
      field("value_clip_eps", "Value clip", true),
      field("value_loss_coef", "Value coef", true),
      field("gamma", "Gamma", true),
      field("lam", "Lambda", true),
    );
  }
  if (roleFields.length > 0) {
    sections.push({ title: "Algorithm", note: `${algo.toUpperCase()}-specific roles and loss`, fields: roleFields });
  }

  sections.push(
    {
      title: "Probe",
      note: "optional smoke and auto tune helpers",
      fields: [
        checkField("tune_params", "Tune params"),
        field("mem_frac", "Memory frac", true),
        field("tune_max_samples", "Tune samples", true),
      ],
    },
    {
      title: "Output",
      note: "checkpointing, metrics, and escape hatch",
      fields: [
        field("save_path", "Save path"),
        field("save_interval", "Save interval", true),
        field("metrics_dir", "Metrics dir"),
        field("extra_args", "Extra args"),
      ],
    },
  );
  return sections;
}

function field(key, label, compact = false) {
  return { key, label, compact };
}

function selectField(key, label, options, compact = false) {
  return { key, label, compact, type: "select", options };
}

function checkField(key, label) {
  return { key, label, compact: true, type: "checkbox" };
}

function ServeForm({ config, setConfig, onStart }) {
  return (
    <div className="formGrid">
      {[
        ["model_path", "Model path"],
        ["model_hub", "Model hub"],
        ["host", "Host"],
        ["port", "Port"],
        ["world_size", "World"],
        ["tp_size", "TP"],
        ["max_running_prompts", "Running prompts"],
        ["default_max_tokens", "Default max tokens"],
        ["decode_progress_interval_s", "Progress interval"],
        ["attn_backend", "Attention backend"],
        ["eager_decode", "Eager decode"],
        ["disable_thinking", "Disable thinking"],
        ["extra_args", "Extra args"],
      ].map(([key, label]) => <Field key={key} label={label} value={config[key]} onChange={(value) => setConfig({ ...config, [key]: value })} compact={key !== "model_path"} />)}
      <button className="primaryButton launchButton wide" onClick={onStart}><Play size={16} /> Start serve</button>
    </div>
  );
}

function Field({ label, value, onChange, compact, type = "text", options = [] }) {
  if (type === "checkbox") {
    return (
      <label className={classNames("field", "compact", "checkField")}>
        <input type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(event.target.checked)} />
        <span>{label}</span>
      </label>
    );
  }
  return (
    <label className={classNames("field", compact && "compact")}>
      <span>{label}</span>
      {type === "select" ? (
        <select value={value ?? ""} onChange={(event) => onChange(event.target.value)}>
          {options.map((option) => <option key={option} value={option}>{option}</option>)}
        </select>
      ) : (
        <input value={value ?? ""} onChange={(event) => onChange(event.target.value)} />
      )}
    </label>
  );
}

function EmptyState({ title, text }) {
  return (
    <div className="empty">
      <Box size={18} />
      <strong>{title}</strong>
      <span>{text}</span>
    </div>
  );
}

const rootElement = typeof document !== "undefined" ? document.getElementById("root") : null;
if (rootElement) {
  createRoot(rootElement).render(<App />);
}

export { App, MetricChart, metricNamesFrom, resolveActiveMetricName };
