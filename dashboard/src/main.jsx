import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Activity,
  Bot,
  Box,
  CircleStop,
  Cpu,
  FileText,
  History,
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
} from "lucide-react";
import "./styles.css";

const API_BASE = new URL(".", window.location.href).pathname;

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

const AGENT_CHAT_STORAGE_KEY = "areno-dashboard-agent-chat";
const AGENT_SESSIONS_STORAGE_KEY = "areno-dashboard-agent-chat-sessions";
const AGENT_ACTIVE_SESSION_STORAGE_KEY = "areno-dashboard-agent-active-chat";
const AGENT_DRAFT_STORAGE_KEY = "areno-dashboard-agent-draft";

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
  };
}

function loadAgentSessions() {
  try {
    const parsed = JSON.parse(localStorage.getItem(AGENT_SESSIONS_STORAGE_KEY) || "[]");
    if (Array.isArray(parsed) && parsed.length) {
      return parsed.map((session) => ({ ...session, messages: session.messages?.length ? session.messages : defaultAgentMessages }));
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
  ckpt: "Qwen/Qwen3-0.6B",
  dataset_path: "yahma/alpaca-cleaned:train",
  dataset_loader_fn: "examples/sft/alpaca/dataset_loader.py",
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
  model_path: "Qwen/Qwen3-0.6B",
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
  const [activePage, setActivePage] = useState("jobs");
  const [launcherMode, setLauncherMode] = useState("train");
  const [theme, setTheme] = useState(() => localStorage.getItem("areno-dashboard-theme") || "dark");
  const [busy, setBusy] = useState("");
  const [agentSettingsOpen, setAgentSettingsOpen] = useState(false);
  const [jobPage, setJobPage] = useState(1);
  const [refreshNonce, setRefreshNonce] = useState(0);
  const chatMessagesRef = useRef(null);
  const env = usePolling(() => api("/api/env"), 5000);
  const jobs = usePolling(() => api("/api/jobs"), 2000);
  const jobDetail = usePolling(() => selectedJobId ? api(`/api/jobs/${selectedJobId}`) : Promise.resolve(null), 3000, [selectedJobId]);

  const jobList = jobs.data?.jobs || [];
  const jobPageSize = 10;
  const jobPageCount = Math.max(1, Math.ceil(jobList.length / jobPageSize));
  const currentJobPage = Math.min(jobPage, jobPageCount);
  const pagedJobs = jobList.slice((currentJobPage - 1) * jobPageSize, currentJobPage * jobPageSize);
  const selectedJob = jobDetail.data?.job || (selectedJobId ? jobList.find((job) => job.id === selectedJobId) : null) || null;
  const activeAgentSession = useMemo(() => {
    return agentSessions.find((session) => session.id === activeAgentSessionId) || agentSessions[0] || createAgentSession();
  }, [agentSessions, activeAgentSessionId]);
  const agentMessages = activeAgentSession.messages || defaultAgentMessages;

  useEffect(() => {
    if (selectedJobId && jobList.length && !jobList.some((job) => job.id === selectedJobId)) {
      setSelectedJobId(null);
    }
  }, [jobList.length, selectedJobId]);

  useEffect(() => {
    if (jobPage > jobPageCount) setJobPage(jobPageCount);
  }, [jobPage, jobPageCount]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("areno-dashboard-theme", theme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem("areno-dashboard-agent-provider", JSON.stringify(agentProvider));
  }, [agentProvider]);

  useEffect(() => {
    localStorage.setItem(AGENT_SESSIONS_STORAGE_KEY, JSON.stringify(agentSessions.slice(-40)));
  }, [agentSessions]);

  useEffect(() => {
    if (!agentSessions.length) {
      const session = createAgentSession();
      setAgentSessions([session]);
      setActiveAgentSessionId(session.id);
      return;
    }
    if (!agentSessions.some((session) => session.id === activeAgentSessionId)) {
      setActiveAgentSessionId(agentSessions[0].id);
    }
  }, [agentSessions, activeAgentSessionId]);

  useEffect(() => {
    if (activeAgentSessionId) {
      localStorage.setItem(AGENT_ACTIVE_SESSION_STORAGE_KEY, activeAgentSessionId);
    }
  }, [activeAgentSessionId]);

  useEffect(() => {
    localStorage.setItem(AGENT_DRAFT_STORAGE_KEY, agentPrompt);
  }, [agentPrompt]);

  useEffect(() => {
    const node = chatMessagesRef.current;
    if (node) {
      node.scrollTop = node.scrollHeight;
    }
  }, [agentMessages, agentChatTab]);

  const pages = [
    { id: "jobs", label: "Jobs", icon: <Activity size={16} /> },
    { id: "runtime", label: "Runtime", icon: <Server size={16} /> },
    { id: "launcher", label: "Launcher", icon: <Play size={16} /> },
    { id: "agent", label: "Agent", icon: <Bot size={16} /> },
  ];
  const pageCopy = {
    jobs: selectedJob
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

  async function stopJob(id) {
    setBusy("Stopping job...");
    try {
      await api(`/api/jobs/${id}/stop`, { method: "POST", body: "{}" });
      await jobs.refresh();
    } finally {
      setBusy("");
    }
  }

  async function runAgent() {
    const prompt = agentPrompt.trim();
    if (!prompt) return;
    setAgentPrompt("");
    const assistantId = `assistant-${Date.now()}`;
    setAgentMessages((messages) => [
      ...messages,
      { role: "user", content: prompt },
      { id: assistantId, role: "assistant", content: "", events: [], streaming: true },
    ]);
    setBusy("Agent analyzing...");
    try {
      await streamAgentResponse({
        prompt,
        job_id: selectedJob?.id || null,
        provider: agentProvider,
        history: compactAgentHistory(agentMessages),
        onEvent: (event) => applyAgentEvent(assistantId, event),
      });
    } catch (err) {
      applyAgentEvent(assistantId, { type: "error", content: `Agent request failed: ${err.message || err}` });
    } finally {
      applyAgentEvent(assistantId, { type: "done" });
      setBusy("");
    }
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

  function newAgentChat() {
    const session = createAgentSession();
    setAgentSessions((sessions) => [session, ...sessions]);
    setActiveAgentSessionId(session.id);
    setAgentPrompt("");
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

  async function streamAgentResponse({ prompt, job_id, provider, history, onEvent }) {
    const response = await fetch(`${API_BASE}api/agent/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, job_id, provider, history }),
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
    if (activePage === "runtime") {
      return (
        <div className="tabGrid">
          <RuntimeDeck env={env.data} />
          <GpuDeck gpus={env.data?.gpus || []} />
        </div>
      );
    }
    if (activePage === "launcher") {
      return (
        <section className="panel launcher">
          <div className="panelHeader">
            <div>
              <h2>Task Launcher</h2>
              <p>Builds AReno CLI commands without importing training internals.</p>
            </div>
            <div className="tabs">
              <button className={classNames(launcherMode === "train" && "active")} onClick={() => setLauncherMode("train")}>Train</button>
              <button className={classNames(launcherMode === "serve" && "active")} onClick={() => setLauncherMode("serve")}>Serve</button>
            </div>
          </div>
          {launcherMode === "train" ? (
            <TrainForm config={trainConfig} setConfig={setTrainConfig} onStart={startTrain} />
          ) : (
            <ServeForm config={serveConfig} setConfig={setServeConfig} onStart={startServe} />
          )}
        </section>
      );
    }
    if (activePage === "agent") {
      return (
        <section className="panel chatPanel">
          <div className="panelHeader">
            <div>
              <h2>Agent Chat</h2>
              <p>{selectedJob ? `Context: ${selectedJob.name}` : "No job selected. The agent will use runtime context only."}</p>
            </div>
            <div className="agentHeaderActions">
              <button className="secondaryButton" onClick={newAgentChat}><Plus size={15} /> New Chat</button>
              <button className="secondaryButton" onClick={() => setAgentSettingsOpen(true)}><Settings2 size={15} /> Settings</button>
            </div>
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
                    {message.events?.length ? <AgentEventList events={message.events} /> : <MarkdownBlock text={message.content} />}
                    {message.streaming && <div className="streamingHint">thinking...</div>}
                  </div>
                ))}
              </div>
              <div className="chatComposer">
                <textarea
                  value={agentPrompt}
                  onChange={(event) => setAgentPrompt(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      runAgent();
                    }
                  }}
                />
                <button className="primaryButton" onClick={runAgent}><Send size={16} /> Send</button>
              </div>
            </>
          )}
          {agentSettingsOpen && (
            <Modal title="Agent Settings" onClose={() => setAgentSettingsOpen(false)}>
              <AgentProviderForm provider={agentProvider} setProvider={setAgentProvider} />
            </Modal>
          )}
        </section>
      );
    }
    if (!selectedJob) {
      return (
        <section className="panel jobListPage">
          <div className="panelHeader">
            <div>
              <h2>Jobs</h2>
              <p>Registered AReno train/serve processes. Select a row to open job details.</p>
            </div>
            <div className="pagerControls">
              <button className="secondaryButton" disabled={currentJobPage <= 1} onClick={() => setJobPage((page) => Math.max(1, page - 1))}>Prev</button>
              <span>{currentJobPage} / {jobPageCount}</span>
              <button className="secondaryButton" disabled={currentJobPage >= jobPageCount} onClick={() => setJobPage((page) => Math.min(jobPageCount, page + 1))}>Next</button>
            </div>
          </div>
          <div className="jobList">
            {jobList.length === 0 && <EmptyState title="No jobs yet" text="Start a train or serve task from the launcher." />}
            {pagedJobs.map((job) => (
              <button key={job.id} className="jobRow large" onClick={() => setSelectedJobId(job.id)}>
                <div className="jobIcon">{job.kind === "serve" ? <Server size={16} /> : <Layers size={16} />}</div>
                <div className="jobInfo">
                  <div className="jobTitle">{job.name}</div>
                  <div className="jobMeta">
                    {job.kind} · {job.status} · step {job.step ?? 0}
                  </div>
                </div>
                <span className={classNames("statusDot", job.status)} />
              </button>
            ))}
          </div>
          {jobList.length > jobPageSize && (
            <div className="listFooter">
              Showing {(currentJobPage - 1) * jobPageSize + 1}-{Math.min(currentJobPage * jobPageSize, jobList.length)} of {jobList.length} jobs
            </div>
          )}
        </section>
      );
    }
    return (
      <section className="panel detailPanel">
          <div className="panelHeader">
            <div>
              <h2>{selectedJob.name}</h2>
              <p>{selectedJob.kind} · {selectedJob.status}</p>
            </div>
            <div className="detailActions">
              <button className="secondaryButton" onClick={() => setSelectedJobId(null)}>Back</button>
              {selectedJob.status === "running" && (
                <button className="dangerButton" onClick={() => stopJob(selectedJob.id)}><CircleStop size={16} /> Stop</button>
              )}
            </div>
          </div>
          <Timeline job={selectedJob} />
          <JobMetricsView job={selectedJob} refreshNonce={refreshNonce} />
          <SampleView samples={selectedJob?.samples || []} />
          <div className="split">
            <ConfigView config={selectedJob?.config} launch={selectedJob?.launch} />
            <LogView logs={selectedJob?.logs || []} />
          </div>
      </section>
    );
  }

  const [pageTitle, pageDescription] = pageCopy[activePage] || pageCopy.jobs;

  return (
    <div className="shell">
      <aside className="rail">
        <div className="brand">
          <div className="brandMark">A</div>
          <div>
            <div className="brandName">AReno Ops</div>
            <div className="brandMeta">runtime workbench</div>
          </div>
        </div>
        <nav className="nav">
          {pages.map((page) => (
            <button key={page.id} className={classNames("navItem", activePage === page.id && "active")} onClick={() => setActivePage(page.id)}>
              {page.icon} {page.label}
            </button>
          ))}
        </nav>
        <div className="railFooter">
          <div className="tinyLabel">Repo</div>
          <div className="monoLine">{env.data?.repo?.branch || "unknown"} · {env.data?.repo?.commit || "no git"}</div>
        </div>
      </aside>

      <main className="main">
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
              className="iconButton"
              onClick={() => {
                env.refresh();
                jobs.refresh();
                jobDetail.refresh();
                setRefreshNonce((value) => value + 1);
              }}
              title="Refresh"
            >
              <RefreshCw size={16} />
            </button>
          </div>
        </header>

        {busy && <div className="notice">{busy}</div>}

        {renderPage()}
      </main>
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

function JobMetricsView({ job, refreshNonce }) {
  return (
    <div className="jobMetricsGrid">
      <div className="panel insetPanel">
        <MetricChart jobId={job?.id} metricsDir={job?.metrics_dir} refreshNonce={refreshNonce} />
      </div>
      <div className="panel insetPanel">
        <TimePerfView rows={job?.timeperf || []} />
      </div>
    </div>
  );
}

function AgentEventList({ events }) {
  return (
    <div className="agentEventList">
      {events.map((event, index) => {
        if (event.type === "reasoning") return <ReasoningBlock key={index} text={event.text} />;
        if (event.type === "content") return <MarkdownBlock key={index} text={event.text} />;
        if (event.type === "tool_call") return <ToolCallCard key={index} call={event.call} live={event.live} />;
        if (event.type === "tool_result") return <ToolResultCard key={index} result={event.result} />;
        return null;
      })}
    </div>
  );
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
  const algo = configValue(job, "algo");
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
      { id: "advantage", label: "advantage", aliases: ["advantage_start", "advantage_end"], roles: ["critic"] },
      { id: "critic_train", label: "critic train", aliases: ["train_start", "train_end"], roles: ["critic"] },
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
  if (job?.status && ["succeeded", "failed", "stopped", "exited"].includes(job.status)) return "done";
  const stage = String(job?.stage || "created");
  const steps = timelineSteps(job);
  const match = steps.find((item) => timelineItemMatches(item, stage, job));
  return match?.id || stage;
}

function timelineItemMatches(item, stage, job) {
  const stageMatches = item.id === stage || item.aliases?.includes(stage);
  if (!stageMatches) return false;
  if (!item.roles?.length) return true;
  return item.roles.includes(String(job?.role || ""));
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

function MetricChart({ jobId, metricsDir, refreshNonce }) {
  const [selectedName, setSelectedName] = useState("");
  const [smooth, setSmooth] = useState(0.6);
  const [metricList, setMetricList] = useState([]);
  const [points, setPoints] = useState([]);
  const [metricLoading, setMetricLoading] = useState(false);
  useEffect(() => {
    let cancelled = false;
    setMetricList([]);
    setPoints([]);
    setSelectedName("");
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
  }, [jobId, refreshNonce]);
  useEffect(() => {
    let cancelled = false;
    setPoints([]);
    if (!jobId || !selectedName) return undefined;
    setMetricLoading(true);
    api(`/api/jobs/${jobId}/metric?name=${encodeURIComponent(selectedName)}&limit=500`)
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
  }, [jobId, selectedName]);
  const names = metricList.map((item) => item.name).sort();
  const activeName = selectedName && names.includes(selectedName) ? selectedName : names[0] || "";
  const visiblePoints = points.slice(-240);
  const smoothed = smoothTensorboard(visiblePoints, smooth);
  const plot = buildMetricPlot(visiblePoints, smoothed);
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
        <svg className="metricPlot" viewBox="0 0 720 180" role="img">
          <g className="plotGrid">
            {[0, 1, 2, 3].map((item) => <line key={item} x1="0" x2="720" y1={30 + item * 42} y2={30 + item * 42} />)}
          </g>
          <polyline className="rawLine" points={plot.raw} />
          <polyline className="smoothLine" points={plot.smooth} />
          {visiblePoints.slice(-24).map((point, index) => (
            <circle key={`${point.step}-${index}`} cx={plot.coords[index + Math.max(0, visiblePoints.length - 24)]?.x || 0} cy={plot.coords[index + Math.max(0, visiblePoints.length - 24)]?.y || 0} r="2.2">
              <title>{`${activeName} step ${point.step}: ${point.value}`}</title>
            </circle>
          ))}
        </svg>
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

function buildMetricPlot(rawPoints, smoothPoints) {
  if (!rawPoints.length) return { raw: "", smooth: "", coords: [], minLabel: "n/a", maxLabel: "n/a" };
  const allValues = [...rawPoints, ...smoothPoints].map((point) => point.value);
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);
  const span = Math.max(max - min, 1e-9);
  const stepMin = rawPoints[0].step;
  const stepMax = rawPoints[rawPoints.length - 1].step;
  const stepSpan = Math.max(stepMax - stepMin, 1);
  const coord = (point) => ({
    x: ((point.step - stepMin) / stepSpan) * 700 + 10,
    y: 168 - ((point.value - min) / span) * 146,
  });
  const rawCoords = rawPoints.map(coord);
  const smoothCoords = smoothPoints.map(coord);
  return {
    raw: rawCoords.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" "),
    smooth: smoothCoords.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" "),
    coords: rawCoords,
    minLabel: compactNumber(min),
    maxLabel: compactNumber(max),
  };
}

function compactNumber(value) {
  if (!Number.isFinite(value)) return "n/a";
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) return value.toExponential(2);
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function TimePerfView({ rows }) {
  const visible = sampleTimePerfRows(rows || [], 7).reverse();
  const avgTotal = rows?.length ? rows.reduce((sum, row) => sum + Number(row.total_s || 0), 0) / rows.length : 0;
  const maxTotal = Math.max(1, ...visible.map((row) => Number(row.total_s || 0)));
  const segmentNames = Array.from(new Set((rows || []).flatMap((row) => (row.segments || []).map((segment) => segment.name)))).slice(0, 8);
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

function SampleView({ samples }) {
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
  return (
    <div className="sampleCard">
      <div className="codeTitle sampleTitle">
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
      </div>
      {sample ? (
        <div className="sampleGrid">
          <div>
            <span>Prompt</span>
            <p>{sample.prompt || "n/a"}</p>
          </div>
          <div>
            <span>Completion</span>
            <p>{sample.completion || "n/a"}</p>
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

function TrainForm({ config, setConfig, onStart }) {
  const algo = String(config.algo || "sft").toLowerCase();
  const sections = trainLauncherSections(algo);
  const updateField = (key, value) => setConfig({ ...config, [key]: value });
  return (
    <div className="launcherSections">
      {sections.map((section) => (
        <div className="launcherSection" key={section.title}>
          <div className="launcherSectionHeader">
            <strong>{section.title}</strong>
            {section.note && <span>{section.note}</span>}
          </div>
          <div className="formGrid">
            {section.fields.map((field) => (
              <Field
                key={field.key}
                label={field.label}
                value={config[field.key]}
                onChange={(value) => updateField(field.key, value)}
                compact={field.compact}
                type={field.type}
                options={field.options}
              />
            ))}
          </div>
        </div>
      ))}
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

createRoot(document.getElementById("root")).render(<App />);
