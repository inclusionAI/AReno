"""Training dashboard analyzer — parses AReno training logs and renders charts.

Usage:
    # Local analysis of a log file
    python examples/agentic/logic_diagnosis/dashboard.py /path/to/training.log

    # Or pipe stdout directly
    areno train ... 2>&1 | tee /tmp/train.log
    python examples/agentic/logic_diagnosis/dashboard.py /tmp/train.log

    # Live server mode — watches a log file and serves charts at http://127.0.0.1:8769
    python examples/agentic/logic_diagnosis/dashboard.py --serve /tmp/train.log
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


def parse_log(path: str) -> dict:
    """Parse an AReno training log file into structured metrics."""
    metrics: dict[str, list] = {
        "step": [],
        "reward_mean": [],
        "rollout_logprob_mean": [],
        "response_len": [],
        "loss": [],
        "grad_norm": [],
        "tool_calls": [],
        "tool_results": [],
        "messages": [],
        "tokens": [],
        "rollout_time_s": [],
        "train_time_s": [],
    }

    # step-level metrics from train_stats
    _STAT_RE = re.compile(
        r"step=(\d+).*?"
        r"reward_mean value=([-\d.]+).*?"
        r"rollout_logprob_mean value=([-\d.]+).*?"
        r"response_len[= ]+([\d.]+).*?"
        r"'loss':\s*([\de.\-]+).*?"
        r"'grad_norm':\s*([\de.\-]+)"
    )

    # batch-level from "agentic train batch built"
    _BATCH_RE = re.compile(
        r"agentic train batch built.*?"
        r"tokens=(\d+).*?"
        r"messages=(\d+).*?"
        r"tool_calls=(\d+).*?"
        r"tool_results=(\d+)"
    )

    # timing
    _TIME_RE = re.compile(
        r"time rollout=([\d.]+).*?train=([\d.]+)"
    )

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    steps_seen = set()
    for m in _STAT_RE.finditer(content):
        step = int(m.group(1))
        if step in steps_seen:
            continue
        steps_seen.add(step)
        metrics["step"].append(step)
        metrics["reward_mean"].append(float(m.group(2)))
        metrics["rollout_logprob_mean"].append(float(m.group(3)))
        metrics["response_len"].append(float(m.group(4)))
        metrics["loss"].append(float(m.group(5)))
        metrics["grad_norm"].append(float(m.group(6)))

    # Batch-level data (approximate mapping to nearest step)
    batches = list(_BATCH_RE.finditer(content))
    times = list(_TIME_RE.finditer(content))
    for i, step in enumerate(metrics["step"]):
        # Use batch data from the same step region
        if i < len(batches):
            metrics["tokens"].append(int(batches[i].group(1)))
            metrics["messages"].append(int(batches[i].group(2)))
            metrics["tool_calls"].append(int(batches[i].group(3)))
            metrics["tool_results"].append(int(batches[i].group(4)))
        else:
            for k in ("tokens", "messages", "tool_calls", "tool_results"):
                metrics[k].append(0)
        if i < len(times):
            metrics["rollout_time_s"].append(float(times[i].group(1)))
            metrics["train_time_s"].append(float(times[i].group(2)))
        else:
            metrics["rollout_time_s"].append(0)
            metrics["train_time_s"].append(0)

    return metrics


def metrics_to_json(metrics: dict) -> str:
    """Serialize to JSON for the frontend."""
    return json.dumps(metrics, separators=(",", ":"))


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AReno Training Dashboard</title>
<style>
:root{font-family:Inter,ui-monospace,system-ui,sans-serif;color:#eee;background:#1a1a2e}
body{margin:0;padding:20px;min-height:100vh}
h1{font-size:22px;margin:0 0 16px;color:#ffd166}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.chart{background:#16213e;border:2px solid #333;border-radius:12px;padding:14px}
.chart h2{font-size:14px;margin:0 0 8px;color:#aaa}
.chart.full{grid-column:1/-1}
svg text{fill:#ccc;font-size:10px;font-family:Inter,ui-monospace,monospace}
svg line{stroke:#333;stroke-width:0.5}
.stats{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px}
.stat{background:#16213e;border:2px solid #333;border-radius:10px;padding:10px 16px;text-align:center}
.stat .val{font-size:24px;font-weight:900}
.stat .lbl{font-size:11px;color:#888;margin-top:2px}
.good{color:#9be564}.bad{color:#e07070}.warn{color:#ffd166}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<h1>AReno Training Dashboard</h1>
<div class="stats" id="stats"></div>
<div class="grid" id="charts"></div>
<script>
function render(data){
  if(!data.step.length) return;
  const last = data.step.length - 1;

  // Summary stats
  const reward = data.reward_mean;
  const recentR = reward.slice(Math.max(0,last-10));
  const avgR = recentR.reduce((a,b)=>a+b,0)/recentR.length;
  const maxR = Math.max(...reward);
  const tools = data.tool_calls[last] || 0;
  const timeTotal = data.rollout_time_s.reduce((a,b)=>a+b,0) + data.train_time_s.reduce((a,b)=>a+b,0);
  document.getElementById('stats').innerHTML = `
    <div class="stat"><div class="val ${avgR>0?'good':'bad'}">${avgR.toFixed(3)}</div><div class="lbl">reward (recent 10)</div></div>
    <div class="stat"><div class="val ${maxR>0?'good':'warn'}">${maxR.toFixed(3)}</div><div class="lbl">max reward</div></div>
    <div class="stat"><div class="val">${data.step[last]}</div><div class="lbl">steps</div></div>
    <div class="stat"><div class="val">${tools}</div><div class="lbl">tool_calls (last)</div></div>
    <div class="stat"><div class="val">${(timeTotal/60).toFixed(1)}m</div><div class="lbl">total time</div></div>
  `;

  // Charts
  const charts = [
    {title:'Reward Mean', key:'reward_mean', color:'#9be564', threshold:0},
    {title:'Tool Calls per Step', key:'tool_calls', color:'#7db8d8'},
    {title:'Response Length (tokens)', key:'response_len', color:'#ffd166'},
    {title:'Policy Loss', key:'loss', color:'#e07070'},
    {title:'Gradient Norm', key:'grad_norm', color:'#e8a0d0'},
    {title:'Rollout Logprob Mean', key:'rollout_logprob_mean', color:'#ffb347'},
    {title:'Tokens per Step', key:'tokens', color:'#aaa'},
    {title:'Step Time (s)', key:'step_time', color:'#aaa'},
  ];

  let html = '';
  for(const c of charts){
    const full = c.key === 'reward_mean' ? ' full' : '';
    html += `<div class="chart${full}"><h2>${c.title}</h2>${lineChart(data,c)}</div>`;
  }
  document.getElementById('charts').innerHTML = html;
}

function lineChart(data, cfg){
  const w=540, h=180, pad={t:10,r:10,b:25,l:40};
  let vals = cfg.key === 'step_time'
    ? data.rollout_time_s.map((r,i)=>r+(data.train_time_s[i]||0))
    : (data[cfg.key] || []);
  if(!vals.length) return '<svg width="'+w+'" height="'+h+'"><text x="20" y="30">no data</text></svg>';

  const min=Math.min(...vals), max=Math.max(...vals);
  const range = max-min || 1;
  const n=vals.length;
  const xs = vals.map((_,i)=>pad.l + i*(w-pad.l-pad.r)/Math.max(n-1,1));
  const ys = vals.map(v=>pad.t + (h-pad.t-pad.b)*(1-(v-min)/range));

  // Y-axis ticks
  let ticks = '';
  for(let i=0;i<=4;i++){
    const v = min + range*i/4;
    const y = pad.t + (h-pad.t-pad.b)*(1-i/4);
    ticks += `<text x="${pad.l-4}" y="${y+3}" text-anchor="end">${v.toFixed(2)}</text>`;
    ticks += `<line x1="${pad.l}" y1="${y}" x2="${w-pad.r}" y2="${y}" stroke="#222"/>`;
  }

  // Line
  let path = '';
  for(let i=0;i<n;i++) path += `${i?'L':'M'}${xs[i].toFixed(1)},${ys[i].toFixed(1)} `;

  // Threshold line
  let thresh = '';
  if(cfg.threshold !== undefined){
    const ty = pad.t + (h-pad.t-pad.b)*(1-(cfg.threshold-min)/range);
    thresh = `<line x1="${pad.l}" y1="${ty}" x2="${w-pad.r}" y2="${ty}" stroke="#fff3" stroke-dasharray="4,4"/>`;
  }

  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
    ${ticks}${thresh}
    <path d="${path}" fill="none" stroke="${cfg.color}" stroke-width="2"/>
  </svg>`;
}

fetch('api/data').then(r=>r.json()).then(render);
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    metrics: dict = {}

    def do_GET(self):
        route = urlparse(self.path).path.rstrip("/") or "/"
        if route in ("/", "/index.html"):
            self._html(INDEX_HTML)
        elif route == "/api/data":
            self._json(self.metrics)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _html(self, html):
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj):
        data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass


def serve(log_path: str, host: str = "127.0.0.1", port: int = 8769):
    """Watch a log file and serve charts via HTTP."""
    handler = DashboardHandler
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Dashboard at http://{host}:{port}")
    print(f"Watching {log_path}")

    try:
        while True:
            if os.path.exists(log_path):
                handler.metrics = parse_log(log_path)
            time.sleep(5)
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(description="AReno training dashboard analyzer")
    parser.add_argument("logfile", help="Path to training log file")
    parser.add_argument("--serve", action="store_true", help="Start live HTTP server")
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()

    if args.serve:
        serve(args.logfile, port=args.port)
    else:
        metrics = parse_log(args.logfile)
        print(json.dumps(metrics["step"], separators=(",", ":")))
        for i, step in enumerate(metrics["step"]):
            print(
                f"step={step:>4}  "
                f"reward={metrics['reward_mean'][i]:>8.4f}  "
                f"loss={metrics['loss'][i]:>10.6f}  "
                f"grad_norm={metrics['grad_norm'][i]:>8.2f}  "
                f"resp_len={metrics['response_len'][i]:>6.0f}  "
                f"tools={metrics['tool_calls'][i]:>3}  "
                f"tokens={metrics['tokens'][i]:>5}"
            )


if __name__ == "__main__":
    main()