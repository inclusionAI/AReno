// Dataset-independent Chromium capture worker. Node 22+ provides WebSocket.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import http from 'node:http';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { setTimeout as delay } from 'node:timers/promises';

const spec = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
const root = await fs.realpath(spec.workspace);
await fs.mkdir(spec.output, { recursive: true });
const report = { views: {}, interactions: [], errors: [], resource_failures: [] };
const mime = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.svg': 'image/svg+xml', '.png': 'image/png' };
const server = http.createServer(async (req, res) => {
  try {
    const file = await fs.realpath(path.join(root, decodeURIComponent(new URL(req.url, 'http://localhost').pathname)));
    if (!file.startsWith(root + path.sep)) throw new Error('outside workspace');
    res.writeHead(200, { 'Content-Type': mime[path.extname(file)] || 'application/octet-stream' });
    res.end(await fs.readFile(file));
  } catch {
    if (req.url === '/favicon.ico') { res.writeHead(204); res.end(); }
    else { res.writeHead(404); res.end('Not found'); }
  }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
let executable = spec.chromium;
if (!executable) {
  for (const dir of await fs.readdir('/ms-playwright')) {
    if (!dir.startsWith('chromium-')) continue;
    for (const suffix of ['chrome-linux/chrome', 'chrome-linux64/chrome', 'chrome-linux-arm64/chrome']) {
      const candidate = path.join('/ms-playwright', dir, suffix);
      try { await fs.access(candidate); executable = candidate; break; } catch {}
    }
    if (executable) break;
  }
}
if (!executable) throw new Error('configure a Chromium executable or use a preinstalled browser image');
const profile = await fs.mkdtemp(path.join(os.tmpdir(), 'areno-browser-'));
const chrome = spawn(executable, ['--headless', '--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu',
  '--disable-background-networking', '--no-first-run', '--no-default-browser-check',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'], { stdio: 'ignore' });
const watchdog = setTimeout(() => { chrome.kill('SIGKILL'); process.exit(2); }, spec.timeout_ms || 60000);
let ws;
try {
  let port;
  for (let i = 0; i < 150; i++) {
    try { port = (await fs.readFile(path.join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]; break; }
    catch { if (chrome.exitCode !== null) throw new Error('Chromium exited during startup'); await delay(100); }
  }
  if (!port) throw new Error('Chromium startup timed out');
  const pages = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  ws = new WebSocket(pages.find(p => p.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  let nextId = 0;
  const pending = new Map();
  ws.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.id) {
      const callback = pending.get(message.id);
      pending.delete(message.id);
      if (callback) message.error ? callback.reject(new Error(message.error.message)) : callback.resolve(message.result);
    } else if (message.method === 'Runtime.exceptionThrown' && report.errors.length < 30) {
      report.errors.push(message.params.exceptionDetails.text);
    } else if (message.method === 'Page.javascriptDialogOpening') {
      call('Page.handleJavaScriptDialog', { accept: false }).catch(() => {});
    } else if (message.method === 'Runtime.consoleAPICalled' && message.params.type === 'error' && report.errors.length < 30) {
      report.errors.push(message.params.args.map(a=>a.value || a.description || '').join(' ').slice(0,500));
    } else if (message.method === 'Network.loadingFailed' && report.resource_failures.length < 30) {
      report.resource_failures.push(message.params.errorText);
    } else if (message.method === 'Network.responseReceived' && message.params.response.status >= 400 && report.resource_failures.length < 30) {
      report.resource_failures.push(`${message.params.response.status}: ${message.params.response.url}`);
    }
  };
  function call(method, params = {}) {
    const id = ++nextId;
    return new Promise((resolve, reject) => { pending.set(id, { resolve, reject }); ws.send(JSON.stringify({ id, method, params })); });
  }
  async function evaluate(expression) {
    const result = await call('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
    return result.result.value;
  }
  async function snapshot(name, viewport) {
    const options = { format: 'png', captureBeyondViewport: Boolean(viewport) };
    if (viewport) {
      const metrics = await call('Page.getLayoutMetrics');
      const visible = metrics.cssVisualViewport;
      options.clip = { x: 0, y: 0, width: visible.clientWidth,
        height: Math.min(Math.max(visible.clientHeight, metrics.cssContentSize.height), 6000),
        scale: viewport.width / visible.clientWidth };
    }
    const image = await call('Page.captureScreenshot', options);
    await fs.writeFile(path.join(spec.output, `${name}.png`), Buffer.from(image.data, 'base64'));
    return `${name}.png`;
  }
  await call('Page.enable');
  await call('Runtime.enable');
  await call('Network.enable');
  await call('Fetch.enable', { patterns: [{ urlPattern: '*' }] });
  // Keep task rendering offline and prevent page scripts from probing services.
  const original = ws.onmessage;
  ws.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.method === 'Fetch.requestPaused') {
      const url = message.params.request.url;
      const allowed = url.startsWith(origin + '/') || url.startsWith('data:') || url.startsWith('blob:');
      call(allowed ? 'Fetch.continueRequest' : 'Fetch.failRequest',
        allowed ? { requestId: message.params.requestId } : { requestId: message.params.requestId, errorReason: 'BlockedByClient' }).catch(() => {});
    } else original(event);
  };
  for (const [name, viewport] of Object.entries(spec.viewports)) {
    if (!/^[a-zA-Z0-9_-]+$/.test(name)) throw new Error('invalid viewport name');
    await call('Emulation.setDeviceMetricsOverride', { ...viewport, deviceScaleFactor: 1, mobile: Boolean(viewport.mobile) });
    const navigation = await call('Page.navigate', { url: `${origin}/${spec.entrypoint}` });
    if (navigation.errorText) throw new Error(navigation.errorText);
    await evaluate(`new Promise(resolve => { const ready = () => document.fonts.ready.then(resolve); if (document.readyState === 'complete') ready(); else addEventListener('load', ready, {once:true}); })`);
    await evaluate(`(() => { const style=document.createElement('style'); style.textContent='*,*::before,*::after{animation:none!important;transition:none!important;caret-color:transparent!important}'; document.head.append(style); })()`);
    await delay(spec.settle_ms || 250);
    const observation = await evaluate(`(() => ({
      title: document.title, text: document.body.innerText.slice(0,16000),
      document_height: document.documentElement.scrollHeight, inner_width: innerWidth, screenshot_height_limit: 6000,
      visible_content: !!document.body.innerText.trim() || [...document.querySelectorAll('img,svg,canvas,video')].some(e=>e.getBoundingClientRect().width>0 && e.getBoundingClientRect().height>0),
      headings: [...document.querySelectorAll('h1,h2,h3')].map(e=>e.innerText).slice(0,40),
      horizontal_overflow: document.documentElement.scrollWidth > innerWidth + 1,
      buttons: [...document.querySelectorAll('button')].map(e=>e.innerText || e.getAttribute('aria-label') || '').slice(0,40),
      unlabeled_inputs: [...document.querySelectorAll('input,textarea,select')].filter(e=>!e.labels?.length && !e.getAttribute('aria-label') && !e.getAttribute('aria-labelledby')).length,
      broken_images: [...document.images].filter(e=>!e.complete || e.naturalWidth===0).length
    }))()`);
    report.views[name] = { ...observation, viewport, screenshot: await snapshot(name, viewport) };
    for (const [index, action] of (spec.interactions || []).entries()) {
      if (action.viewport && action.viewport !== name) continue;
      const outcome = await evaluate(`(() => {
        const action=${JSON.stringify(action)}, element=document.querySelector(action.selector);
        if (!element) return {passed:false, error:'action selector missing'};
        if (action.action==='click') element.click();
        else if (action.action==='fill') { element.value=action.value; element.dispatchEvent(new Event('input',{bubbles:true})); element.dispatchEvent(new Event('change',{bubbles:true})); }
        else return {passed:false,error:'unsupported action'};
        return {executed:true};
      })()`);
      await delay(150);
      if (outcome.executed) {
        outcome.passed = await evaluate(`(() => {
          const expected=${JSON.stringify(action.expect || null)};
          if (!expected) return false;
          const element=document.querySelector(expected.selector);
          if (!element) return false;
          const visible=!!(element.offsetWidth || element.offsetHeight || element.getClientRects().length);
          return (expected.visible===undefined || visible===expected.visible)
            && (expected.text_contains===undefined || element.textContent.includes(expected.text_contains))
            && (expected.value===undefined || element.value===expected.value);
        })()`);
      }
      report.interactions.push({ viewport: name, action, ...outcome, screenshot: await snapshot(`${name}-action-${index}`) });
    }
  }
  await fs.writeFile(path.join(spec.output, 'report.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally {
  clearTimeout(watchdog);
  ws?.close();
  const closed = chrome.exitCode !== null || chrome.signalCode !== null;
  const exited = closed ? null : once(chrome, 'close');
  chrome.kill('SIGKILL');
  if (exited) await exited;
  server.close();
  await fs.rm(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
}
