#!/usr/bin/env python3
"""Serve the llamafile / llama.cpp web UI and route its API calls to Ollama.

The UI in ./public is the unmodified prebuilt llama.cpp web UI (build b10441,
the one llamafile embeds). It expects a llama-server backend, so this server
emulates llama-server in "router" mode and translates each call to Ollama:

  GET  /props[?model=]           -> /api/show, /api/ps
  GET  /v1/models                -> /api/tags + /api/ps
  POST /v1/chat/completions      -> /api/chat (streamed back as OpenAI SSE)
  POST /models/load, /unload     -> /api/generate with keep_alive
  GET  /models/sse               -> model status events (polls /api/ps)

Standard library only. Usage: python3 server.py [--port 8080] [--ollama URL]
Run once with --install to keep it running across reboots.
"""

import argparse
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")
OLLAMA = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "0")) or None
DEFAULT_CTX = 4096


# --------------------------------------------------------------------------
# Ollama client helpers


def ollama(path, body=None, timeout=600, stream=False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        OLLAMA + path, data=data, method="POST" if data is not None else "GET",
        headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    if stream:
        return resp
    with resp:
        return json.loads(resp.read() or b"{}")


_show_cache = {}


def show(model):
    if model not in _show_cache:
        _show_cache[model] = ollama("/api/show", {"model": model})
    return _show_cache[model]


def running():
    """Map of loaded model name -> /api/ps entry."""
    try:
        return {m["name"]: m for m in ollama("/api/ps", timeout=5).get("models", [])}
    except Exception:
        return {}


def context_length(model, ps_entry=None):
    if ps_entry and ps_entry.get("context_length"):
        return ps_entry["context_length"]
    if NUM_CTX:
        return NUM_CTX
    try:
        for k, v in show(model).get("model_info", {}).items():
            if k.endswith(".context_length"):
                return min(v, DEFAULT_CTX)
    except Exception:
        pass
    return DEFAULT_CTX


def capabilities(model):
    try:
        return show(model).get("capabilities", []) or []
    except Exception:
        return []


# --------------------------------------------------------------------------
# Model status broadcasting for /models/sse


class StatusHub:
    def __init__(self):
        self.lock = threading.Lock()
        self.listeners = set()
        self.status = {}  # model -> status string
        self.pending = set()  # models with a load/unload in flight

    def subscribe(self):
        q = queue.Queue()
        with self.lock:
            self.listeners.add(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.listeners.discard(q)

    def set(self, model, status, **extra):
        with self.lock:
            if self.status.get(model) == status and not extra:
                return
            self.status[model] = status
            listeners = list(self.listeners)
        event = {"event": "model_status", "model": model,
                 "data": {"status": status, **extra}}
        for q in listeners:
            q.put(event)

    def poll_forever(self):
        """Pick up loads/evictions that happen outside the UI."""
        while True:
            time.sleep(3)
            loaded = running()
            with self.lock:
                known = dict(self.status)
                pending = set(self.pending)
            for model in set(known) | set(loaded):
                if model in pending:
                    continue
                want = "loaded" if model in loaded else "unloaded"
                if known.get(model) in ("loaded", "unloaded", None) and known.get(model) != want:
                    self.set(model, want)


hub = StatusHub()


def load_model(model, unload=False):
    with hub.lock:
        hub.pending.add(model)
    hub.set(model, "unloaded" if unload else "loading")
    try:
        ollama("/api/generate", {"model": model, "keep_alive": 0 if unload else "5m"})
        hub.set(model, "unloaded" if unload else "loaded")
    except Exception as e:
        hub.set(model, "failed", error=str(e), exit_code=1)
    finally:
        with hub.lock:
            hub.pending.discard(model)


# --------------------------------------------------------------------------
# Request / response translation


def to_ollama_messages(messages):
    out, call_names = [], {}
    for m in messages:
        msg = {"role": m.get("role", "user")}
        content = m.get("content")
        if isinstance(content, list):
            texts, images = [], []
            for part in content:
                t = part.get("type")
                if t == "text":
                    texts.append(part.get("text", ""))
                elif t == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    images.append(url.split(",", 1)[1] if url.startswith("data:") else url)
            msg["content"] = "\n".join(texts)
            if images:
                msg["images"] = images
        else:
            msg["content"] = content or ""
        if m.get("reasoning_content"):
            msg["thinking"] = m["reasoning_content"]
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = c.get("function", {})
                args = fn.get("arguments") or "{}"
                try:
                    args = json.loads(args) if isinstance(args, str) else args
                except ValueError:
                    args = {}
                call_names[c.get("id")] = fn.get("name")
                calls.append({"function": {"name": fn.get("name"), "arguments": args}})
            msg["tool_calls"] = calls
        if msg["role"] == "tool" and m.get("tool_call_id") in call_names:
            msg["tool_name"] = call_names[m["tool_call_id"]]
        out.append(msg)
    return out


# llama-server request field -> Ollama option name
OPTION_MAP = {
    "temperature": "temperature", "top_k": "top_k", "top_p": "top_p",
    "min_p": "min_p", "typ_p": "typical_p", "repeat_penalty": "repeat_penalty",
    "repeat_last_n": "repeat_last_n", "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty", "seed": "seed", "stop": "stop",
    "max_tokens": "num_predict",
}


def to_ollama_request(body):
    model = body.get("model")
    options = {v: body[k] for k, v in OPTION_MAP.items() if body.get(k) is not None}
    if NUM_CTX:
        options["num_ctx"] = NUM_CTX
    req = {"model": model, "messages": to_ollama_messages(body.get("messages", [])),
           "stream": bool(body.get("stream")), "options": options}
    if body.get("tools"):
        req["tools"] = body["tools"]
    kwargs = body.get("chat_template_kwargs") or {}
    if "thinking" in capabilities(model):
        req["think"] = kwargs.get("enable_thinking", True) is not False
    return req


def timings(prompt_n, prompt_ms, predicted_n, predicted_ms):
    return {
        "cache_n": 0,
        "prompt_n": prompt_n, "prompt_ms": prompt_ms,
        "prompt_per_second": prompt_n / prompt_ms * 1000 if prompt_ms else 0,
        "predicted_n": predicted_n, "predicted_ms": predicted_ms,
        "predicted_per_second": predicted_n / predicted_ms * 1000 if predicted_ms else 0,
    }


def final_timings(chunk):
    return timings(chunk.get("prompt_eval_count", 0), chunk.get("prompt_eval_duration", 0) / 1e6,
                   chunk.get("eval_count", 0), chunk.get("eval_duration", 0) / 1e6)


def openai_tool_calls(calls, start=0):
    return [{"index": start + i, "id": "call_" + uuid.uuid4().hex[:24], "type": "function",
             "function": {"name": c["function"]["name"],
                          "arguments": json.dumps(c["function"].get("arguments", {}))}}
            for i, c in enumerate(calls)]


def finish_reason(chunk, had_tools):
    if had_tools:
        return "tool_calls"
    return "length" if chunk.get("done_reason") == "length" else "stop"


# --------------------------------------------------------------------------
# HTTP handler


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def log_message(self, fmt, *args):
        if not self.path.startswith("/models/sse"):
            super().log_message(fmt, *args)

    # -- plumbing

    def send_json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def start_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def sse(self, obj):
        payload = obj if isinstance(obj, str) else json.dumps(obj)
        self.wfile.write(f"data: {payload}\n\n".encode())
        self.wfile.flush()

    def route(self):
        return urllib.parse.urlsplit(self.path).path.rstrip("/") or "/"

    def query(self):
        return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.path).query))

    # -- dispatch

    def do_GET(self):
        path = self.route()
        try:
            if path == "/health":
                return self.send_json({"status": "ok"})
            if path == "/props":
                return self.get_props()
            if path in ("/v1/models", "/models"):
                return self.get_models()
            if path == "/models/sse":
                return self.models_sse()
            if path == "/slots":
                return self.send_json([])
            if path in ("/tools", "/mcp-servers"):
                return self.send_json({"error": {"code": 404, "message": "Not supported"}}, 404)
        except urllib.error.URLError as e:
            return self.send_json({"error": {"code": 503, "message": f"Ollama unreachable: {e}"}}, 503)
        # Single-page app: unknown non-file paths fall back to index.html.
        if not os.path.exists(os.path.join(ROOT, path.lstrip("/"))):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        path = self.route()
        try:
            body = self.read_json()
            if path in ("/v1/chat/completions", "/chat/completions"):
                return self.chat(body)
            if path in ("/models/load", "/models/unload"):
                model = body.get("model")
                threading.Thread(target=load_model, args=(model, path.endswith("unload")),
                                 daemon=True).start()
                return self.send_json({"success": True})
            if path == "/v1/chat/completions/control":
                return self.send_json({"success": False, "error": "Not supported by Ollama"})
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            return self.send_json({"error": {"code": e.code, "message": msg}}, e.code)
        except urllib.error.URLError as e:
            return self.send_json({"error": {"code": 503, "message": f"Ollama unreachable: {e}"}}, 503)
        self.send_json({"error": {"code": 404, "message": "File Not Found"}}, 404)

    # -- endpoints

    def get_props(self):
        model = self.query().get("model")
        base = {
            "build_info": "ollama-proxy",
            "webui": True,
            "cors_proxy_enabled": False,
            "modalities": {"vision": False, "audio": False},
            "default_generation_settings": {"n_ctx": NUM_CTX or DEFAULT_CTX, "params": {}},
        }
        if not model:
            return self.send_json({**base, "role": "router"})
        info, ps = show(model), running().get(model)
        caps = info.get("capabilities", []) or []
        details = info.get("details", {})
        self.send_json({
            **base,
            "role": "model",
            "model_alias": model,
            "model_path": model,
            "chat_template": info.get("template", ""),
            "modalities": {"vision": "vision" in caps, "audio": "audio" in caps},
            "default_generation_settings": {
                "n_ctx": context_length(model, ps),
                "params": {},
            },
            "total_slots": 1,
            "details": details,
        })

    def get_models(self):
        tags = ollama("/api/tags").get("models", [])
        loaded = running()
        data, extra = [], []
        for m in tags:
            name = m["name"]
            status = hub.status.get(name) if name in hub.pending else None
            status = status or ("loaded" if name in loaded else "unloaded")
            hub.status.setdefault(name, status)
            caps = capabilities(name)
            inputs = ["text"] + (["image"] if "vision" in caps else []) + \
                     (["audio"] if "audio" in caps else [])
            data.append({
                "id": name, "object": "model", "owned_by": "ollama",
                "created": int(time.time()), "aliases": [],
                "status": {"value": status},
                "architecture": {"input_modalities": inputs},
                "meta": {"size": m.get("size"), **(m.get("details") or {})},
            })
            extra.append({"name": name, "model": name, "capabilities": caps,
                          "details": m.get("details")})
        self.send_json({"object": "list", "data": data, "models": extra})

    def models_sse(self):
        self.start_sse()
        q = hub.subscribe()
        try:
            while True:
                try:
                    self.sse(q.get(timeout=15))
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except OSError:
            pass
        finally:
            hub.unsubscribe(q)

    def chat(self, body):
        req = to_ollama_request(body)
        model = req["model"]
        cid = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        if not req["stream"]:
            return self.chat_blocking(req, cid, created)

        upstream = ollama("/api/chat", req, stream=True)
        hub.set(model, "loaded")
        self.start_sse()

        def chunk(delta, finish=None, **extra):
            return {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": model, "choices": [{"index": 0, "delta": delta,
                                                 "finish_reason": finish}], **extra}

        t0, first, n, tool_count = time.time(), None, 0, 0
        try:
            self.sse(chunk({"role": "assistant", "content": None}))
            for line in upstream:
                if not line.strip():
                    continue
                c = json.loads(line)
                if c.get("error"):
                    self.sse({"error": {"code": 500, "message": c["error"]}})
                    break
                msg = c.get("message") or {}
                delta = {}
                if msg.get("thinking"):
                    delta["reasoning_content"] = msg["thinking"]
                if msg.get("content"):
                    delta["content"] = msg["content"]
                if msg.get("tool_calls"):
                    delta["tool_calls"] = openai_tool_calls(msg["tool_calls"], tool_count)
                    tool_count += len(msg["tool_calls"])
                if delta:
                    now = time.time()
                    first = first or now
                    n += 1
                    live = timings(0, (first - t0) * 1000, n, (now - first) * 1000)
                    self.sse(chunk(delta, timings=live))
                if c.get("done"):
                    t = final_timings(c)
                    self.sse(chunk({}, finish_reason(c, tool_count), timings=t, usage={
                        "prompt_tokens": t["prompt_n"], "completion_tokens": t["predicted_n"],
                        "total_tokens": t["prompt_n"] + t["predicted_n"]}))
                    break
            self.sse("[DONE]")
        except ConnectionError:
            pass  # user pressed stop; closing upstream makes Ollama stop too
        finally:
            upstream.close()

    def chat_blocking(self, req, cid, created):
        c = ollama("/api/chat", req)
        msg = c.get("message") or {}
        out = {"role": "assistant", "content": msg.get("content", "")}
        if msg.get("thinking"):
            out["reasoning_content"] = msg["thinking"]
        if msg.get("tool_calls"):
            out["tool_calls"] = openai_tool_calls(msg["tool_calls"])
        t = final_timings(c)
        self.send_json({
            "id": cid, "object": "chat.completion", "created": created, "model": req["model"],
            "choices": [{"index": 0, "message": out,
                         "finish_reason": finish_reason(c, bool(msg.get("tool_calls")))}],
            "usage": {"prompt_tokens": t["prompt_n"], "completion_tokens": t["predicted_n"],
                      "total_tokens": t["prompt_n"] + t["predicted_n"]},
            "timings": t,
        })


# --------------------------------------------------------------------------
# Run-at-boot installation (systemd user service, launchd, Windows task)

SERVICE = "llamafile-ollama-ui"
WSL_TASK = "llamafile-ollama-ui (start WSL)"
WIN_TASK = "llamafile-ollama-ui"


def sh(*cmd, check=True):
    import subprocess
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{r.stderr.strip()}")
    return r.stdout.strip()


def is_wsl():
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def powershell(script):
    return sh("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script)


def install(args):
    import platform
    import shlex
    import sys
    exe = [sys.executable, os.path.abspath(__file__), "--host", args.host,
           "--port", str(args.port), "--ollama", args.ollama]
    env = {"OLLAMA_NUM_CTX": str(NUM_CTX)} if NUM_CTX else {}

    if platform.system() == "Darwin":
        plist = os.path.expanduser(f"~/Library/LaunchAgents/com.{SERVICE}.plist")
        items = "".join(f"<string>{a}</string>" for a in exe)
        envs = "".join(f"<key>{k}</key><string>{v}</string>" for k, v in env.items())
        os.makedirs(os.path.dirname(plist), exist_ok=True)
        with open(plist, "w") as f:
            f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>com.{SERVICE}</string>
<key>ProgramArguments</key><array>{items}</array>
<key>EnvironmentVariables</key><dict>{envs}</dict>
<key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
<key>StandardErrorPath</key><string>/tmp/{SERVICE}.log</string>
</dict></plist>
""")
        sh("launchctl", "unload", plist, check=False)
        sh("launchctl", "load", "-w", plist)
        print(f"Installed LaunchAgent {plist}")
    elif platform.system() == "Linux":
        unit = os.path.expanduser(f"~/.config/systemd/user/{SERVICE}.service")
        os.makedirs(os.path.dirname(unit), exist_ok=True)
        envs = "".join(f"Environment={k}={v}\n" for k, v in env.items())
        with open(unit, "w") as f:
            f.write(f"""[Unit]
Description=llamafile web UI routed to Ollama
# Pulls in a user-level ollama.service if there is one; harmless otherwise.
Wants=ollama.service
After=network.target ollama.service

[Service]
ExecStart={shlex.join(exe)}
{envs}Restart=always
RestartSec=3

[Install]
WantedBy=default.target
""")
        sh("systemctl", "--user", "daemon-reload")
        sh("systemctl", "--user", "enable", "--now", SERVICE)
        sh("systemctl", "--user", "restart", SERVICE)
        # Linger starts user services at boot, without anyone logging in.
        user = os.environ.get("USER") or sh("id", "-un")
        if "Linger=yes" not in sh("loginctl", "show-user", user, "-p", "Linger", check=False):
            sh("loginctl", "enable-linger", user)
        print(f"Installed systemd user service {unit}")
        if is_wsl():
            install_wsl_task()
    elif platform.system() == "Windows":
        install_windows(exe)
    else:
        raise SystemExit("--install supports Windows, Linux (systemd), WSL and macOS")
    print(f"Running now and on every boot: http://{args.host}:{args.port}/")


def ps_quote(s):
    return "'" + s.replace("'", "''") + "'"


def install_windows(exe):
    """Scheduled task running pythonw.exe (no console window).

    As administrator it starts at boot, before anyone logs in (S4U logon, no
    stored password). Otherwise it starts when this user logs in.
    """
    import subprocess
    import sys
    pyw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    if not os.path.exists(pyw):
        pyw = sys.executable
    arguments = subprocess.list2cmdline(exe[1:])
    envs = "".join(f"[Environment]::SetEnvironmentVariable({ps_quote(k)}, {ps_quote(v)}, 'User')\n"
                   for k, v in ({"OLLAMA_NUM_CTX": str(NUM_CTX)} if NUM_CTX else {}).items())
    out = powershell(f"""
{envs}$admin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole('Administrators')
$a = New-ScheduledTaskAction -Execute {ps_quote(pyw)} -Argument {ps_quote(arguments)} `
     -WorkingDirectory {ps_quote(os.path.dirname(os.path.abspath(__file__)))}
$s = New-ScheduledTaskSettingsSet -ExecutionTimeLimit 0 -AllowStartIfOnBatteries `
     -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew `
     -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
if ($admin) {{
  $t = New-ScheduledTaskTrigger -AtStartup
  $p = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\\$env:USERNAME" -LogonType S4U
}} else {{
  $t = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\\$env:USERNAME"
  $p = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\\$env:USERNAME" -LogonType Interactive
}}
Stop-ScheduledTask -TaskName {ps_quote(WIN_TASK)} -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName {ps_quote(WIN_TASK)} -Action $a -Trigger $t -Settings $s `
     -Principal $p -Description 'llamafile web UI routed to Ollama' -Force | Out-Null
Start-ScheduledTask -TaskName {ps_quote(WIN_TASK)}
if ($admin) {{ 'boot' }} else {{ 'logon' }}
""")
    when = "Windows starts" if out.endswith("boot") else \
        "you log in (run as administrator to start at boot instead)"
    print(f"Installed scheduled task '{WIN_TASK}'; it starts every time {when}")
    print(f"Log: {log_path()}")


def log_path():
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, SERVICE, "server.log")


def install_wsl_task():
    """WSL only boots when something launches it, so start it at Windows logon.

    The task runs `sleep infinity` in the distro through a headless console:
    no window, and the open session stops WSL shutting the distro down idle.
    """
    distro = os.environ.get("WSL_DISTRO_NAME") or "Ubuntu"
    powershell(f"""
$a = New-ScheduledTaskAction -Execute 'conhost.exe' `
     -Argument '--headless wsl.exe -d "{distro}" --exec sleep infinity'
$t = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$s = New-ScheduledTaskSettingsSet -ExecutionTimeLimit 0 -AllowStartIfOnBatteries `
     -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName '{WSL_TASK}' -Action $a -Trigger $t -Settings $s `
     -Description 'Starts WSL ({distro}) at logon so {SERVICE} runs' -Force | Out-Null
Start-ScheduledTask -TaskName '{WSL_TASK}'
""")
    print(f"Installed Windows logon task '{WSL_TASK}' (keeps WSL {distro} running)")


def uninstall():
    import platform
    if platform.system() == "Windows":
        powershell(f"Stop-ScheduledTask -TaskName {ps_quote(WIN_TASK)} -ErrorAction SilentlyContinue; "
                   f"Unregister-ScheduledTask -TaskName {ps_quote(WIN_TASK)} -Confirm:$false "
                   "-ErrorAction SilentlyContinue")
    elif platform.system() == "Darwin":
        plist = os.path.expanduser(f"~/Library/LaunchAgents/com.{SERVICE}.plist")
        sh("launchctl", "unload", "-w", plist, check=False)
        if os.path.exists(plist):
            os.remove(plist)
    else:
        sh("systemctl", "--user", "disable", "--now", SERVICE, check=False)
        unit = os.path.expanduser(f"~/.config/systemd/user/{SERVICE}.service")
        if os.path.exists(unit):
            os.remove(unit)
        sh("systemctl", "--user", "daemon-reload", check=False)
        if is_wsl():
            # Leaves the running WSL session alone; it just won't start at logon.
            powershell(f"Unregister-ScheduledTask -TaskName '{WSL_TASK}' -Confirm:$false "
                       "-ErrorAction SilentlyContinue")
    print(f"Uninstalled {SERVICE}; it will no longer start at boot")


def status():
    import platform
    if platform.system() == "Windows":
        print(powershell(f"$t = Get-ScheduledTask -TaskName {ps_quote(WIN_TASK)} -ErrorAction "
                         "SilentlyContinue; if ($t) { 'Scheduled task: ' + $t.State } "
                         "else { 'not installed' }"))
        print(f"Log: {log_path()}")
        return
    if platform.system() == "Darwin":
        print(sh("launchctl", "list", f"com.{SERVICE}", check=False) or "not installed")
        return
    print(sh("systemctl", "--user", "status", SERVICE, "--no-pager", "-n", "5", check=False)
          or "not installed")
    if is_wsl():
        print(powershell(f"(Get-ScheduledTask -TaskName '{WSL_TASK}' -ErrorAction "
                         "SilentlyContinue).State") or "Windows logon task: not installed")


def main():
    global OLLAMA
    import sys
    if sys.stdout is None or sys.stderr is None:  # pythonw.exe has no console
        os.makedirs(os.path.dirname(log_path()), exist_ok=True)
        sys.stdout = sys.stderr = open(log_path(), "a", buffering=1, encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--ollama", default=OLLAMA, help="Ollama base URL (default %(default)s)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--install", action="store_true",
                      help="run now and on every boot (with these --host/--port/--ollama)")
    mode.add_argument("--uninstall", action="store_true", help="stop and remove from boot")
    mode.add_argument("--status", action="store_true", help="show the boot service status")
    args = p.parse_args()
    OLLAMA = args.ollama.rstrip("/")
    args.ollama = OLLAMA
    if args.install:
        return install(args)
    if args.uninstall:
        return uninstall()
    if args.status:
        return status()
    threading.Thread(target=hub.poll_forever, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"llamafile UI -> Ollama at {OLLAMA}\nOpen http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
