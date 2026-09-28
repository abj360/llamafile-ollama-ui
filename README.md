# llamafile web UI → Ollama

The same chat frontend that llamafile ships (the llama.cpp server web UI,
build b10441, the one llamafile embeds), served by a small Python proxy that
routes it to the models in your Ollama server.

    python3 server.py                                     # http://127.0.0.1:8080
    python server.py                                      # Windows PowerShell
    python3 server.py --port 9000 --ollama http://gpu-box:11434
    OLLAMA_NUM_CTX=16384 python3 server.py                # pass num_ctx to Ollama

## Run detached (in the background, until reboot)

Windows PowerShell:

    Start-Process pythonw -ArgumentList 'server.py' -WorkingDirectory $PWD -WindowStyle Hidden
    # log: %LOCALAPPDATA%\llamafile-ollama-ui\server.log
    # stop:
    Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" | Where-Object CommandLine -match 'server.py' | ForEach-Object { Stop-Process -Id $_.ProcessId }

Linux / macOS / WSL:

    nohup python3 server.py > server.log 2>&1 &
    # stop:
    pkill -f 'python3 server.py'

Add any flags after `server.py`, e.g. `'server.py --host 0.0.0.0 --port 9000'`.
It keeps running after you close the terminal, but not after a reboot. For that,
use `--install` below.

## Run permanently (survives reboots)

Run this once on the machine that should host the UI:

    python3 server.py --install                           # same flags as above
    python3 server.py --install --host 0.0.0.0 --port 9000 --ollama http://gpu-box:11434

It starts right away, restarts if it crashes, and starts again on every boot,
with no startup script to run. Use `--status` to check it and `--uninstall` to
remove it.

| Platform | What `--install` sets up |
|----------|--------------------------|
| Windows  | Scheduled task `llamafile-ollama-ui` running `pythonw.exe` (no console window), restarted if it crashes. From an **administrator** PowerShell it starts at boot, before anyone logs in; from a normal PowerShell it starts when you log in. Output goes to `%LOCALAPPDATA%\llamafile-ollama-ui\server.log` |
| Linux    | systemd user service `~/.config/systemd/user/llamafile-ollama-ui.service`, with lingering enabled so it starts at boot without a login. It also wants `ollama.service` if a user-level one exists |
| WSL 2    | The Linux service above (needs `systemd=true` in `/etc/wsl.conf`), plus a Windows logon task that starts the distro hidden and keeps it running. Otherwise WSL stays off after a Windows restart until you open a terminal |
| macOS    | LaunchAgent `~/Library/LaunchAgents/com.llamafile-ollama-ui.plist` (RunAtLoad + KeepAlive) |

On Windows, run it in PowerShell with the Python from python.org or the
Microsoft Store:

    python server.py --install                  # admin PowerShell: starts at boot

To reach the UI from other devices, add `--host 0.0.0.0` and allow Python
through Windows Firewall when prompted. Ollama for Windows starts when you log
in. If the UI starts first at boot, it shows "Ollama unreachable" until Ollama
is up, then works without a restart.

The service runs the `server.py` in its current folder, so install from where
you want to keep the code. Ollama must also be set to start at boot, which the
standard Ollama installers already do.

No dependencies beyond Python 3.8+. `public/` is the unmodified prebuilt UI.
The proxy emulates llama-server's "router" mode, so the model picker lists your
Ollama models and can load and unload them.

| UI calls                 | Proxy uses on Ollama                                 |
|--------------------------|------------------------------------------------------|
| `/v1/models`             | `/api/tags` + `/api/ps` (loaded/unloaded status)     |
| `/props?model=`          | `/api/show` (vision, template, context length)       |
| `/v1/chat/completions`   | `/api/chat` (streaming, thinking, tools, images)     |
| `/models/load`, `unload` | `/api/generate` with `keep_alive`                    |
| `/models/sse`            | status events, also polls `/api/ps` every 3s         |

**Thinking on/off:** for models Ollama reports as thinking-capable (qwen3,
deepseek-r1, gpt-oss, …) the **+** menu in the message box shows **Reasoning**:
Default / Off / Low / Medium / High / Max. Off sends `think: false`, and the
other levels turn thinking on. gpt-oss gets the matching low/medium/high level;
other models have no token budgets in Ollama, so the levels just mean on. The
menu appears once the model is loaded (after your first message to it).

Samplers mapped to Ollama options: temperature, top_k, top_p, min_p, typ_p,
repeat_penalty, repeat_last_n, presence/frequency penalty, seed, stop, max_tokens.
Not supported by Ollama, so ignored: DRY, XTC, sampler order, the
"stop reasoning" control, server-side MCP tools and the CORS proxy.

## Troubleshooting "Failed to connect to server"

That message means the browser lost its connection to `server.py` itself.
Ollama errors show their own text instead.

- **Is it still running?** Run `python server.py --status`, or check the
  terminal or log. The server now refuses to start a second copy on a port
  that's already in use, and tells you so.
- **Windows console freeze:** clicking inside the PowerShell window where
  `python server.py` runs used to pause the server (QuickEdit mode). It now
  turns that off at startup. Running it detached or with `--install` avoids
  the console entirely.
- **Wrong address:** use `--host 0.0.0.0` if you open the UI from another
  device, and check that `--ollama` points to where Ollama actually runs.

## Licenses
- `public/`: llama.cpp web UI, © The ggml authors, MIT (`LICENSE.llama.cpp`)
- llamafile, © Mozilla, Apache-2.0 (`LICENSE.llamafile`)
