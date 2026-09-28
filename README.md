# llamafile web UI → Ollama

The same chat frontend that llamafile ships (the llama.cpp server web UI,
build b10441, the one llamafile embeds), served by a small Python proxy that
routes it to the models in your Ollama server.

    python3 server.py                                     # http://127.0.0.1:8080
    python3 server.py --port 9000 --ollama http://gpu-box:11434
    OLLAMA_NUM_CTX=16384 python3 server.py                # pass num_ctx to Ollama

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

Samplers mapped to Ollama options: temperature, top_k, top_p, min_p, typ_p,
repeat_penalty, repeat_last_n, presence/frequency penalty, seed, stop, max_tokens.
Not supported by Ollama, so ignored: DRY, XTC, sampler order, the
"stop reasoning" control, server-side MCP tools and the CORS proxy.

## Licenses
- `public/`: llama.cpp web UI, © The ggml authors, MIT (`LICENSE.llama.cpp`)
- llamafile, © Mozilla, Apache-2.0 (`LICENSE.llamafile`)
