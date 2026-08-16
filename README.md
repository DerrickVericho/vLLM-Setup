<h1 align="center">⚡ vLLM on a Local GPU</h1>

<p align="center">
  <em>An OpenAI-compatible LLM server on your own hardware, in one <code>docker compose up</code>.</em>
</p>

<p align="center">
  <img alt="vLLM" src="https://img.shields.io/badge/vLLM-v0.27.1-30A3DC?style=flat-square">
  <img alt="Open WebUI" src="https://img.shields.io/badge/Open%20WebUI-v0.11.0-000000?style=flat-square">
  <img alt="Docker Compose" src="https://img.shields.io/badge/Docker%20Compose-2496ED?style=flat-square&logo=docker&logoColor=white">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-NVIDIA-76B900?style=flat-square&logo=nvidia&logoColor=white">
  <img alt="OpenAI-compatible API" src="https://img.shields.io/badge/API-OpenAI--compatible-412991?style=flat-square&logo=openai&logoColor=white">
</p>

---

Point any OpenAI client at `http://127.0.0.1:8000/v1` and it works — no code changes, no vendor SDK.
Open WebUI is included as an optional chat UI, but vLLM does not depend on it.

```
  your code ──► http://127.0.0.1:8000/v1 ──► [ vllm ] ──► GPU
                                                │
  browser ────► http://127.0.0.1:3000 ──► [ open-webui ]   (optional)
```

## 📦 Requirements

- NVIDIA GPU with a current driver. VRAM must fit the model weights plus the KV cache.
- Docker Desktop (or Docker Engine) with GPU support enabled.
- ~30 GB of free disk for the vLLM image, plus room for model weights.

## 🚀 Setup

### 1. Download the model first

The container can download the model itself, but then the first `up` looks like a 10–20 minute hang with
no visible progress. Pull it on the host instead — it is a one-time, resumable, visible step:

```bash
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3-4B-AWQ
```

This writes to your HuggingFace cache (`~/.cache/huggingface` by default). That directory is bind-mounted
into the container, so the same weights are reused by the container *and* by native Python on the host —
one copy, not two. Gated repos (e.g. `meta-llama/*`) need `hf auth login` first, and `HF_TOKEN` set in
`.env`.

### 2. Configure

```bash
cp .env.example .env
```

Fill in two values:

- `VLLM_API_KEY` — generate one with `openssl rand -hex 32`
- `HF_CACHE_DIR` — absolute path to the cache from step 1, forward slashes
  (e.g. `C:/Users/you/.cache/huggingface`)

### 3. Start

```bash
docker compose up -d vllm
docker compose logs -f vllm
```

Wait for `Application startup complete`. The first start is slow — the model is loaded onto the GPU and
CUDA graphs are captured. Subsequent starts are much faster.

## ✅ Verify

```bash
# no auth needed
curl http://127.0.0.1:8000/health

# everything else needs the key
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "chat", "messages": [{"role": "user", "content": "Hi"}]}'
```

Note `"model": "chat"` — that is `MODEL_ALIAS`, not `MODEL_ID`. Clients address the alias, so you can swap
the underlying model without touching client code.

## 🦜 Quick test with LangChain

```bash
pip install langchain-openai
```

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="<VLLM_API_KEY>",
    model="chat",  # MODEL_ALIAS, not MODEL_ID
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

print(llm.invoke("Say hello in one sentence.").content)
```

`extra_body` is passed straight through to vLLM and is what turns off Qwen3's reasoning mode. It matters:
on a short prompt, thinking-enabled generated 240 output tokens where thinking-disabled generated 2. Leave
it on when you actually want reasoning, off for everything else.

Streaming and chains work the same as with any `ChatOpenAI`:

```python
for chunk in llm.stream("Count to five."):
    print(chunk.content, end="", flush=True)
```

## 💬 Optional: Open WebUI

A browser chat UI, useful for eyeballing the model without writing code. Nothing depends on it.

```bash
docker compose up -d   # brings up both services
```

Then open http://127.0.0.1:3000 and create the first account (it becomes the admin).

The image is pinned to a release tag on purpose. For Open WebUI, `:latest` and `:main` resolve to the
*same* digest — a build off the main branch, not a stable release. Bump the pin deliberately after reading
the release notes.

**To run without it:** delete the `open-webui` service block and the `open_webui` volume from
`docker-compose.yml`. vLLM is unaffected — the `vllm` service references neither.

## ⚙️ Configuration

All settings live in `.env`.

| Variable | Default | What it does |
| --- | --- | --- |
| `MODEL_ID` | `Qwen/Qwen3-4B-AWQ` | HuggingFace repo to serve. AWQ/GPTQ safetensors only. |
| `MODEL_ALIAS` | `chat` | Name clients use in the `model` field. Keep it stable. |
| `HF_TOKEN` | *(empty)* | Only needed for gated repos. |
| `VLLM_API_KEY` | *(required)* | Bearer token for the API. |
| `GPU_MEM_UTIL` | `0.90` | Fraction of **total** VRAM for weights + KV cache. |
| `MAX_MODEL_LEN` | `8192` | Context window. The biggest VRAM knob. |
| `MAX_NUM_SEQS` | `64` | Requests batched concurrently; the rest queue. |
| `HF_CACHE_DIR` | *(required)* | Host HuggingFace cache, bind-mounted to `/hf`. |
| `BIND_ADDR` | `127.0.0.1` | Interface the API binds to. |
| `WEBUI_BIND_ADDR` | `127.0.0.1` | Interface the UI binds to. |

Rough AWQ 4-bit weight sizes, to sanity-check against your VRAM:

| Model | Weights | Notes |
| --- | --- | --- |
| `Qwen/Qwen3-4B-AWQ` | 2.48 GiB | Comfortable on 8 GB with room for KV cache |
| `Qwen/Qwen3-8B-AWQ` | 5.68 GiB | Very tight on 8 GB — short context only |
| `Qwen/Qwen3-14B-AWQ` | 9.29 GiB | Needs 16 GB or more |

On CUDA OOM, tune in this order: `MAX_MODEL_LEN` → `MAX_NUM_SEQS` → `GPU_MEM_UTIL` last, since lowering it
shrinks the KV cache you are trying to make room for.

## 🧩 Running a second model

One vLLM process serves exactly one model. For a second one, copy the `vllm` service under a new name with
its own port and `MODEL_ALIAS`, pointing at the same `HF_CACHE_DIR`:

```yaml
  vllm-embed:
    # ...same as vllm...
    command: >
      --model BAAI/bge-m3
      --served-model-name embed
      ...
    ports:
      - "${BIND_ADDR:-127.0.0.1}:8001:8000"
```

Both containers share one GPU, so split `GPU_MEM_UTIL` between them (e.g. 0.55 / 0.35) — each one's
fraction is of total VRAM, and they do not coordinate.

## ❓ FAQ

Q: **I get `401 Unauthorized` opening `http://127.0.0.1:8000/v1/models` in a browser.**
A: Expected. Every endpoint except `/health` requires `Authorization: Bearer <VLLM_API_KEY>`, and a browser doesn't send one. Use `curl` or a client. Hit `/health` to check the server is alive.

Q: **The container says "Up" but nothing responds.**
A: Cold start takes 10–20 minutes for a large model — the container is up long before the server is. Watch `docker compose logs -f vllm` and wait for `Application startup complete`. This is why the healthcheck has a 1800s `start_period`; a shorter one would kill the container mid-load and cause a restart loop.

Q: **`CUDA out of memory` at startup.**
A: Lower `MAX_MODEL_LEN` first, then `MAX_NUM_SEQS`, and only then `GPU_MEM_UTIL`. Note `GPU_MEM_UTIL` is a fraction of *total* VRAM, not free VRAM — on Windows the desktop compositor already holds several hundred MB before vLLM starts, which is why this box runs 0.80 rather than 0.90.

Q: **Can I use a GGUF model?**
A: No. GGUF is llama.cpp's format; vLLM's support for it is experimental, slow, and fails on many architectures. Use AWQ or GPTQ safetensors.

Q: **Tool / function calls return `400 Bad Request`.**
A: vLLM needs `--enable-auto-tool-choice` and a matching `--tool-call-parser`. Both are already set in `docker-compose.yml` (`hermes`, which is right for Qwen). A different model family may need a different parser.

Q: **Open WebUI starts unhealthy and never loads.**
A: On first boot it downloads a sentence-transformers embedding model. If that download was interrupted, it stalls behind stale `.incomplete` marker files in the cache. `HF_HUB_OFFLINE: "1"` (already set) forces it to use the cached copy instead. If it is still stuck, delete the `*.incomplete` files under `models--sentence-transformers--all-MiniLM-L6-v2` in your HF cache.

Q: **Why is the vLLM image ~30 GB?**
A: It bundles CUDA, PyTorch, and the compiled kernels. That is normal and a one-time download.

Q: **Qwen3 burns an absurd number of output tokens.**
A: Thinking mode is on by default. Disable it per request: `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`. On a short prompt this was the difference between 240 and 2 output tokens.

1: **`docker pull` fails on `ghcr.io` with DNS errors.**
DNS inside the `docker-desktop` WSL distro, not your network. It is usually transient — retry first. If it
persists, check `/etc/resolv.conf` inside that distro
(`wsl -d docker-desktop cat /etc/resolv.conf`) and point it at a working resolver.

**Running under WSL2?**
`VLLM_WSL2_ENABLE_PIN_MEMORY: "1"` is already set in the compose file. Without it, startup fails during
memory pinning.

**Connecting from a notebook running in another container.**
`127.0.0.1` refers to that container, not the host. Attach it to the `llm` network and use
`http://vllm:8000/v1`, or use `http://host.docker.internal:8000/v1`.

**Is mounting my HuggingFace cache safe?**
That directory also holds your HF `token` file, so the container can read it. Fine on a dev laptop. On a
shared or production box, use a separate cache directory with a scoped token — or no token at all.

**Why is Open WebUI pinned to `v0.11.0` instead of `:latest`?**
Because for this image `:latest` and `:main` resolve to the same digest — an unstable main-branch build,
not a release. A pinned release tag means `docker compose pull` months from now gives you the UI you
actually tested.
