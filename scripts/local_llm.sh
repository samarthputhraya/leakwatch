#!/usr/bin/env bash
# `make llm-local`: a local OpenAI-compatible model for the copilot on CPU (no key, no GPU).
#
# Downloads (once, SHA-256 checked) the llama.cpp CPU server b12038 and Qwen3-4B-Instruct-2507
# (Q3_K_M GGUF, 2.08 GB, Apache-2.0) into $LLM_HOME (default ~/llm), then serves it on
# http://127.0.0.1:${LLM_PORT:-8080}/v1 with tool calling (--jinja). Point the copilot at it with
#   LLM_BASE_URL=http://127.0.0.1:8080/v1  LLM_API_KEY=local  LLM_MODEL=qwen3-4b-instruct-2507
# and use configs/local_llm.yaml (longer timeouts for CPU inference). A hosted model is faster and
# stronger; this is for development and for running the evaluation without an account.
set -euo pipefail
LLM_HOME="${LLM_HOME:-$HOME/llm}"
PORT="${LLM_PORT:-8080}"
THREADS="${LLM_THREADS:-8}"
BUILD=b12038
BIN_TGZ="llama-$BUILD-bin-ubuntu-x64.tar.gz"
BIN_SHA=46b89b9a84c7a0fa46e89911034c4f6d3024150cb5d477e1426b921f81f4ddab
MODEL=Qwen3-4B-Instruct-2507-Q3_K_M.gguf
MODEL_SHA=9c6e0763577125a994a9bea0bbd7a737ac4498b8a6a4e0f788727553af1806c9

mkdir -p "$LLM_HOME/bin" "$LLM_HOME/models"
cd "$LLM_HOME"
if [ ! -x "bin/llama-$BUILD/llama-server" ]; then
  curl -sSL --retry 5 -o "bin/$BIN_TGZ" \
    "https://github.com/ggml-org/llama.cpp/releases/download/$BUILD/$BIN_TGZ"
  echo "$BIN_SHA  bin/$BIN_TGZ" | sha256sum -c -
  tar -xzf "bin/$BIN_TGZ" -C bin
fi
if [ ! -f "models/$MODEL" ]; then
  curl -sSL --retry 5 -C - -o "models/$MODEL.part" \
    "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/$MODEL"
  echo "$MODEL_SHA  models/$MODEL.part" | sha256sum -c -
  mv "models/$MODEL.part" "models/$MODEL"
fi

exec "bin/llama-$BUILD/llama-server" -m "models/$MODEL" --alias qwen3-4b-instruct-2507 \
  --host 127.0.0.1 --port "$PORT" --jinja -c 16384 -fa on -ctk q8_0 -ctv q8_0 \
  -t "$THREADS" --parallel 1 --temp 0
