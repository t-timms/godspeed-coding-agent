#!/usr/bin/env bash
# Start llama-server for Qwen3.8-27B (ISTA GSQ-RCO IQ3_XXS with the embedded MTP head) on a 16 GB GPU.
# The configuration measured in docs/local_qwen38_27b.md (RTX 5070 Ti, WSL2, llama.cpp 3173a56):
# ~78 tok/s on real agent requests (54.6 without speculation), peak VRAM 14,128 MiB of 16,303 MiB.
#
# Needs a llama.cpp build that has MTP (upstream PR #22673). MTP requires exactly one slot (-np 1).
# Override paths with LLAMA_SERVER / MODEL; PORT defaults to 8080 (where Godspeed attaches).
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-$HOME/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf}"
PORT="${PORT:-8080}"
CTX="${CTX:-32768}"

[ -x "$LLAMA_SERVER" ] || { echo "llama-server not found: $LLAMA_SERVER (set LLAMA_SERVER)" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "model not found: $MODEL (set MODEL)" >&2; exit 1; }

exec "$LLAMA_SERVER" \
  --model "$MODEL" \
  -c "$CTX" -ngl 999 -fa on -np 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -b 2048 -ub 512 -t 4 \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  --ctx-checkpoints 32 --cache-ram 8192 \
  --jinja --no-webui --host 127.0.0.1 --port "$PORT"
