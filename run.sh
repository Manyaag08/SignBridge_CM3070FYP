#!/usr/bin/env bash
# Start SignBridge: backend (Whisper, tone tagger, sentence smoother, call signalling) + web app on one port.
#   bash run.sh            then open http://localhost:8000 in Chrome
# First run creates .venv and installs backend/requirements.txt (Whisper + PyTorch: a few minutes, ~2 GB).
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r backend/requirements.txt
fi
if curl -s --max-time 1 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Ollama detected: sentence smoother will use ${SIGNBRIDGE_LLM_MODEL:-llama3.2:3b} (run: ollama pull ${SIGNBRIDGE_LLM_MODEL:-llama3.2:3b})"
else
  echo "Ollama not running: the sentence smoother will use its rule-based path only."
fi
exec .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port "${PORT:-8000}"
