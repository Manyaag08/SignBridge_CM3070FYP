#!/usr/bin/env bash
# Re-run the complete SignBridge evaluation. Results are written to artifacts/eval/.
#
#   bash evaluation/run_all.sh
#
# First run: creates evaluation/.venv-eval and installs evaluation/requirements-eval.txt, downloads the
# ASLNow! landmarks (~8 MB) and, on first use, the tone model and small LLM from Hugging Face (~1.5 GB).
# Needs: python3, macOS `say` (synthetic voices), internet, and the ASL_Recognition clone (see README).
# Surrey landmarks are cached in artifacts/eval/surrey_landmarks.csv; to rebuild them from the images set
# SURREY_DIR to the extracted fingerspelling5/dataset5 folder and delete that CSV.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=evaluation/.venv-eval
if [ ! -x "$VENV/bin/python" ]; then
  echo "== creating $VENV (one-off)"
  python3 -m venv --system-site-packages "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r evaluation/requirements-eval.txt
fi
PY="$VENV/bin/python"

[ -d ASL_Recognition ] || { echo "Missing ASL_Recognition - run: git clone https://github.com/MohdDilshad-nitk/ASL_Recognition"; exit 1; }
export ASLNOW_DIR="${ASLNOW_DIR:-evaluation/data/aslnow}"
"$PY" evaluation/fetch_aslnow.py "$ASLNOW_DIR"
export SURREY_DIR="${SURREY_DIR:-}"
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=error

step() { echo; echo "== $1"; }
step "single-signer results (Chapter 4)";          "$PY" run_experiment.py
step "multi-signer: leave-one-source-out";          "$PY" evaluation/eval_multisigner.py
step "calibration temperature (validation signer)"; "$PY" evaluation/select_calibration_temperature.py
step "multi-signer: calibration + ONNX export";      "$PY" evaluation/eval_multisigner.py
step "Whisper speech-to-text";                      "$PY" evaluation/eval_speech.py
step "tone tagger";                                 "$PY" evaluation/eval_tone.py
step "sentence smoother";                           "$PY" evaluation/eval_smoother.py
step "message typing + decoder (validation)";     "$PY" evaluation/tune_message_decoding.py
step "end to end";                                  "$PY" evaluation/eval_end_to_end.py
step "tests";                                       "$PY" -m pytest tests -q
step "summary";                                     "$PY" evaluation/summarise.py > /dev/null
echo; echo "Done - see artifacts/eval/EVALUATION_SUMMARY.md"
