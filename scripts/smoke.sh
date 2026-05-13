#!/usr/bin/env bash
set -euo pipefail
base=${1:-http://localhost:8080}
audio=${2:-debug_recording.wav}
curl -sS "$base/health" | python3 -m json.tool
curl -sS -X POST "$base/transcribe" -F "file=@$audio" | python3 -m json.tool
