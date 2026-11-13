#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------
# Environment variables setup
# ----------------------------------------------------------------------
# LLM credentials - must be configured for your provider
export LLM_API_KEY="sk-__gjdKkr5rNcW8bhWz-8rA"
export LLM_BASE_URL="https://api.neuraldeep.ru/v1"

# Paths - adjust to your local environment
export TASKS="$(pwd)/examples/task_estimate_calibrator/data/tasks"
export OUT="$(pwd)/out"

# Container identity
export TEAM="net_team"

# ----------------------------------------------------------------------
# Prepare output directory and run the container
# ----------------------------------------------------------------------
mkdir -p "$OUT" && chmod 777 "$OUT"

docker build -t "hack:latest" -f examples/task_estimate_calibrator/Dockerfile . && \
timeout -k 1000s 1h docker run --rm --name "run-${TEAM}" \
  --network hack-net \
  -e LLM_BASE_URL -e LLM_API_KEY \
  -v "$TASKS:/data/tasks:rw" \
  --read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp \
  --cap-drop ALL --security-opt no-new-privileges \
  --cpus 2 --memory 1g --pids-limit 512 \
  --workdir /app --entrypoint python \
  "hack:latest" \
  /app/examples/task_estimate_calibrator/main.py --tasks-dir /data/tasks