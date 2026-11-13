#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------
# Environment variables setup
# ----------------------------------------------------------------------
# LLM credentials - must be configured for your provider
export LLM_API_KEY="sk-__gjdKkr5rNcW8bhWz-8rA"
export LLM_BASE_URL="https://api.neuraldeep.ru/v1"

# Paths - adjust to your local environment
export TASKS="/home/dasha/juiwen/dashas_team/examples/"
export OUT="$(pwd)/out"

# Container identity
export TEAM="net_team"

# ----------------------------------------------------------------------
# Prepare output directory and run the container
# ----------------------------------------------------------------------
mkdir -p "$OUT" && chmod 777 "$OUT"

docker build -t "hack:latest" . && \
timeout -k 1000s 1h docker run --rm --name "run-${TEAM}" \
  --network hack-net \
  -e LLM_BASE_URL -e LLM_API_KEY \
  -v "$TASKS:/data/tasks:rw" \
  --read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp \
  --cap-drop ALL --security-opt no-new-privileges \
  --cpus 8 --memory 16g --pids-limit 512 \
  --workdir /app --entrypoint python \
  "hack:latest" \
  /app/run_agent.py --tasks-dir /data/tasks