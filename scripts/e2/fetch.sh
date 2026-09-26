#!/usr/bin/env bash
# Pull E2 results (JSON metrics, logs, gradient diagnostics — not the ~5 GB
# checkpoints) back from the rented machine into ./results/.
#
#   bash scripts/e2/fetch.sh <host> <port> [user]
set -euo pipefail
HOST=${1:?usage: fetch.sh <host> <port> [user]}
PORT=${2:?usage: fetch.sh <host> <port> [user]}
USER_=${3:-root}
cd "$(dirname "$0")/../.."
mkdir -p results
ssh -p "$PORT" "$USER_@$HOST" \
    'cd ~/sev && tar czf - results' \
  | tar xzf - -C .
echo "fetched into ./results:"
ls -1 results
