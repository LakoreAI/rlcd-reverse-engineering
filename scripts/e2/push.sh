#!/usr/bin/env bash
# Copy this repo (code + configs, no venv/checkpoints/results) to a rented
# GPU machine over plain ssh — no rsync or git needed on the remote.
#
#   bash scripts/e2/push.sh <host> <port> [user]
set -euo pipefail
HOST=${1:?usage: push.sh <host> <port> [user]}
PORT=${2:?usage: push.sh <host> <port> [user]}
USER_=${3:-root}
cd "$(dirname "$0")/../.."

# Record exactly which code the runs used (uncommitted changes included).
{
  echo "rev: $(git rev-parse HEAD)"
  echo "dirty: $(git status --porcelain | wc -l | tr -d ' ') changed files"
  git diff HEAD --stat
} > GIT_REV.txt

COPYFILE_DISABLE=1 tar czf - --no-mac-metadata \
  --exclude=.env --exclude=.venv --exclude=.git --exclude=checkpoints --exclude=results \
  --exclude=__pycache__ --exclude=.pytest_cache --exclude=.ruff_cache \
  --exclude='*.ipynb_checkpoints' . \
  | ssh -p "$PORT" "$USER_@$HOST" \
      'mkdir -p ~/rlcd-reverse-engineering && tar xzf - -C ~/rlcd-reverse-engineering'
rm -f GIT_REV.txt
echo "pushed to $USER_@$HOST:~/rlcd-reverse-engineering"
