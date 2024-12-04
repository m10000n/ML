#!/bin/bash
set -e
source "$(python -m helper.path local_shell)/source.sh"

# local

source $SSH_SETUP

ssh -t -i "$IDENTITY_FILE" -p "$PORT" "$USER"@"$HOST" "
set -e
if tmux has-session -t my_session 2>/dev/null; then
  tmux kill-session -t my_session
  echo '----->>> tmux session killed. <<<-----'
else
  echo '----->>> There is no tmux session to kill. <<<-----'
fi" 2>&1 | grep -v 'Connection to [0-9]\+\.[0-9]\+\.[0-9]\+\.[0-9]\+ closed.'
