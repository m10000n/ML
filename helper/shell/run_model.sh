#!/bin/bash
set -e
source "$(python -m helper.path local_shell)/source.sh"

# local

ACTIVATE_ENV="cd $REMOTE_BASE_DIR && conda activate $ENV_NAME"
RUN_MODEL="python -m $MODEL.pipeline"
INFO="This server will be stopped after running the pipeline."

source $TO_TRAIN

# shellcheck disable=SC2087
ssh -i "$IDENTITY_FILE" -p "$PORT" "$USER"@"$HOST" -T << EOF
set -e
if tmux has-session -t my_session 2>/dev/null; then
  echo '----->>> Cannot run the model on the server. There is already an existing tmux session. <<<-----'
else
  echo '----->>> Starting a new tmux session. <<<-----'
  tmux new-session -d -s my_session
  tmux send-keys -t my_session "clear; $ACTIVATE_ENV && echo $INFO && $RUN_MODEL; $STOP_AWS" C-m
  echo '----->>> Model runs on server. <<<-----'
fi
EOF
