#!/bin/bash
set -e
source "$(python -m helper.path local_shell)/source.sh"

# local

ACTIVATE_ENV="cd $REMOTE_BASE_DIR && conda activate $ENV_NAME"
DOWNLOAD_DATASET="python -m $DATASET.dataset"
INFO="This server will be stopped after downloading the dataset."

source $TO_TRAIN

# shellcheck disable=SC2087
ssh -i "$IDENTITY_FILE" -p "$PORT" "$USER"@"$HOST" -T << EOF
set -e
if tmux has-session -t my_session 2>/dev/null; then
  echo '----->>> Cannot start downloading the dataset on the server. There is already an existing tmux session. <<<-----';
else
  echo '----->>> Starting a new tmux session. <<<-----'
  tmux new-session -d -s my_session
  tmux send-keys -t my_session "clear; $ACTIVATE_ENV && echo $INFO && $DOWNLOAD_DATASET; $STOP_AWS" C-m
  echo '----->>> Start downloading the dataset on the server. <<<-----'
fi
EOF
