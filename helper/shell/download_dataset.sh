#!/bin/bash
set -e
source source.sh

DOWNLOAD_DATASET='python -m "$(python -m helper.path dataset m).dataset"'
INFO="This server will be stopped after downloading the dataset."

stop_aws.sh --test
tmux.sh new_session
echo "----->>> Starting dataset download. <<<-----"
tmux send-keys -t my_session "clear && echo $INFO && $DOWNLOAD_DATASET; tmux.sh write_log; stop_aws.sh" C-m
