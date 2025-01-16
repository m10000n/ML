#!/bin/bash
set -e
source source.sh

RUN_MODEL='python -m "$(python -m helper.path model m).pipeline"'
INFO="This server will be stopped after running the model."

tmux.sh new_session
stop_aws.sh --test
echo "----->>> Start running model. <<<-----"
tmux send-keys -t my_session "clear && echo $INFO && $RUN_MODEL; tmux.sh write_log; stop_aws.sh" C-m
