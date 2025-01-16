#!/bin/bash
set -e
source source.sh

case $1 in
download_dataset | run_model | attach | kill_tmux)
	ACTION=$1
	;;
*)
	echo "Usage: $0 [download_dataset | run_model | attach | kill_tmux]"
	exit 1
	;;
esac

source ssh_setup.sh
if [[ "$ACTION" == "download_dataset" || "$ACTION" == "run_model" ]]; then
	sync.sh to_train
fi

REMOTE_COMMAND="set -e"

if [ "$ACTION" == "download_dataset" ]; then
	REMOTE_COMMAND="$REMOTE_COMMAND; download_dataset.sh"
elif [ "$ACTION" == "attach" ]; then
	REMOTE_COMMAND="$REMOTE_COMMAND; tmux.sh attach"
elif [ "$ACTION" == "kill_tmux" ]; then
	REMOTE_COMMAND="$REMOTE_COMMAND; tmux.sh kill_session"
elif [ "$ACTION" == "run_model" ]; then
	REMOTE_COMMAND="$REMOTE_COMMAND; run_model.sh"
fi

echo "---------->>> Connecting to server. <<<----------"
ssh -t -i "$IDENTITY_FILE" -p "$PORT" "$USER"@"$HOST" "$REMOTE_COMMAND" 2>&1 | grep -v 'Connection to [0-9]\+\.[0-9]\+\.[0-9]\+\.[0-9]\+ closed.'
echo "---------->>> Connection to server closed. <<<----------"
