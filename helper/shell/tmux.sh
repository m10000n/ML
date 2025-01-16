#!/bin/bash
set -e

case $1 in
new_session | attach | kill_session | write_log)
	ACTION=$1
	;;
*)
	echo "Usage: $0 [new_session | attach | kill_session | write_log]"
	exit 1
	;;
esac

if [ "$ACTION" == "new_session" ]; then
	if tmux has-session -t my_session 2>/dev/null; then
		echo "----->>> Cannot create a new tmux session. There is already an existing one. <<<-----"
		exit 1
	else
		echo "----->>> Starting a new tmux session. <<<-----"
		tmux new-session -d -s my_session
	fi
elif [ "$ACTION" == "attach" ]; then
	if tmux has-session -t my_session 2>/dev/null; then
		tmux attach-session -t my_session
	else
		echo "----->>> There is no tmux session to attach. <<<-----"
	fi
elif [ "$ACTION" == "kill_session" ]; then
	if tmux has-session -t my_session 2>/dev/null; then
		tmux kill-session -t my_session
		echo "----->>> tmux session killed. <<<-----"
	else
		echo "----->>> There is no tmux session to kill. <<<-----"
		exit 1
	fi
elif [ "$ACTION" == "write_log" ]; then
	tmux capture-pane -p -t my_session:0.0 >~/.tmux.log
fi
