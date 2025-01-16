#!/bin/bash
set -e

case $1 in
to_train | from_train)
	ACTION=$1
	;;
*)
	echo "Usage: $0 [to_train | from_train]"
	exit 1
	;;
esac

source ssh_setup.sh

if [ "$ACTION" == "to_train" ]; then
	python -m helper.sync to_train
elif [ "$ACTION" == "from_train" ]; then
	python -m helper.sync from_train
fi
