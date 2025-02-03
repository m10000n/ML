#!/bin/bash
set -e

case $1 in
to_train | copy_result)
	ACTION=$1
	;;
*)
	echo "Usage: $0 [to_train | copy_result]"
	exit 1
	;;
esac

python -m helper.sync "$ACTION"
