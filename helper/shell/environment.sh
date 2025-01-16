#!/bin/bash
set -e
source source.sh

case $1 in
create | update)
	ACTION=$1
	;;
*)
	echo "Usage: $0 [create | update]"
	exit 1
	;;
esac

ENV_FILE="$ENV_PATH/env.yml"

if [ "$ACTION" == "create" ]; then
	conda env create -f "$ENV_FILE"
elif [ "$ACTION" == "update" ]; then
	conda env update -f "$ENV_FILE" --prune
fi
