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
	echo -e "${START_COLOR}${SECONDARY_0}Start creating Anaconda environment.${SECONDARY_1}${RESET_COLOR}"
	conda env create -f "$ENV_FILE"
	echo -e "${END_COLOR}${SECONDARY_0}Finished creating Anaconda environment.${SECONDARY_1}${RESET_COLOR}"
elif [ "$ACTION" == "update" ]; then
	echo -e "${START_COLOR}${SECONDARY_0}Start updating Anaconda environment.${SECONDARY_1}${RESET_COLOR}"
	conda env update -f "$ENV_FILE" --prune
	echo -e "${END_COLOR}${SECONDARY_0}Finished updating Anaconda environment.${SECONDARY_1}${RESET_COLOR}"
fi
