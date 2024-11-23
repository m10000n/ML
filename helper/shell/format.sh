#!/bin/bash
set -e
source "$(python -m helper.path local_shell)/source.sh"

# local

EXCLUDE_PATTERN="$DATALOADER_PATH/*/data/*"

echo "---------->>Start formating<<---------"
echo "----->>Start removing unused imports<<-----"
autoflake --remove-all-unused-imports --recursive --in-place --exclude "$EXCLUDE_PATTERN" "$LOCAL_BASE_DIR"
echo "----->>Finished removing unused imports<<-----"
echo "----->>Start sorting imports<<-----"
isort "$LOCAL_BASE_DIR" --skip-glob "$EXCLUDE_PATTERN"
echo "----->>Finished sorting imports<<-----"
echo "----->>Start linting<<-----"
black "$LOCAL_BASE_DIR" --exclude "$EXCLUDE_PATTERN"
echo "----->>Finished linting<<-----"
echo "---------->>Finished to formating<<---------"