#!/bin/bash
set -e
source source.sh

PRINT_START primary "Start cleaning up Python code."
PRINT_START secondary "Start removing unused imports."
autoflake --remove-all-unused-imports --recursive --in-place --exclude "model/**/result/**" "$PROJECT_ROOT"
PRINT_END secondary "Finished removing unused imports."
PRINT_START secondary "Start sorting imports."
isort "$PROJECT_ROOT" --profile black --skip-glob "model/**/result/**"
PRINT_END secondary "Finished sorting imports."
PRINT_START primary "Start formating."
black "$PROJECT_ROOT" --exclude "model/.*/result/.*"
PRINT_END secondary "Finished formating."
PRINT_END primary "Finished cleaning up Python code."

if command -v shfmt &>/dev/null; then
	PRINT_START primary "Start formating shell script."
	shfmt -w "$SHELL_PATH"
	PRINT_END primary "Finished formating shell script."
else
	echo "To format shell script install 'shfmt'."
fi
