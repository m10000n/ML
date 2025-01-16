#!/bin/bash
set -e
source source.sh

echo "---------->>> Start cleaning up Python code. <<<---------"
echo "----->>> Start removing unused imports. <<<-----"
autoflake --remove-all-unused-imports --recursive --in-place --exclude "$DATASET_PATH/data/*,$MODEL_PATH/result/*" "$BASE_DIR"
echo "----->>> Finished removing unused imports. <<<-----"
echo "----->>> Start sorting imports. <<<-----"
isort "$BASE_DIR" --profile black --skip-glob "$DATASET_PATH/data/*" --skip-glob "$MODEL_PATH/result/*"
echo "----->>> Finished sorting imports. <<<-----"
echo "----->>> Start formating. <<<-----"
black "$BASE_DIR" --exclude "$DATASET_PATH/data/.*|$MODEL_PATH/result/.*"
echo "----->>> Finished formating. <<<-----"
echo "---------->>> Finished cleaning up Python code. <<<---------"

if command -v shfmt &>/dev/null; then
	echo "---------->>> Start formating shell script. <<<---------"
	shfmt -w "$SHELL_PATH"
	echo "---------->>> Finished formating shell script. <<<---------"
else
	echo "To format shell script install shfmt."
fi
