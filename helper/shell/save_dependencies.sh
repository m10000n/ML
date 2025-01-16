#!/bin/bash
set -e

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 <output_file_path>"
	exit 1
fi

OUTPUT_FILE=$1

if [ -z "$CONDA_DEFAULT_ENV" ]; then
	echo "No active Conda environment found. Please activate a Conda environment and try again."
	exit 1
fi

conda list --export >"$OUTPUT_FILE"

chmod 444 "$OUTPUT_FILE"
