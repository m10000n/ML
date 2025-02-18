#!/bin/bash
set -e

BASE_DIR="$(python -m helper.path base_dir a)"
MODEL_PATH="$(python -m helper.path model a)"
RESULT_PATH="$(python -m helper.path result a)"
DATALOADER_PATH="$(python -m helper.path dataloader a)"
DATASET_PATH="$(python -m helper.path dataset a)"
CONFIG_PATH="$(python -m helper.path config a)"
HELPER_PATH="$(python -m helper.path helper a)"
SHELL_PATH="$(python -m helper.path shell a)"
ENV_PATH="$(python -m helper.path env a)"

BASE_DIR_REMOTE="$(python -m helper.path base_dir rmt)"
SOURCE_REMOTE="$(python -m helper.path shell rmt)/source.sh"

ENV="$ENV_PATH/env.yml"
ENV_NAME=$(grep 'name: ' "$ENV" | awk '{print $2}' | tr -d '[:space:]')

ALIAS_TRAIN="$HELPER_PATH/alias/alias_train"
