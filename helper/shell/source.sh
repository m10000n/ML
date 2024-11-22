#!/bin/bash
set -e

LOCAL_BASE_DIR="$(python -m helper.path local_base_dir)"
REMOTE_BASE_DIR="$(python -m helper.path remote_base_dir)"

LOCAL_ENV="$(python -m helper.path local_env)/env.yml"
REMOTE_ENV="$(python -m helper.path remote_env)/env.yml"
ENV_NAME=$(grep 'name: ' "$LOCAL_ENV" | awk '{print $2}' | tr -d '[:space:]')

SHELL=$(python -m helper.path shell)
LOCAL_SHELL=$(python -m helper.path local_shell)
REMOTE_SHELL=$(python -m helper.path remote_shell)
SSH_SETUP="$SHELL/ssh_setup.sh"
TO_TRAIN="$SHELL/to_train.sh"
STOP_AWS="$SHELL/stop_aws.sh"

REMOTE_HELPER=$(python -m helper.path remote_helper)
REMOTE_ALIAS="$REMOTE_HELPER/alias/alias_train"

DATASET=$(python -m helper.path dataset -m)
MODEL=$(python -m helper.path model -m)