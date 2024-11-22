#!/bin/bash
set -e
source "$(python -m helper.path local_shell)/source.sh"

# local

source $SSH_SETUP
python -m helper.sync from_train