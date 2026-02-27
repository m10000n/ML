#!/bin/bash

export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$(pwd)/helper/shell${PATH:+:$PATH}"

ML() {
    source helper/ML.sh "$@"
}
