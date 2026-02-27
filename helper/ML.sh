#!/bin/bash

python3 -m helper.env is_ready
env_is_ready=$?

if [[ "$env_is_ready" -eq 0 ]]; then
    python3 -m helper.debug is_active
    debug_is_active=$?

    if [[ "$debug_is_active" -eq 0 ]]; then
        (
            eval "$(python3 -m helper.env get_activate_command)"
            python3 -m helper.clt.main "$@"
        )
    else
        (
            eval "$(python3 -m helper.env get_activate_command)"
            python3 -O -m helper.clt.main "$@"
        )
    fi
    exit_code=$?
elif [[ "$env_is_ready" -eq 1 ]]; then
    python3 -m helper.clt.fallback "$@"
    exit_code=$?
else
    echo "This should never happen" >&2
    return 1
fi

if [[ $1 == "setup" ]]; then
    source "$(python3 -m helper.bash get_config_path)"
elif [[ "$1" == "history" ]]; then
    history -r
fi

return $exit_code
