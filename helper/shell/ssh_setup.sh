#!/bin/bash
set -e

HOST=$(python -m config.ssh host)
USER=$(python -m config.ssh user)
PORT=$(python -m config.ssh port)
IDENTITY_FILE=$(python -m config.ssh identity_file)

if ! ssh-keygen -F "$HOST" >/dev/null; then
	echo "----->>> Start adding host. <<<-----"
	ssh-keyscan -H "$HOST" >>~/.ssh/known_hosts
	echo "----->>> Finished adding host <<<-----"
fi

if ! pgrep -x "ssh-agent" >/dev/null; then
	eval "$(ssh-agent -s)" >/dev/null
fi

STARTED=false
while ! ssh-add -l | grep -q "$(ssh-keygen -lf "$IDENTITY_FILE" | awk '{print $2}')"; do
	if [ "$STARTED" = false ]; then
		echo "----->>> Start adding password for ssh key. <<<-----"
		STARTED=true
	fi

	if ssh-add -t $((3600 * 24)) "$IDENTITY_FILE"; then
		echo "----->>> Finished adding password for ssh key. <<<-----"
		break
	else
		echo "Failed to add SSH key. Please enter the correct passphrase."
	fi
done
