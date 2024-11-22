#!/bin/bash
set -e
source "$(python -m helper.path local_shell)/source.sh"

# local

#ToDo: remote_definition deprecated in anaconda
#ToDo: updates

echo "---------->>Starting setup of training server<<----------"
source $TO_TRAIN

echo "----->>Connecting to server<<-----"
# shellcheck disable=SC2087
ssh -i "$IDENTITY_FILE" -p "$PORT" "$USER"@"$HOST" -T << EOF
set -e
BASH_CONFIG_FILE="\$HOME/.bashrc"

chmod -R +x "$REMOTE_SHELL"

if ! grep -q "source $REMOTE_ALIAS" "\$BASH_CONFIG_FILE"; then
    echo "source $REMOTE_ALIAS" >> "\$BASH_CONFIG_FILE"
fi

if ! command -v rsync &> /dev/null; then
  echo "----->>Start installing rsync<<-----"
  sudo yum install -y rsync
  sudo yum clean all
  echo "----->>Finished installing rsync<<-----"
fi

if ! command -v tmux &> /dev/null; then
  echo "----->>Start installing tmux<<-----"
  sudo yum install -y tmux
  sudo yum clean all
  echo "----->>Finished installing tmux<<-----"
fi

if ! command -v conda &> /dev/null; then
  echo "----->>Start installing Miniconda3<<-----"
  mkdir -p ~/miniconda3
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
  bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
  rm ~/miniconda3/miniconda.sh
  source ~/miniconda3/bin/activate
  conda init --all > /dev/null
  echo "----->>Finished installing Miniconda3<-----"
fi

if ! conda env list | grep -q "$ENV_NAME"; then
  echo "----->>Start creating Anaconda environment<<-----"
  conda env create --file "$REMOTE_ENV"
  echo "----->>Finished creating Anaconda environment<<-----"
else
  echo "----->>Start updating Anaconda environment<<-----"
  conda env update --name "$ENV_NAME" --file "$REMOTE_ENV" --prune
  echo "----->>Finished updating Anaconda environment<<-----"
fi
EOF
echo "---------->>Finished setup of training server<<----------"