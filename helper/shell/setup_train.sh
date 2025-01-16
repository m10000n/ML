#!/bin/bash
set -e
source source.sh

echo "---------->>> Starting setup of training server. <<<----------"
sync.sh to_train
source ssh_setup.sh
echo "----->>> Connecting to server. <<<-----"
# shellcheck disable=SC2087
ssh -i "$IDENTITY_FILE" -p "$PORT" "$USER"@"$HOST" -T <<EOF
set -e
BASH_CONFIG_FILE="\$HOME/.bashrc"

if ! command -v conda &> /dev/null; then
  echo "----->>> Start installing Miniconda3. <<<-----"
  mkdir -p ~/miniconda3
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
  bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
  rm ~/miniconda3/miniconda.sh
  source ~/miniconda3/bin/activate
  conda init --all > /dev/null
  echo "----->>> Finished installing Miniconda3. <<<-----"
fi

if [[ -z "\$PYTHONPATH" || ! ":\$PYTHONPATH:" =~ ":$BASE_DIR_REMOTE:" ]]; then
  echo "----->>> Adding python path. <<<-----"
  if [ -z "\$PYTHONPATH" ]; then
    echo "export PYTHONPATH=\"$BASE_DIR_REMOTE\"" >> "\$BASH_CONFIG_FILE"
  else
    echo "export PYTHONPATH=\"$BASE_DIR_REMOTE:\$PYTHONPATH\"" >> "\$BASH_CONFIG_FILE"
  fi
fi

source "\$BASH_CONFIG_FILE"
source "$SOURCE_REMOTE"

if [[ -z "\$PATH" || ! ":\$PATH:" =~ ":\$HELPER_PATH/shell:" ]]; then
  echo "----->>> Adding shell path. <<<-----"
  if [ -z "\$PATH" ]; then
    echo "export PATH=\"\$HELPER_PATH/shell\"" >> "\$BASH_CONFIG_FILE"
  else
    echo "export PATH=\"\$HELPER_PATH/shell:\$PATH\"" >> "\$BASH_CONFIG_FILE"
  fi
fi

if ! grep -Fxq "source \$ALIAS_TRAIN" "\$BASH_CONFIG_FILE"; then
  echo "----->>> Adding aliases. <<<-----"
  echo "source \$ALIAS_TRAIN" >> "\$BASH_CONFIG_FILE"
fi

if ! conda env list | grep -Fq "$ENV_NAME"; then
  echo "----->>> Start creating Anaconda environment. <<<-----"
  conda env create --file "\$ENV"
  echo "----->>> Finished creating Anaconda environment. <<<-----"
else
  echo "----->>> Start updating Anaconda environment. <<<-----"
  conda env update --name "$ENV_NAME" --file "\$ENV" --prune
  echo "----->>> Finished updating Anaconda environment. <<<-----"
fi

if ! grep -Fxq "conda activate $ENV_NAME" "\$BASH_CONFIG_FILE"; then
  echo "----->>> Making $ENV_NAME default environment. <<<-----"
  echo "conda activate $ENV_NAME" >> "\$BASH_CONFIG_FILE"
fi

if ! command -v rsync &> /dev/null; then
  echo "----->>> Start installing rsync. <<<-----"
  sudo yum install -y rsync
  sudo yum clean all
  echo "----->>> Finished installing rsync. <<<-----"
fi

if ! command -v htop &> /dev/null; then
  echo "----->>> Start installing htop. <<<-----"
  sudo yum install -y htop
  sudo yum clean all
  echo "----->>> Finished installing htop. <<<-----"
fi

if ! command -v tmux &> /dev/null; then
  echo "----->>> Start installing tmux. <<<-----"
  sudo yum install -y tmux
  sudo yum clean all
  echo "----->>> Finished installing tmux. <<<-----"
fi
cat <<EOT > ~/.tmux.conf
set -g mouse on
set -g history-limit 10000
EOT

EOF
echo "---------->>> Finished setup of training server. <<<----------"
