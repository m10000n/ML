#!/bin/bash
set -e
source source.sh

BASH_CONFIG_FILE="$HOME/.bashrc"

echo -e "${START_COLOR}${PRIMARY_0}Start setup.${PRIMARY_1}${RESET_COLOR}"

# miniconda3
if command -v conda &> /dev/null; then
  echo -e "${INFO_COLOR}${SECONDARY_0}Anaconda already installed.${SECONDARY_1}${RESET_COLOR}"
else
  echo -e "${START_COLOR}${SECONDARY_0}Start installing miniconda3.${SECONDARY_1}${RESET_COLOR}"
  mkdir -p ~/miniconda3
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
  bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
  rm ~/miniconda3/miniconda.sh
  source ~/miniconda3/bin/activate
  conda init --all > /dev/null
  echo -e "${END_COLOR}${SECONDARY_0}Finished installing miniconda3.${SECONDARY_1}${RESET_COLOR}"
fi

# env
if ! conda env list | grep -Fq "$ENV_NAME"; then
  env.sh create
else
  env.sh update
fi

# PYTHONPATH
if ! grep -q "^export PYTHONPATH=.*$PROJECT_ROOT" "$BASH_CONFIG_FILE"; then
    PP=$(grep '^export PYTHONPATH=' "$BASH_CONFIG_FILE" | cut -d '=' -f2- | tr -d '"')
    printf 'export PYTHONPATH="%s%s"\n' "$PROJECT_ROOT" "${PP:+:$PP}" >> "$BASH_CONFIG_FILE"
    echo -e "${INFO_COLOR}${SECONDARY_0}Added Python path.${SECONDARY_1}${RESET_COLOR}"
fi

# PATH
if ! grep -q "^export PATH=.*$SHELL_PATH" "$BASH_CONFIG_FILE"; then
    SP=$(grep '^export PATH=' "$BASH_CONFIG_FILE" | cut -d '=' -f2- | tr -d '"')
    printf 'export PATH="%s%s"\n' "$SHELL_PATH" "${SP:+:$SP}" >> "$BASH_CONFIG_FILE"
    echo -e "${INFO_COLOR}${SECONDARY_0}Added shell path.${SECONDARY_1}${RESET_COLOR}"
fi



#if [[ -z "\$PATH" || ! ":\$PATH:" =~ ":\$HELPER_PATH/shell:" ]]; then
#  echo "----->>> Adding shell path. <<<-----"
#  if [ -z "\$PATH" ]; then
#    echo "export PATH=\"\$HELPER_PATH/shell\"" >> "\$BASH_CONFIG_FILE"
#  else
#    echo "export PATH=\"\$HELPER_PATH/shell:\$PATH\"" >> "\$BASH_CONFIG_FILE"
#  fi
#fi
#
#if ! grep -Fxq "source \$ALIAS_TRAIN" "\$BASH_CONFIG_FILE"; then
#  echo "----->>> Adding aliases. <<<-----"
#  echo "source \$ALIAS_TRAIN" >> "\$BASH_CONFIG_FILE"
#fi
#
#if ! conda env list | grep -Fq "$ENV_NAME"; then
#  echo "----->>> Start creating Anaconda environment. <<<-----"
#  conda env create --file "\$ENV"
#  echo "----->>> Finished creating Anaconda environment. <<<-----"
#else
#  echo "----->>> Start updating Anaconda environment. <<<-----"
#  conda env update --name "$ENV_NAME" --file "\$ENV" --prune
#  echo "----->>> Finished updating Anaconda environment. <<<-----"
#fi
#
#if ! grep -Fxq "conda activate $ENV_NAME" "\$BASH_CONFIG_FILE"; then
#  echo "----->>> Making $ENV_NAME default environment. <<<-----"
#  echo "conda activate $ENV_NAME" >> "\$BASH_CONFIG_FILE"
#fi
#
#if ! command -v rsync &> /dev/null; then
#  echo "----->>> Start installing rsync. <<<-----"
#  sudo yum install -y rsync
#  sudo yum clean all
#  echo "----->>> Finished installing rsync. <<<-----"
#fi
#
#if ! command -v htop &> /dev/null; then
#  echo "----->>> Start installing htop. <<<-----"
#  sudo yum install -y htop
#  sudo yum clean all
#  echo "----->>> Finished installing htop. <<<-----"
#fi
#
#if ! command -v tmux &> /dev/null; then
#  echo "----->>> Start installing tmux. <<<-----"
#  sudo yum install -y tmux
#  sudo yum clean all
#  echo "----->>> Finished installing tmux. <<<-----"
#fi
#cat <<EOT > ~/.tmux.conf
#set -g mouse on
#set -g history-limit 10000
#EOT
#
#EOF
#echo "---------->>> Finished setup of training server. <<<----------"
