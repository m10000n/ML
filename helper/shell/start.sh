#chmod -R +x helper/shell
source ./helper/alias/alias
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PATH="$(pwd)/helper/shell:$PATH"