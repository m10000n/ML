**This branch contains a reference implementation of the Vision Transformer (ViT) model, as introduced in “An Image is Worth 16x16 Words: Transformers for Image 
Recognition at Scale” by Dosovitskiy et al. (2020).**

---

## Setup
in root directory of this project:  
`chmod -R +x helper/shell`  
`source ./helper/alias/alias`  
`export PYTHONPATH="$(pwd):$PYTHONPATH"`  
`export PATH="$(pwd)/helper/shell:$PATH"`

### Anaconda environment
change environment name in */helper/env/env.yml*  
create environment: `create_env`  
update environment: `update_env`


## Server Setup
ssh configuration in */config/ssh.py*  
setup the server: `setup_train`

---

## Commands
#### format
format all Python files and shell scripts in this project: `format`

#### transfer
transfer configuration in */config/transfer.py*  
transfer to training server: `to_train`  
transfer from training server: `from_train`

#### download
dataset configuration */config/config.py*  
download dataset on server: `download_dataset`

#### run model
model configuration in */config/config.py*  
run model on server: `run_model`

#### tmux
attach to tmux session on server: `attach`  
kill tmux session on server: `kill_tmux`


## Commands on server
#### download
download dataset: `download_dataset`

#### run model
run model: `run_model`

#### tmux
attach to tmux session: `attach`  
kill tmux session: `kill_tmux`

#### monitor
monitor gpu usage: `gpu_usage`  
monitor cpu usage: `cpu_usage`

---

## Prerequisites
- Anaconda
- rsync
- server runs Amazon Linux and GPU drivers are installed

---

## AWS
- inbound: ssh
- outbound: https
- permissions policies: AmazonS3ReadOnlyAccess, ec2:StopInstances, SecretsManagerReadWrite
