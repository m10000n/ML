## Setup
in root directory of this project:  
`chmod -R +x helper/shell`  
`source ./helper/alias/alias`  

#### Anaconda environment
change environment name in */helper/env/env.yml*  
create: `create_env`  
update: `update_env`


## Server Setup
ssh configuration in */config/ssh.py*  
`setup_train`

---

## Commands
#### format
format all python files in the project: `format`

#### transfer
transfer configuration in */config/transfer.py*  
to training server: `to_train`  
from training server: `from_train`

#### run model on server  
model configuration in */config/config.py*  
`run_model`

#### download dataset on server  
dataset configuration */config/config.py*  
`download_dataset`


## Commands on server
attach to tmux session: `attach`

---

## Prerequisites
- Anaconda
- rsync
- server runs Amazon Linux

---

## AWS
- inbound: ssh
- outbound: https
- permissions policy: ec2:StopInstances