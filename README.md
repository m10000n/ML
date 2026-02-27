# ML Dev Toolkit
ML is a command-line tool designed to streamline machine learning development. It automates common tasks and speeds up iteration across models, datasets, and hyperparameters. It includes reference implementations of machine learning papers.

## Features
- Environment
    - Install prerequisites
    - Set up Python environment
- System
    - Show system information
    - Monitor CPU and GPU resources
    - Configure: seed, device (CPU/GPUs), autocast, number of workers, prefetch
- Models
    - Show summaries
    - Show forward-pass FLOPs
- Experiments
    - Run experiments (with optional cross-validation and timing)
    - Show results
    - Plot results
- Development
    - Format Python code and clean up imports
    - Run static type checking
    - Count lines of code
- Remote
    - Sync files between a server and a local machine
    - Connect remotely to a Python debugger
    - Automatically shut down cloud instances after task completion

## Dependencies
### Core Dependencies
The following dependencies are necessary for the core functionality of ML.

- Python >= 3.8 (packages are defined in `config/env.yml`)
- direnv
- mamba | conda
- tmux

### Optional Dependencies
The following dependencies are only necessary for additional features of ML.

- cloc
- htop
- rsync
- sensors
- watch

## Setup
1. In the project root, run `source ./helper/start.sh`.
1. Run `ML setup` if core dependencies are installed, or `ML setup --install` to install all dependencies.
1. Run `direnv allow` to enable automatic environment loading.

## Usage
Run `ML` or `ML --help` to see available commands.

## Notes
Results for finished experiments do not include model weights, but can be provided on request. Reach out to me: [m10000n@proton.me](mailto:m10000n@proton.me)
