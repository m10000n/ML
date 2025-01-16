from config.ssh import USER

EXCLUDED_FILES = ["**/.DS_Store", "**/__pycache__"]

REMOTE_BASE_DIR = f"/home/{USER}/exp"

INCLUDE_DATASET = False

if not REMOTE_BASE_DIR:
    raise ValueError("You have not specified the remote base directory.")
