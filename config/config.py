import argparse

MODEL = (
    "Wang_X_2020_Decoding_and_mapping_task_states_of_the_human_brain_via_deep_learning"
)

DATASET = "hcp_openacces"

if not MODEL:
    raise ValueError("You have not specified the current model.")

if not DATASET:
    raise ValueError("You have not specified the current dataset.")


if __name__ == "__main__":
    commands = {
        "model": MODEL,
        "dataset": DATASET,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=list(commands.keys()),
    )
    args = parser.parse_args()

    print(commands)
