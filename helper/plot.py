import matplotlib.pyplot as plt
import torch

from config.exp_file_names import (
    CONFUSION_PLOT_FILE_NAME,
    LOG_FILE_NAME,
    LOG_LOSS_ACCURACY_PLOT_FILE_NAME,
    LOSS_ACCURACY_PLOT_FILE_NAME,
)
from data.hcp_openacces.dataset import TaskDataset
from helper.log import Log
from helper.path import RESULT_PATH_A

EXP_NAME = "large"
CLASS_NAME = TaskDataset.TASKS


def loss_accuracy(exp_name, loss, accuracy, test_accuracy, log=False):
    loss = torch.tensor(loss)
    accuracy = torch.tensor(accuracy)
    test_accuracy = torch.tensor(test_accuracy)

    if log:
        loss = torch.log(loss)

    n_epochs = len(loss)

    fig, ax1 = plt.subplots(figsize=(8, 6))
    plt.title(f"{"Log " if log else ""}Loss and Accuracy - {exp_name}", fontsize=18)

    color_loss = "#1f77b4"
    min_loss_idx = torch.argmin(loss).item()
    min_loss = loss[min_loss_idx]
    ax1.scatter(
        min_loss_idx + 1, min_loss, color="black", label="Lowest Loss", zorder=1
    )
    ax1.plot(
        range(1, n_epochs + 1),
        loss,
        label="Loss",
        linewidth=2,
        color=color_loss,
        zorder=0,
    )
    ax1.set_xlabel("Epoch", fontsize=14)
    ax1.set_ylabel(
        f"Validation {"Log " if log else ""}Loss", fontsize=14, color=color_loss
    )
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.set_xlim(1, n_epochs)
    ax1.set_xticks(range(1, n_epochs + 1, n_epochs // 5))

    ax2 = ax1.twinx()
    color_accuracy = "#87CEEB"
    max_acc_idx = torch.argmax(accuracy).item()
    max_acc = accuracy[max_acc_idx]
    ax2.scatter(
        max_acc_idx + 1, max_acc, color="black", label="Highest Accuracy", zorder=1
    )
    ax2.plot(
        range(1, n_epochs + 1),
        accuracy,
        label="Accuracy",
        linewidth=2,
        color=color_accuracy,
        zorder=0,
    )
    ax2.set_ylabel("Validation Accuracy", fontsize=14, color=color_accuracy)
    ax2.tick_params(axis="y", labelcolor=color_accuracy)

    if log:
        min_loss = torch.exp(min_loss)
    plt.text(
        x=0.0,
        y=-0.12,
        s=(
            f"Lowest Validation Loss: {min_loss:.4f}\n"
            f"Highest Validation Accuracy: {max_acc:.4f}\n"
            f"Test Accuracy: {test_accuracy:.4f}"
        ),
        fontsize=12,
        color="black",
        ha="left",
        va="top",
        transform=ax1.transAxes,
    )

    fig.tight_layout()
    file_name_format = LOSS_ACCURACY_PLOT_FILE_NAME
    if log:
        file_name_format = LOG_LOSS_ACCURACY_PLOT_FILE_NAME
    file_name = file_name_format.format(exp_name=exp_name)
    path = RESULT_PATH_A / exp_name / file_name
    plt.savefig(path, dpi=300, bbox_inches="tight")


def confusion_matrix(exp_name, actual, predicted, class_names):
    actual = torch.tensor(actual)
    predicted = torch.tensor(predicted)

    n_classes = len(class_names)

    conf_matrix = torch.zeros(n_classes, n_classes)

    for actual_, predicted_ in zip(actual, predicted):
        conf_matrix[actual_, predicted_] += 1

    conf_matrix = conf_matrix / conf_matrix.sum(dim=1, keepdim=True)

    plt.figure(figsize=(8, 6))
    if exp_name:
        plt.title(label=f"Confusion - {exp_name}", fontsize=18)
    plt.imshow(X=conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = torch.arange(n_classes)
    plt.xticks(ticks=tick_marks, labels=class_names, rotation=45, fontsize=10)
    plt.yticks(ticks=tick_marks, labels=class_names, fontsize=10)

    threshold = conf_matrix.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            value = conf_matrix[i, j]
            color = "white" if value > threshold else "black"
            plt.text(
                x=j,
                y=i,
                s=format(value, ".2f"),
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )

    plt.xlabel(xlabel="Predicted Class", fontsize=14)
    plt.ylabel(ylabel="True Class", fontsize=14)

    plt.tight_layout()
    path = RESULT_PATH_A / exp_name / CONFUSION_PLOT_FILE_NAME.format(exp_name=exp_name)
    plt.savefig(path, dpi=300, bbox_inches="tight")


def plot_all(exp_name, log, class_names):
    confusion_matrix(
        exp_name=exp_name,
        actual=log.get_confusion_actual(),
        predicted=log.get_confusion_predicted(),
        class_names=class_names,
    )
    loss_accuracy(
        exp_name=exp_name,
        loss=log.get_loss_val(),
        accuracy=log.get_accuracy_val(),
        test_accuracy=log.get_accuracy_test(),
    )
    loss_accuracy(
        exp_name=exp_name,
        loss=log.get_loss_val(),
        accuracy=log.get_accuracy_val(),
        test_accuracy=log.get_accuracy_test(),
        log=True,
    )


if __name__ == "__main__":
    if not EXP_NAME:
        raise ValueError("You have not specified the name of the experiment.")
    if not CLASS_NAME:
        raise ValueError("You have not specified the names of the classes.")

    log_ = Log.from_json(
        RESULT_PATH_A / EXP_NAME / LOG_FILE_NAME.format(exp_name=EXP_NAME)
    )
    plot_all(exp_name=EXP_NAME, log=log_, class_names=CLASS_NAME)
