import matplotlib.pyplot as plt


def save_plot(model_name, loss, accuracy, folder, log=False):
    plt.figure()

    y_label = "loss"
    file_name = f"{folder}/{model_name}_plot.png"
    if log:
        y_label = "loss[log]"
        file_name = f"{folder}/{model_name}_plot_log.png"

    plt.subplots_adjust(top=0.85)
    plt.plot(list(range(len(loss))), loss)
    plt.title(model_name, fontsize=20, pad=20)
    plt.xlabel("epoch", fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.text(
        0.98,
        0.98,
        f"accuracy: {accuracy:.2f}%",
        fontsize=14,
        horizontalalignment="right",
        verticalalignment="top",
        transform=plt.gca().transAxes,
    )

    plt.savefig(file_name)
