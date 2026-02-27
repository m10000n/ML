from typing import cast

from helper import file, path, statistic
from helper.helper_ import flatten, round_to_int, round_to_str
from helper.plot import compare
from helper.plot import experiment as plot_experiment
from helper.plot import general as plot_general
from helper.plot.color import GREY, QualColors
from helper.plot.main import Legend
from helper.print import print_end, print_start
from model.cross_validation import CrossValidation

CONFIDENCE_LEVEL = 0.95
SEED = 42
N_BOOTSTRAP_RESAMPLES = 5000
SAMPLE_TO_EXCLUDE = [4, 1740]

RESNET4D_PATH = path.cross_validation(model_name="ResNet4D")
INCEPTRON_PATH = path.cross_validation(model_name="Inceptron")
BRT_PATH = path.cross_validation(model_name="BrT")

CV_PATHS = [
    INCEPTRON_PATH / "inceptron__prep__00.0",
    RESNET4D_PATH / "resnet4d_152_bn__prep__00.0",
    BRT_PATH / "brt_medium_t4c2_prep__00.0",
    INCEPTRON_PATH / "inceptron__unp__01.0",
    RESNET4D_PATH / "resnet4d_101_bn__unp__01.0",
    BRT_PATH / "brt_small_t4c2_unp__01.0",
]

NAMES = ["Inceptron", "ResNet4D", "BrT", "Inceptron", "ResNet4D", "BrT"]

MUW_THESIS_DIR = path.project_root() / "muw_thesis"

PAGE_WIDTH = 6.3
PAGE_HEIGHT = 9.72


def main() -> None:
    print_start("Start evaluation MUW thesis.", mode="primary")

    exp_names = [f"{prep_name} - preprocessed" for prep_name in NAMES[0:3]] + [
        f"{unp_name} - unprocessed" for unp_name in NAMES[3:6]
    ]

    # load cross validations
    print_start("Start loading cross validations.")
    cvs = [CrossValidation.load(path) for path in CV_PATHS]
    print_end("Finished loading cross validations.")

    ids = []
    actual = []
    predicted = []

    # Exclude the missing unprocessed fMRI file from the preprocessed data.
    for cv in cvs[:3]:
        cv_ids = []
        cv_actual = []
        cv_predicted = []
        for i, fold in enumerate(cv.runs):
            fold_ids = fold.get_confusion_ids()
            fold_actual = fold.get_confusion_actual()
            fold_predicted = fold.get_confusion_predicted()

            if i == SAMPLE_TO_EXCLUDE[0]:
                fold_ids = fold_ids[: SAMPLE_TO_EXCLUDE[1]] + fold_ids[SAMPLE_TO_EXCLUDE[1] + 1 :]
                fold_actual = fold_actual[: SAMPLE_TO_EXCLUDE[1]] + fold_actual[SAMPLE_TO_EXCLUDE[1] + 1 :]
                fold_predicted = fold_predicted[: SAMPLE_TO_EXCLUDE[1]] + fold_predicted[SAMPLE_TO_EXCLUDE[1] + 1 :]

            cv_ids.append(fold_ids)
            cv_actual.append(fold_actual)
            cv_predicted.append(fold_predicted)

        ids.append(cv_ids)
        actual.append(cv_actual)
        predicted.append(cv_predicted)

    for cv in cvs[3:]:
        cv_ids = []
        cv_actual = []
        cv_predicted = []

        for fold in cv.runs:
            cv_ids.append(fold.get_confusion_ids())
            cv_actual.append(fold.get_confusion_actual())
            cv_predicted.append(fold.get_confusion_predicted())

        ids.append(cv_ids)
        actual.append(cv_actual)
        predicted.append(cv_predicted)

    if not all(ids_ == ids[0] for ids_ in ids[1:]):
        raise ValueError("IDs do not match across experiments.")

    ids_ = ids[0]
    ids_flat = flatten(ids_)

    if not all(actual_ == actual[0] for actual_ in actual[1:]):
        raise ValueError("Actual labels do not match across experiments.")

    actual_ = actual[0]
    actual_flat = flatten(actual_)

    for cv_predicted in predicted[1:]:
        for i, fold_predicted in enumerate(cv_predicted):
            if not len(fold_predicted) == len(predicted[0][i]):
                raise ValueError("The number of predicted labels does not match across experiment folds.")

    predicted_flat = [flatten(cv_predicted) for cv_predicted in predicted]

    if not len(ids_flat) == len(actual_flat) == len(predicted_flat[0]):
        raise ValueError("The number of ids, actual labels and predicted labels does not match across experiments.")

    class_names = []
    epochs = []
    train_durations = []
    warmup_loss = []
    train_loss = []
    val_loss = []
    test_loss = []
    val_acc = []
    test_acc = []

    for cv in cvs:
        class_names.append(cv.class_names)
        cv_epochs = []
        cv_train_durations = []
        cv_warmup_loss = []
        cv_train_loss = []
        cv_val_loss = []
        cv_test_loss = []
        cv_val_acc = []
        cv_test_acc = []

        for run in cv.runs:
            cv_epochs.append(run.get_total_epochs())
            cv_train_durations.append(round_to_int(x=run.get_duration("train").total_seconds()) / 3600)
            cv_warmup_loss.append(run.get_warmup_loss_as_epochs())
            cv_train_loss.append(cast(list[float], run.get_loss("train")))
            cv_val_loss.append(cast(list[float], run.get_loss("val")))
            cv_test_loss.append(cast(float, run.get_loss("test")))
            cv_val_acc.append(cast(list[float], run.get_accuracy("val")))
            cv_test_acc.append(cast(float, run.get_accuracy("test")))

        epochs.append(cv_epochs)
        train_durations.append(cv_train_durations)
        warmup_loss.append(cv_warmup_loss)
        train_loss.append(cv_train_loss)
        val_loss.append(cv_val_loss)
        test_loss.append(cv_test_loss)
        val_acc.append(cv_val_acc)
        test_acc.append(cv_test_acc)

    if not all(class_names_ == class_names[0] for class_names_ in class_names[1:]):
        raise ValueError("Cross validations have different class names.")

    class_names_ = class_names[0]
    n_classes = len(class_names_)
    support = statistic.get_support(actual=actual_flat, n_classes=n_classes)

    confusion_matrices = []
    macro_f1_scores_bs = []
    accuracies_bs = []
    balanced_accuracies_bs = []
    precisions_bs = []
    recalls_bs = []
    f1_scores = []

    for predicted_ in predicted_flat:
        confusion_matrices.append(
            statistic.create_confusion_matrix(actual=actual_flat, predicted=predicted_, n_classes=n_classes)
        )

        macro_f1_scores_bs.append(
            cast(
                tuple[float, float, float],
                statistic.bca_ci(
                    ids=ids_flat,
                    actual=actual_flat,
                    predicted=predicted_,
                    n_classes=n_classes,
                    metric="macro_f1_score",
                    n_resamples=N_BOOTSTRAP_RESAMPLES,
                    confidence_level=CONFIDENCE_LEVEL,
                    seed=SEED,
                ),
            )
        )

        accuracies_bs.append(
            statistic.bca_ci(
                ids=ids_flat,
                actual=actual_flat,
                predicted=predicted_,
                n_classes=n_classes,
                metric="accuracy",
                n_resamples=N_BOOTSTRAP_RESAMPLES,
                confidence_level=CONFIDENCE_LEVEL,
                seed=SEED,
            )
        )

        balanced_accuracies_bs.append(
            statistic.bca_ci(
                ids=ids_flat,
                actual=actual_flat,
                predicted=predicted_,
                n_classes=n_classes,
                metric="balanced_accuracy",
                n_resamples=N_BOOTSTRAP_RESAMPLES,
                confidence_level=CONFIDENCE_LEVEL,
                seed=SEED,
            )
        )

        precisions_bs.append(
            statistic.bca_ci(
                ids=ids_flat,
                actual=actual_flat,
                predicted=predicted_,
                n_classes=n_classes,
                metric="precision",
                n_resamples=N_BOOTSTRAP_RESAMPLES,
                confidence_level=CONFIDENCE_LEVEL,
                seed=SEED,
            )
        )

        recalls_bs.append(
            statistic.bca_ci(
                ids=ids_flat,
                actual=actual_flat,
                predicted=predicted_,
                n_classes=n_classes,
                metric="recall",
                n_resamples=N_BOOTSTRAP_RESAMPLES,
                confidence_level=CONFIDENCE_LEVEL,
                seed=SEED,
            )
        )

        f1_scores.append(
            statistic.bca_ci(
                ids=ids_flat,
                actual=actual_flat,
                predicted=predicted_,
                n_classes=n_classes,
                metric="f1_score",
                n_resamples=N_BOOTSTRAP_RESAMPLES,
                confidence_level=CONFIDENCE_LEVEL,
                seed=SEED,
            )
        )

    macro_f1_scores_fold = []
    accuracies_fold = []
    balanced_accuracies_fold = []
    precisions_fold = []
    recalls_fold = []
    f1_fold = []

    for cv_predicted in predicted:
        cv_macro_f1_scores = []
        cv_accuracies = []
        cv_balanced_accuracies = []
        cv_precisions = []
        cv_recalls = []
        cv_f1_scores = []

        for fold_actual, fold_predicted in zip(actual_, cv_predicted):
            cv_macro_f1_scores.append(
                statistic.calculate_macro_f1_score(actual=fold_actual, predicted=fold_predicted, n_classes=n_classes)
            )
            cv_accuracies.append(
                statistic.calculate_accuracy(actual=fold_actual, predicted=fold_predicted, n_classes=n_classes)
            )
            cv_balanced_accuracies.append(
                statistic.calculate_balanced_accuracy(actual=fold_actual, predicted=fold_predicted, n_classes=n_classes)
            )
            cv_precisions.append(
                statistic.calculate_precision(actual=fold_actual, predicted=fold_predicted, n_classes=n_classes)
            )
            cv_recalls.append(
                statistic.calculate_recall(actual=fold_actual, predicted=fold_predicted, n_classes=n_classes)
            )
            cv_f1_scores.append(
                statistic.calculate_f1_score(actual=fold_actual, predicted=fold_predicted, n_classes=n_classes)
            )

        macro_f1_scores_fold.append(
            (statistic.calculate_mean(cv_macro_f1_scores), statistic.calculate_std(cv_macro_f1_scores))
        )
        accuracies_fold.append((statistic.calculate_mean(cv_accuracies), statistic.calculate_std(cv_accuracies)))
        balanced_accuracies_fold.append(
            (statistic.calculate_mean(cv_balanced_accuracies), statistic.calculate_std(cv_balanced_accuracies))
        )

        precision_mean_std_ = []
        recall_mean_std_ = []
        f1_score_mean_std_ = []

        for i in range(n_classes):
            class_precision = [precision[i] for precision in cv_precisions]
            class_recall = [recall[i] for recall in cv_recalls]
            class_f1_score = [f1_score[i] for f1_score in cv_f1_scores]
            precision_mean_std_.append(
                (statistic.calculate_mean(class_precision), statistic.calculate_std(class_precision))
            )
            recall_mean_std_.append((statistic.calculate_mean(class_recall), statistic.calculate_std(class_recall)))
            f1_score_mean_std_.append(
                (statistic.calculate_mean(class_f1_score), statistic.calculate_std(class_f1_score))
            )

        precisions_fold.append(precision_mean_std_)
        recalls_fold.append(recall_mean_std_)
        f1_fold.append(f1_score_mean_std_)

    macro_f1_diffs = []

    for predicted_prep, predicted_unp in zip(predicted_flat[:3], predicted_flat[3:]):
        macro_f1_diffs.append(
            cast(
                tuple[float, float, float],
                statistic.bca_ci_diff(
                    ids=ids_flat,
                    actual=actual_flat,
                    predicted_a=predicted_prep,
                    predicted_b=predicted_unp,
                    n_classes=n_classes,
                    metric="macro_f1_score",
                    n_resamples=N_BOOTSTRAP_RESAMPLES,
                    confidence_level=CONFIDENCE_LEVEL,
                    seed=SEED,
                ),
            ),
        )

    correlation_pairs = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (0, 3), (1, 4), (2, 5)]
    pearson = []
    spearman = []

    for pair in correlation_pairs:
        pearson.append(
            statistic.error_pattern_corr(
                cm1=confusion_matrices[pair[0]],
                cm2=confusion_matrices[pair[1]],
                method="pearson",
                exclude_diagonal=True,
                normalize="row",
            )[0]
        )
        spearman.append(
            statistic.error_pattern_corr(
                cm1=confusion_matrices[pair[0]],
                cm2=confusion_matrices[pair[1]],
                method="spearman",
                exclude_diagonal=True,
                normalize="row",
            )[0]
        )

    tail_instabilities_train_loss = []
    tail_instabilities_val_loss = []

    for train_loss_, val_loss_ in zip(train_loss, val_loss):
        train_instability = []
        val_instability = []

        for tl, vl in zip(train_loss_, val_loss_):
            train_instability.append(statistic.tail_instability(values=tl, tail_fraction=2 / 3, relative=True))
            val_instability.append(statistic.tail_instability(values=vl, tail_fraction=2 / 3, relative=True))

        tail_instabilities_train_loss.append(
            (statistic.calculate_mean(values=train_instability), statistic.calculate_std(values=train_instability))
        )
        tail_instabilities_val_loss.append(
            (statistic.calculate_mean(values=val_instability), statistic.calculate_std(values=val_instability))
        )

    # create text file
    print_start("Start creating text file.")

    result = ["Epochs:"]
    longest_name = max(len(name) for name in exp_names)
    exp_names_just = [f"{name}:".ljust(longest_name + 1) for name in exp_names]

    for epoch, name in zip(epochs, exp_names_just):
        result.append(
            f"\t{name} {", ".join(str(epoch_) for epoch_ in epoch)} (sum: {sum(epoch)}, "
            f"mean: {round_to_str(x=statistic.calculate_mean(values=[float(epoch_) for epoch_ in epoch]), digits=0)}, "
            f"std: {round_to_str(x=statistic.calculate_std(values=[float(epoch_) for epoch_ in epoch]), digits=0)})"
        )

    result.append("")
    result.append("Train Duration (hours):")
    for train_duration, name in zip(train_durations, exp_names_just):
        train_duration_str = [f"{round_to_str(x=duration, digits=2)}" for duration in train_duration]
        result.append(
            f"\t{name} {", ".join(train_duration_str)} (sum: {round_to_str(x=sum(train_duration), digits=2)}, "
            f"mean: {round_to_str(x=statistic.calculate_mean(values=train_duration), digits=2)}, "
            f"std: {round_to_str(x=statistic.calculate_std(values=train_duration), digits=2)})"
        )

    result.append("")
    result.append("BS Macro F1 Score:")
    for macro_f1_score_bs, name in zip(macro_f1_scores_bs, exp_names_just):
        point, low, high = cast(tuple[float, float, float], macro_f1_score_bs)
        result.append(
            f"\t{name} {round_to_str(x=point, digits=3)} [{round_to_str(x=low, digits=3)}–{round_to_str(x=high, digits=3)}]"
        )

    result.append("")
    result.append("BS Accuracy:")
    for accuracy_bs, name in zip(accuracies_bs, exp_names_just):
        point, low, high = cast(tuple[float, float, float], accuracy_bs)
        result.append(
            f"\t{name} {round_to_str(x=point, digits=3)} [{round_to_str(x=low, digits=3)}–{round_to_str(x=high, digits=3)}]"
        )

    result.append("")
    result.append("BS Balanced Accuracy:")
    for balanced_accuracy_bs, name in zip(balanced_accuracies_bs, exp_names_just):
        point, low, high = cast(tuple[float, float, float], balanced_accuracy_bs)
        result.append(
            f"\t{name} {round_to_str(x=point, digits=3)} [{round_to_str(x=low, digits=3)}–{round_to_str(x=high, digits=3)}]"
        )

    result.append("")
    result.append("Fold Macro F1 Score (mean, std):")
    for macro_f1_score_fold, name in zip(macro_f1_scores_fold, exp_names_just):
        result.append(
            f"\t{name} {round_to_str(x=macro_f1_score_fold[0], digits=3)} ± {round_to_str(x=macro_f1_score_fold[1], digits=3)}"
        )

    result.append("")
    result.append("Fold Accuracy (mean, std):")
    for accuracy_fold, name in zip(accuracies_fold, exp_names_just):
        result.append(
            f"\t{name} {round_to_str(x=accuracy_fold[0], digits=3)} ± {round_to_str(x=accuracy_fold[1], digits=3)}"
        )

    result.append("")
    result.append("Fold Balanced Accuracy (mean, std):")
    for balanced_accuracy_fold, name in zip(balanced_accuracies_fold, exp_names_just):
        result.append(
            f"\t{name} {round_to_str(x=balanced_accuracy_fold[0], digits=3)} ± {round_to_str(x=balanced_accuracy_fold[1], digits=3)}"
        )

    result.append("")
    result.append(f"Class Names: {", ".join(class_names_)}")
    result.append(f"Support: {", ".join(str(support_) for support_ in support)}")

    result.append("")
    result.append("BS Precision:")
    for precision_bs, name in zip(precisions_bs, exp_names_just):
        points, lows, highs = cast(tuple[list[float], list[float], list[float]], precision_bs)
        precision_point_ci_str = [
            f"{round_to_str(x=point, digits=3)} [{round_to_str(x=low, digits=3)}–{round_to_str(x=high, digits=3)}]"
            for point, low, high in zip(points, lows, highs)
        ]
        result.append(f"\t{name} {", ".join(precision_point_ci_str)}")

    result.append("")
    result.append("BS Recall:")
    for recall_bs, name in zip(recalls_bs, exp_names_just):
        points, lows, highs = cast(tuple[list[float], list[float], list[float]], recall_bs)
        recall_point_ci_str = [
            f"{round_to_str(x=point, digits=3)} [{round_to_str(x=low, digits=3)}–{round_to_str(x=high, digits=3)}]"
            for point, low, high in zip(points, lows, highs)
        ]
        result.append(f"\t{name} {", ".join(recall_point_ci_str)}")

    result.append("")
    result.append("BS F1 Score:")
    for f1_score_bs, name in zip(f1_scores, exp_names_just):
        points, lows, highs = cast(tuple[list[float], list[float], list[float]], f1_score_bs)
        f1_score_point_ci_str = [
            f"{round_to_str(x=point, digits=3)} [{round_to_str(x=low, digits=3)}–{round_to_str(x=high, digits=3)}]"
            for point, low, high in zip(points, lows, highs)
        ]
        result.append(f"\t{name} {", ".join(f1_score_point_ci_str)}")

    result.append("")
    result.append("Fold Precision (mean, std):")
    for precision_fold, name in zip(precisions_fold, exp_names_just):
        precision_mean_std_str = [
            f"{round_to_str(x=precision_mean_std[0], digits=3)} ± {round_to_str(x=precision_mean_std[1], digits=3)}"
            for precision_mean_std in precision_fold
        ]
        result.append(f"\t{name} {", ".join(precision_mean_std_str)}")

    result.append("")
    result.append("Fold Recall (mean, std):")
    for recall_fold, name in zip(recalls_fold, exp_names_just):
        recall_mean_std_str = [
            f"{round_to_str(x=recall_mean_std[0], digits=3)} ± {round_to_str(x=recall_mean_std[1], digits=3)}"
            for recall_mean_std in recall_fold
        ]
        result.append(f"\t{name} {", ".join(recall_mean_std_str)}")

    result.append("")
    result.append("Fold F1 Score (mean, std):")
    for f1_score_fold, name in zip(f1_fold, exp_names_just):
        f1_score_mean_std_str = [
            f"{round_to_str(x=f1_score_mean_std[0], digits=3)} ± {round_to_str(x=f1_score_mean_std[1], digits=3)}"
            for f1_score_mean_std in f1_score_fold
        ]
        result.append(f"\t{name} {", ".join(f1_score_mean_std_str)}")

    result.append("")
    result.append("Macro F1 Score Diff:")
    for macro_f1_diff, name in zip(macro_f1_diffs, NAMES[:3]):
        point, low, high = cast(tuple[float, float, float], macro_f1_diff)
        result.append(
            f"\t{name} {round_to_str(x=point, digits=3)} [{round_to_str(x=low, digits=3)}–{round_to_str(x=high, digits=3)}]"
        )

    result.append("")
    result.append("Error Correlation (Pearson):")
    result.append("\tprep:")
    for corr, pair in zip(pearson[0:3], correlation_pairs[0:3]):
        result.append(f"\t\t{NAMES[pair[0]]} - {NAMES[pair[1]]}: {round_to_str(x=corr, digits=3)}")
    result.append("\tunp:")
    for corr, pair in zip(pearson[3:6], correlation_pairs[3:6]):
        result.append(f"\t\t{NAMES[pair[0]]} - {NAMES[pair[1]]}: {round_to_str(x=corr, digits=3)}")
    result.append("\tprep - unp:")
    for corr, pair in zip(pearson[6:9], correlation_pairs[6:9]):
        result.append(f"\t\t{NAMES[pair[0]]} - {NAMES[pair[1]]}: {round_to_str(x=corr, digits=3)}")

    result.append("")
    result.append("Error Correlation (Spearman):")
    result.append("\tprep:")
    for corr, pair in zip(spearman[0:3], correlation_pairs[0:3]):
        result.append(f"\t\t{NAMES[pair[0]]} - {NAMES[pair[1]]}: {round_to_str(x=corr, digits=3)}")
    result.append("\tunp:")
    for corr, pair in zip(spearman[3:6], correlation_pairs[3:6]):
        result.append(f"\t\t{NAMES[pair[0]]} - {NAMES[pair[1]]}: {round_to_str(x=corr, digits=3)}")
    result.append("\tprep - unp:")
    for corr, pair in zip(spearman[6:9], correlation_pairs[6:9]):
        result.append(f"\t\t{NAMES[pair[0]]} - {NAMES[pair[1]]}: {round_to_str(x=corr, digits=3)}")

    result.append("")
    result.append("Relative Tail Instability (Train Loss):")
    for tail_instability, name in zip(tail_instabilities_train_loss, exp_names_just):
        result.append(
            f"\t{name} {round_to_str(x=tail_instability[0], digits=4)} ± {round_to_str(x=tail_instability[1], digits=4)}"
        )
    result.append("")
    result.append("Relative Tail Instability (Val Loss):")
    for tail_instability, name in zip(tail_instabilities_val_loss, exp_names_just):
        result.append(
            f"\t{name} {round_to_str(x=tail_instability[0], digits=4)} ± {round_to_str(x=tail_instability[1], digits=4)}"
        )

    file.write_lines(path=MUW_THESIS_DIR / "results.txt", lines=result, overwrite=True, lock=True)
    print_end("Finished creating text file.")

    # create plots
    print_start("Start creating plots.")
    names_fontsize = 8
    names_rotation = 15
    y_label_fontsize = 10
    y_ticks_fontsize = 8
    legend_fontsize = 7
    spine_width = 0.6
    tick_length = 2.5
    capsize = 4.5
    capthick = 1.1  # 1.5
    elinewidth = 1.1  # 1.5
    marker_diameter = 2.2  # 3.0

    colors = QualColors().get_n(n=2)
    colors_prep_unp = tuple(colors[0:2])
    colors_folds = QualColors(order=[0, 3, 4, 5, 6])

    ## training duration
    legend_duration = Legend(
        handle_type="round",
        handle_colors=colors_folds,
        labels=[f"fold {i}" for i in range(1, 6)],
        loc="upper left",
        fontsize=legend_fontsize,
        n_col=1,
        bbox_to_anchor=(1.02, 1),
        spine_width=spine_width,
    )

    plot_general.plot_dots_groups(
        dir_=MUW_THESIS_DIR,
        file_name="train_duration.png",
        names=NAMES[0:3],
        y=[train_durations[0:3]],
        marker_diameter=4.0,
        colors=colors_folds.get_n(5),
        colors_are_groups=False,
        fig_size=(PAGE_WIDTH * 0.37, PAGE_HEIGHT * 0.4),
        names_fontsize=names_fontsize,
        names_rotation=names_rotation,
        y_label="Training duration (hours)",
        y_label_fontsize=y_label_fontsize,
        y_ticks_fontsize=y_ticks_fontsize,
        y_lim=(0, max(flatten(train_durations[0:3])) * 1.05),
        spine_width=spine_width,
        tick_length=tick_length,
        grid=True,
        legend=legend_duration,
    )

    ## macro-averaged F1 score
    plot_general.plot_ci_groups(
        dir_=MUW_THESIS_DIR,
        file_name="ma_f1_score.png",
        names=NAMES,
        point=[
            [macro_f1_score[0] for macro_f1_score in macro_f1_scores_bs[:3]],
            [macro_f1_score[0] for macro_f1_score in macro_f1_scores_bs[3:]],
        ],
        ci_lower=[
            [macro_f1_score[1] for macro_f1_score in macro_f1_scores_bs[:3]],
            [macro_f1_score[1] for macro_f1_score in macro_f1_scores_bs[3:]],
        ],
        ci_upper=[
            [macro_f1_score[2] for macro_f1_score in macro_f1_scores_bs[:3]],
            [macro_f1_score[2] for macro_f1_score in macro_f1_scores_bs[3:]],
        ],
        marker_diameter=marker_diameter,
        capsize=capsize,
        capthick=capthick,
        elinewidth=elinewidth,
        colors=colors_prep_unp,
        fig_size=(PAGE_WIDTH * 0.5, PAGE_HEIGHT * 0.5),
        names_fontsize=names_fontsize,
        names_rotation=names_rotation,
        y_label="Macro-averaged F1 score",
        y_label_fontsize=y_label_fontsize,
        y_ticks_fontsize=y_ticks_fontsize,
        y_lim=(0.7, 1.0),
        last_y_label_inside=True,
        spine_width=spine_width,
        tick_length=tick_length,
        grid=True,
    )

    ## macro-averaged F1 score difference
    macro_f1_diff_points = [macro_f1_diff[0] for macro_f1_diff in macro_f1_diffs]
    macro_f1_diff_lows = [macro_f1_diff[1] for macro_f1_diff in macro_f1_diffs]
    macro_f1_diff_highs = [macro_f1_diff[2] for macro_f1_diff in macro_f1_diffs]

    plot_general.plot_ci_groups(
        dir_=MUW_THESIS_DIR,
        file_name="ma_f1_score_diff.png",
        names=NAMES[:3],
        point=[macro_f1_diff_points],
        ci_lower=[macro_f1_diff_lows],
        ci_upper=[macro_f1_diff_highs],
        marker_diameter=marker_diameter,
        capsize=capsize,
        capthick=capthick,
        elinewidth=elinewidth,
        colors=(GREY,),
        fig_size=(PAGE_WIDTH * 0.4, PAGE_HEIGHT * 0.5),
        names_fontsize=names_fontsize,
        names_rotation=names_rotation,
        y_label="Δ Macro-averaged F1 score",
        y_label_fontsize=y_label_fontsize,
        y_ticks_fontsize=y_ticks_fontsize,
        y_lim=(0.0, max(macro_f1_diff_highs) + 0.01),
        spine_width=spine_width,
        tick_length=tick_length,
        grid=True,
    )

    ## confusion matrix
    confusion_dir = MUW_THESIS_DIR / "confusion"
    for predicted_, exp_name in zip(predicted_flat, exp_names):
        exp_name = exp_name.replace("BrT", "Brain Transformer")
        plot_experiment.plot_confusion_matrix(
            exp_name=exp_name.replace(" ", ""),
            dir_=confusion_dir,
            actual=actual_flat,
            predicted=predicted_,
            class_names=class_names_,
            title_text=exp_name,
        )

    plot_general.plot_color_bar(dir_=confusion_dir, fig_size=(0.75, PAGE_WIDTH * 1.5))

    ## log loss
    loss_dir = MUW_THESIS_DIR / "loss"

    loss_prep = flatten(warmup_loss[0:3] + train_loss[0:3] + val_loss[0:3])
    loss_unp = flatten(warmup_loss[3:6] + train_loss[3:6] + val_loss[3:6])
    y_lim = [(min(loss_prep) * 0.95, max(loss_prep) * 1.05)] * 3 + [(min(loss_unp) * 0.95, max(loss_unp) * 1.05)] * 3

    for warmup_loss_, train_loss_, val_loss_, test_loss_, exp_name, y_lim_ in zip(
        warmup_loss, train_loss, val_loss, test_loss, exp_names, y_lim
    ):
        exp_name = exp_name.replace("BrT", "Brain Transformer")
        compare.plot_loss_comparison(
            comp_name=exp_name.replace(" ", ""),
            dir_=loss_dir,
            exp_names=[f"fold {i}" for i in range(1, 6)],
            train_loss=train_loss_,
            val_loss=val_loss_,
            test_loss=test_loss_,
            warmup_loss=warmup_loss_,  # type: ignore [arg-type]
            logarithmic=True,
            colors=colors_folds,
            title_text=exp_name,
            y_lim=y_lim_,
            legend_loc="outside",
            n_col=4,
        )

    ## accuracy
    acc_dir = MUW_THESIS_DIR / "accuracy"
    for val_acc_, test_acc_, exp_name in zip(val_acc, test_acc, exp_names):
        exp_name = exp_name.replace("BrT", "Brain Transformer")
        compare.plot_accuracy_comparison(
            comp_name=exp_name.replace(" ", ""),
            dir_=acc_dir,
            exp_names=[f"fold {i}" for i in range(1, 6)],
            val_acc=val_acc_,
            test_acc=test_acc_,
            colors=colors_folds,
            title_text=exp_name,
            legend_loc="outside",
            n_col=4,
        )

    print_end("Finished creating plots.")

    print_end("Finished evaluation MUW thesis.", mode="primary")
