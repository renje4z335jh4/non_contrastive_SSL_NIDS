"""Functions for computing metrics of ML models
"""
from __future__ import annotations
import typing
import sklearn.metrics as sk_metrics
import numpy as np
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def average_results(results: typing.Dict[str, typing.List[typing.List[float]]]) -> typing.Dict[str, np.ndarray]:
    """Calculate Means and Stds of metrics for different runs

    Parameters
    ----------
    results : Dict[str, List[List[float]]]
        Dictionary of metrics for different runs and over the number of epochs (list of list)

    Returns
    -------
    Dict[str, np.ndarray]
        Means and Stds for the different runs over the number of epochs
    """

    final_results = {}

    for metric_name, metric_values in results.items():
        final_results[metric_name] = {
            'mean': np.mean(metric_values, axis=0).tolist(),
            'std': np.std(metric_values, axis=0).tolist()
        }

    return final_results

def compute_metrics(
    y_true: np.ndarray,
    pred_scores: np.ndarray,
    pos_label: int = 1,
    number_percentiles: int = 100,
    estimation_ratio: float = 0.2,
) -> typing.Dict[str, typing.Any]:
    """Computes metrics for given prediction scores and ground truth values.
    For metrics computing scores with a threshold (e.g. accuracy, precision, recall, F1_Score)
    an optimal (according to F1-Score) threshold is computed using a subset of the given data.

    Parameters
    ----------
    y_true : np.ndarray
        Ground Truth
    pred_scores : np.ndarray
        Scores for each sample,
    pos_label : int, optional
        Label for the positive label (rare class -> anomaly), by default 1
    number_percentiles : int, optional
        Number of different percentiles in the neighborhood of (1-estimation_ratio)
        to find optimal threshold, by default 100
    estimation_ratio : float, optional
        Ratio of y_true, pred_scores used for estimating the optimal threshold, by default 0.2

    Returns
    -------
    typing.Dict[str, Any]
        Resulting scores of all metrics: Accuracy, Precision, Recall, F1_Score, AUROC, AUPR, Confusion Matrix
    """

    # generate indices for tensors (should have same shape)
    indices = np.arange(len(y_true))
    np.random.shuffle(indices)

    # split scores, ground truth in estimation and evaluation sets
    n_eval = int(len(y_true) * (1 - estimation_ratio))

    eval_true, eval_scores = y_true[indices[:n_eval]], pred_scores[indices[:n_eval]]
    estimation_true, estimation_scores = (
        y_true[indices[n_eval:]],
        pred_scores[indices[n_eval:]],
    )

    logging.info("%s %% of the generated scores are used as holdout, to estimate the optimal threshold for the test set.", "{:.1f}".format(estimation_ratio*100))
    optimal_threshold = _estimate_optimal_threshold(
        estimation_true, estimation_scores, pos_label, number_percentiles
    )
    logging.info("%f is calculated as the optimal threshold and is now used to compute the threshold-depending metrics for the remaining %s %% scores", optimal_threshold, "{:.1f}".format((1-estimation_ratio)*100))

    return _compute_metrics(
        eval_true,
        eval_scores,
        threshold=optimal_threshold,
        pos_label=pos_label,
        estimating=False,
    )


def _compute_metrics(
    y_true: np.ndarray,
    pred_scores: np.ndarray,
    threshold: float,
    pos_label: int = 1,
    estimating: bool = False,
) -> typing.Dict[str, float] | float:

    y_pred = (pred_scores >= threshold).astype(int)
    y_true = y_true.astype(int)

    if estimating:
        # if running in estimation mode for find optimal threshold
        return sk_metrics.f1_score(y_true, y_pred)

    # threshold depending metrics
    accuracy = sk_metrics.accuracy_score(y_true, y_pred)

    precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=pos_label
    )
    confusion_matrix = sk_metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])

    # threshold independent metrics
    aupr = sk_metrics.average_precision_score(y_true, pred_scores)
    auroc = sk_metrics.roc_auc_score(y_true, pred_scores)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f_score,
        "AUROC": auroc,
        "AUPR": aupr,
        "Confusion Matrix": confusion_matrix,
    }

    return metrics


def _estimate_optimal_threshold(
    y_true_subset: np.ndarray,
    pred_score_subset: np.ndarray,
    pos_label: int = 1,
    number_percentiles: int = 100,
) -> float:

    # ratio of normal data in subset
    ratio = 100 * sum(y_true_subset == 0) / len(y_true_subset)

    # the threshold that maximizes the F1-Score is typically
    # located in the neighborhood of the (1-ratio)th percentile of the scores
    # compute number_percentiles different percentiles in the neighborhood
    # of (1-ratio)
    q = np.linspace(max(ratio - 5, 0), min(ratio + 5, 100), number_percentiles)

    # compute q-th percentiles of the array elements
    thresholds = np.percentile(pred_score_subset, q)

    # compute F1-Score for each threshold
    f1_scores = [
        _compute_metrics(
            y_true_subset,
            pred_score_subset,
            threshold_i,
            pos_label=pos_label,
            estimating=True,
        )
        for threshold_i in thresholds
    ]

    # take threshold, which maximizes the F1-Score
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    return optimal_threshold
