# -*- coding: utf-8 -*-

import numpy as np
from src.data.data_utils import label_names


def postprocess(predictions: np.array, labels: np.array):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [
        [label_names[lbl] for lbl in label if lbl != -100] for label in labels
    ]
    true_predictions = [
        [label_names[p] for (p, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions
