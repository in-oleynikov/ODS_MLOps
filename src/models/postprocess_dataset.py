# -*- coding: utf-8 -*-
import fire
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
import configparser


def postprocess(predictions: np.array, labels: np.array, labelnames):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[lbl] for lbl in label if lbl != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions
