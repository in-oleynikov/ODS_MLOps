# -*- coding: utf-8 -*-
import fire
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
import configparser

# model_checkpoint = config["DEFAULT"]["MODEL"]
config = configparser.ConfigParser()
config.read('../config')
model_checkpoint = "prajjwal1/bert-tiny"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


def make_dataloader(batch_size, data_type="train"):
    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets[data_type].column_names,
    )
    dataloader = DataLoader(
        tokenized_datasets[data_type],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    return dataloader


def postprocess(predictions: np.array, labels: np.array):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[lbl] for lbl in label if lbl != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions
