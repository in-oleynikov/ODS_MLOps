# -*- coding: utf-8 -*-
import os

from accelerate import Accelerator
from torch.optim import AdamW
from transformers import AutoModelForTokenClassification
from src.models.trainer import Trainer
from src.models.process_dataset import make_dataloader

BATCH_SIZE = os.getenv("BATCH_SIZE")
MODEL_CHECKPOINT = os.getenv("MODEL")
LEARNING_RATE = os.getenv("LEARNING_RATE")


def train_model(input_path, output_path, model_checkpoint):
    train_dataloader = make_dataloader(input_path, MODEL_CHECKPOINT, BATCH_SIZE, "train")
    eval_dataloader = make_dataloader(input_path, MODEL_CHECKPOINT, BATCH_SIZE, "validation")

    id2label = {str(i): label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
        optimizer=AdamW(model.parameters(), lr=LEARNING_RATE)
    )

    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader)
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    trainer = Trainer()

if __name__ == '__main__':
    fire.Fire(train_model)
