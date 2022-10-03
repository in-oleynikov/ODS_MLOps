# -*- coding: utf-8 -*-
import os

from datasets import load_dataset

from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
import evaluate
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch
from transformers import pipeline
from process_dataset import make_dataloader
from src.data.make_dataset import make_dataloader

BATCH_SIZE = os.getenv("BATCH_SIZE")
train_dataloader = make_dataloader(BATCH_SIZE, "train")
eval_dataloader = make_dataloader(BATCH_SIZE, "validation")

class Trainer():
    def __int__(self, model, accelerator, optimizer, lr_scheduler, progress_bar):
        self.model = model
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.progress_bar = progress_bar

    def train_epoch(self):
        self.model.train()
        for batch in train_dataloader:
            outputs = self.model(**batch)
            loss = outputs.loss
            self.accelerator.backward(loss)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.progress_bar.update(1)

    def validate_epoch(self, ):
        self.model.eval()

        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = self.model(**batch)

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]

            # Necessary to pad predictions and labels for being gathered
            predictions = self.accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = self.accelerator.gather(predictions)
            labels_gathered = self.accelerator.gather(labels)

            true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
            metric.add_batch(predictions=true_predictions, references=true_labels)

        return metric

    def train_loop(self,):
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        progress_bar = tqdm(range(num_training_steps))
        for epoch in range(num_train_epochs):
            self.train_epoch()
            self.validate_epoch()

        results = metric.compute()
        print(
            f"epoch {epoch}:",
            {
                key: results[f"overall_{key}"]
                for key in ["precision", "recall", "f1", "accuracy"]
            },
        )

        #     # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir + f'/epoch_{epoch}', save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)