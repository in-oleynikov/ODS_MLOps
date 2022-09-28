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

from src.data.make_dataset import make_dataloader

BATCH_SIZE = os.getenv("BATCH_SIZE")
train_dataloader = make_dataloader(BATCH_SIZE, "train")
eval_dataloader = make_dataloader(BATCH_SIZE, "validation")

class Trainer():
    def __int__(self, model):
        self.model = model
        accelerator
        optimizer
        lr_scheduler
        optimizer
        progress_bar
    def train_epoch(self):
        self.model.train()
        for batch in train_dataloader:
            outputs = self.model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    def validate_epoch():

        id2label = {str(i): label for i, label in enumerate(label_names)}
        label2id = {v: k for k, v in id2label.items()}
        model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            id2label=id2label,
            label2id=label2id,
        optimizer=AdamW(model.parameters(), lr=2e-5)
    )
        accelerator = Accelerator()
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader)
        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = num_train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        progress_bar = tqdm(range(num_training_steps))

        for epoch in range(num_train_epochs):
            # Training
            model.train()
            for batch in train_dataloader:
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            # Evaluation
            model.eval()
            for batch in eval_dataloader:
                with torch.no_grad():
                    outputs = model(**batch)

                predictions = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]

                # Necessary to pad predictions and labels for being gathered
                predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

                predictions_gathered = accelerator.gather(predictions)
                labels_gathered = accelerator.gather(labels)

                true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
                metric.add_batch(predictions=true_predictions, references=true_labels)

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
        #         repo.push_to_hub(
        #             commit_message=f"Training in progress epoch {epoch}", blocking=False
        #         )
    def train_loop():
        pass