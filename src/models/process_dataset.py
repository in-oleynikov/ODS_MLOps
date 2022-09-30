# -*- coding: utf-8 -*-
import fire
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader


def make_dataloader(input_path, model_checkpoint, batch_size, data_type="train"):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    tokenized_datasets = load_from_disk(input_path)

    dataloader = DataLoader(
        tokenized_datasets[data_type],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    return dataloader


# if __name__ == "__main__":
#     fire.Fire(make_dataloader)
