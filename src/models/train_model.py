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

print(train_dataloader)