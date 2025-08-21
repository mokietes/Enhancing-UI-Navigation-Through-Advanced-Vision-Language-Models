#l2
# === Import Libraries ===
import os
import torch
import wandb
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    Trainer, 
    TrainingArguments,
    AutoProcessor, 
    AutoModelForCausalLM,
    default_data_collator
)
import torch.nn as nn
import torch.nn.functional as F


# === Environment Setup ===
os.environ['WANDB_PROJECT'] = "Llama-3.2-11B-finetuned-rico-CombinedLossTrainer"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
HF_TOKEN = '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'
wandb.init(project=os.environ['WANDB_PROJECT'])

def normalize_bbox(bbox, resolution):
    if not resolution or len(resolution) != 2:
        return [0.0, 0.0, 0.0, 0.0]
    width, height = resolution
    x1, y1, x2, y2 = bbox
    return [
        max(0.0, min(x1 / width, 1.0)),
        max(0.0, min(y1 / height, 1.0)),
        max(0.0, min(x2 / width, 1.0)),
        max(0.0, min(y2 / height, 1.0)),
    ]
