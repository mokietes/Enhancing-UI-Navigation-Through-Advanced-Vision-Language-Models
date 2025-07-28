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
os.environ['WANDB_PROJECT'] = "Llama-3.2-11B-finetuned-waveUI-l2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
HF_TOKEN = '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'
wandb.init(project=os.environ['WANDB_PROJECT'])

# === Data Preprocessing ===
def convert_to_conversation(sample):
    bbox = sample.get("bbox", [0, 0, 0, 0])
    name = sample.get("name")
    ocr_label = sample.get("OCR")
    resolution = sample.get("resolution")
    description = sample.get("description")
    language = sample.get("language")
    platform = sample.get("platform")
    purpose = sample.get("purpose")
    expectation = sample.get("expectation")
    instructions = sample.get("instruction")

    global_instruction = (
        "You are given a user interface screenshot. Your task is to identify the target button or text element and return its bounding box in the format [x1, y1, x2, y2]. Do not provide any explanation—just the coordinates."
    )

    dynamic_parts = []
    if name: dynamic_parts.append(f"The element is named '{name}'.")
    if ocr_label: dynamic_parts.append(f"It contains the text label '{ocr_label}'.")
    if resolution: dynamic_parts.append(f"The image resolution is {resolution}.")
    if description: dynamic_parts.append(f"This element is used for {description}.")
    if language: dynamic_parts.append(f"It is presented in {language}.")
    if purpose: dynamic_parts.append(f"The purpose of this element is to {purpose}.")
    if expectation: dynamic_parts.append(f"It is expected to {expectation}.")
    if platform: dynamic_parts.append(f"This UI is part of the {platform} platform.")
    if instructions: dynamic_parts.append(f"Additional instruction context: '{instructions}'.")
    dynamic_parts.append("Return the bounding box coordinates in the format [x1, y1, x2, y2].")

    return {
        "input": global_instruction + " " + " ".join(dynamic_parts),
        "bbox": [float(x) for x in bbox]
    }

# === Load Dataset ===
dataset_path = "/Users/923676946/git-repos/Visual-Data-Mining-AI-Model/training/datasets/wave-ui/data"
dataset = load_dataset("parquet", data_files={
    "train": os.path.join(dataset_path, "train-*.parquet"),
    "validation": os.path.join(dataset_path, "validation-*.parquet"),
})
train_dataset = dataset["train"].map(convert_to_conversation)
val_dataset = dataset["validation"].map(convert_to_conversation)

# === Load Model and Processor ===
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
)
processor = AutoProcessor.from_pretrained("unsloth/Llama-3.2-11B-Vision-Instruct", trust_remote_code=True)
model.gradient_checkpointing_enable()
model.config.output_hidden_states = True  # ✅ ensure hidden states will be returned

# === Add Regression Head ===
class BBoxRegressionHead(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
    def forward(self, last_hidden_state):
        return self.mlp(last_hidden_state[:, 0, :])  # use CLS token

model.regression_head = BBoxRegressionHead(model.config.hidden_size).to(model.device)

# === Tokenization ===
def tokenize(example):
    tokens = processor(
        text=example["input"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    return {
        "input_ids": tokens.input_ids[0],
        "attention_mask": tokens.attention_mask[0],
        "bbox": example["bbox"]
    }

def is_valid_bbox(example):
    bbox = example.get("bbox")
    return isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox)

train_dataset = train_dataset.filter(is_valid_bbox).map(tokenize)
val_dataset = val_dataset.filter(is_valid_bbox).map(tokenize)

