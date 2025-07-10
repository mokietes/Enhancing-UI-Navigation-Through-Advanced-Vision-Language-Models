# === Import Libraries ===
import os
import torch
import wandb
import ast
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    Trainer, 
    TrainingArguments,
    TrainerCallback,
    AutoProcessor, 
    AutoModelForCausalLM
)
from huggingface_hub import HfApi
import torch.nn.functional as F
from torch.autograd import Variable

# === Environment Setup ===
os.environ['WANDB_PROJECT'] = "Llama-3.2-11B-finetuned-SmoothL1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
HF_TOKEN = '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'
wandb.init(project=os.environ['WANDB_PROJECT'])

# === Data Preprocessing Function ===
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
        "label": str(bbox),
    }

# === Load Dataset ===
dataset_path = "/Users/923676946/git-repos/Visual-Data-Mining-AI-Model/training/datasets/wave-ui/data"
dataset = load_dataset("parquet", data_files={
    "train": os.path.join(dataset_path, "train-*.parquet"),
    "validation": os.path.join(dataset_path, "validation-*.parquet"),
})

train_dataset = dataset["train"].select(range(100)).map(convert_to_conversation)
val_dataset = dataset["validation"].select(range(200)).map(convert_to_conversation)

# === Load Model and Processor ===
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
)
processor = AutoProcessor.from_pretrained("unsloth/Llama-3.2-11B-Vision-Instruct", trust_remote_code=True)

# === Enable gradient checkpointing to save memory ===
model.gradient_checkpointing_enable()

# === Tokenize Dataset ===
def tokenize(example):
    tokenized = processor(
        text=example["input"],
        text_target=example["label"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    return {
        "input_ids": tokenized.input_ids[0],
        "attention_mask": tokenized.attention_mask[0],
        "labels": tokenized.labels[0],
    }

train_dataset = train_dataset.map(tokenize, remove_columns=["input", "label"])
val_dataset = val_dataset.map(tokenize, remove_columns=["input", "label"])

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir="./outputs/SmoothL1",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    eval_steps=50,
    save_steps=100,
    logging_steps=10,
    save_total_limit=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    report_to="wandb",
    run_name="llama3-ui-bbox-SmoothL1",
    no_cuda=not torch.cuda.is_available(),
)
