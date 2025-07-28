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

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir="./outputs/L2checkpoints",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    max_steps=-1,
    warmup_steps=200,
    logging_steps=10,
    save_steps=1000,
    eval_steps=500,
    save_total_limit=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,
    report_to="wandb",
    run_name="llama3-bbox-l2",
    no_cuda=not torch.cuda.is_available(),
    dataloader_num_workers=2,
    seed=42
)

#eval metric
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {
        "eval_loss": F.mse_loss(torch.tensor(predictions), torch.tensor(labels)).item()
    }
# === Custom Trainer ===
class DirectBBoxL2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1]
        pred_boxes = model.regression_head(last_hidden)

        try:
            true_boxes = torch.tensor(inputs["bbox"], dtype=torch.float32).to(model.device)
        except Exception as e:
            wandb.log({"malformed_bbox_text": str(inputs.get("bbox", "missing"))})
            true_boxes = torch.zeros(pred_boxes.shape, device=model.device)

        l2 = F.mse_loss(pred_boxes, true_boxes)
        wandb.log({"l2_loss": l2.item()})
        return (l2, outputs) if return_outputs else l2

# === Clear memory before training ===
torch.cuda.empty_cache()

# === Trainer ===
trainer = DirectBBoxL2Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=None,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# === Save Final Model ===
def save_and_push_model(model, processor, repo_id: str, token: str):
    try:
        model.save_pretrained(repo_id, safe_serialization=True)
        processor.save_pretrained(repo_id)
        model.push_to_hub(repo_id, token=token)
        processor.push_to_hub(repo_id, token=token)
        print("✅ Model pushed successfully")
    except Exception as e:
        print(f"❌ Failed to push model: {e}")

# Uncomment below to push to hub
# save_and_push_model(model, processor, "Llama-3.2-11B-finetuned-waveUI-L2", HF_TOKEN)

wandb.finish()