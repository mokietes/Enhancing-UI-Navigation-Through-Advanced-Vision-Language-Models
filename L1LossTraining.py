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

# === Custom Trainer with Smooth L1 Loss Only ===
class SmoothL1LossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            labels=inputs["labels"].to(model.device),
        )

        pred_ids = torch.argmax(outputs.logits, dim=-1)
        decoded_preds = self.decode_bbox(pred_ids)
        decoded_labels = self.decode_bbox(inputs["labels"])

        smooth_l1 = F.smooth_l1_loss(decoded_preds, decoded_labels)
        loss = Variable(smooth_l1, requires_grad=True)
        wandb.log({"smooth_l1_loss": loss.item()})

        return loss if not return_outputs else (loss, outputs)

    def decode_bbox(self, token_ids):
        boxes = []
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        for ids in token_ids:
            text = processor.decode(ids, skip_special_tokens=True)
            try:
                box = ast.literal_eval(text)
                if isinstance(box, list) and len(box) == 4:
                    boxes.append(torch.tensor(box, dtype=torch.float32, device=self.model.device))
                else:
                    boxes.append(torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.model.device))
            except:
                boxes.append(torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.model.device))
        return torch.stack(boxes)

# === Clear memory before training ===
torch.cuda.empty_cache()

# === Trainer ===
trainer = SmoothL1LossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=None,  # avoids tokenizer deprecation warning
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

# Uncomment to save and push
# save_and_push_model(model, processor, "Llama-3.2-11B-finetuned-waveUI-SmoothL1", HF_TOKEN)

wandb.finish()
