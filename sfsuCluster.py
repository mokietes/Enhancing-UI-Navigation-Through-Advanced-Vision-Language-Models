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


# === Data Preprocessing ===
def convert_to_conversation_rico(sample):
    bbox_dict = sample.get("target_bounding_box", {})
    prompt = sample.get("prompt", "")

    # Use normalized bounding box directly
    xmin = bbox_dict.get("xmin", 0.0)
    ymin = bbox_dict.get("ymin", 0.0)
    xmax = bbox_dict.get("xmax", 0.0)
    ymax = bbox_dict.get("ymax", 0.0)

    # Reorder to match [x1, y1, x2, y2] format
    norm_bbox = [xmin, ymin, xmax, ymax]

    global_instruction = (
        "You are given a user interface screenshot. Your task is to identify the target button or text element and return its bounding box in the format [x1, y1, x2, y2]. Do not provide any explanation—just the coordinates."
    )

    return {
        "input": f"{global_instruction} {prompt}",
        "bbox": norm_bbox,
    }


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
    
    norm_bbox = normalize_bbox(bbox, resolution)
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
        # "bbox": [float(x) for x in bbox]
        "bbox": norm_bbox
    }

# === Load Dataset ===

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
            nn.Linear(256, 4),
            nn.Sigmoid()
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
        "bbox": example["bbox"],
        "labels": example["bbox"],
    }

def is_valid_bbox(example):
    bbox = example.get("bbox")
    return isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox)

train_dataset = train_dataset.filter(is_valid_bbox).map(tokenize)
val_dataset = val_dataset.filter(is_valid_bbox).map(tokenize)

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir="./outputs/l1checkpoints",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    max_steps=-1,
    warmup_steps=200,
    logging_steps=10,
    save_steps=500,
    eval_steps=250,
    save_total_limit=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,
    report_to="wandb",
    run_name="llama3-bbox-L1",
    no_cuda=not torch.cuda.is_available(),
    dataloader_num_workers=2,
    seed=42,


    #evaluation_strategy="steps",  # or "epoch"
    
)

#eval metric
def compute_iou(pred, true):
    x1 = max(pred[0], true[0])
    y1 = max(pred[1], true[1])
    x2 = min(pred[2], true[2])
    y2 = min(pred[3], true[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    pred_area = max(0, (pred[2] - pred[0])) * max(0, (pred[3] - pred[1]))
    true_area = max(0, (true[2] - true[0])) * max(0, (true[3] - true[1]))
    union = pred_area + true_area - inter_area
    return inter_area / union if union > 0 else 0.0

def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    l2 = np.mean((preds - labels) ** 2)
    smooth_l1 = np.where(np.abs(preds - labels) < 1, 0.5 * (preds - labels)**2, np.abs(preds - labels) - 0.5).mean()
    iou_scores = [compute_iou(p, l) for p, l in zip(preds, labels)]
    return {
        "eval_l2_loss": l2,
        "eval_smoothl1_loss": smooth_l1,
        "eval_iou": np.mean(iou_scores)
    }



# # === IoU Loss Trainer ===
# def compute_iou_tensor(boxes1, boxes2):
#     x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
#     y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
#     x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
#     y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

#     inter_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
#     area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
#     area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
#     union = area1 + area2 - inter_area
#     iou = inter_area / (union + 1e-6)
#     return iou

# class IoUTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         outputs = model(
#             input_ids=inputs["input_ids"].to(model.device),
#             attention_mask=inputs["attention_mask"].to(model.device),
#             output_hidden_states=True
#         )
#         last_hidden = outputs.hidden_states[-1]
#         pred_boxes = model.regression_head(last_hidden)

       

#         # true_boxes = torch.tensor(inputs["labels"], dtype=torch.float32).to(model.device)
#         true_boxes = inputs["labels"].clone().detach().float().to(model.device)

#         # Already shape (batch_size, 4), no need to flatten
#         # pred_boxes = pred_boxes.view(-1, 4)
#         # true_boxes = true_boxes.view(-1, 4)
        
#         # 🔍 Debug: Print predictions and targets
#         print("🔎 Predicted bbox:", pred_boxes[0].detach().cpu().numpy())
#         print("🎯 True bbox:", true_boxes[0].detach().cpu().numpy())
        
#         iou = compute_iou_tensor(pred_boxes, true_boxes)

#         loss = 1 - iou.mean()

#         wandb.log({"eval_iou_loss": loss.item(), "mean_iou": iou.mean().item()})
#         return (loss, outputs) if return_outputs else loss




# === GIoU / DIoU / CIoU Utilities ===
def bbox_area(box):
    return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

def bbox_iou(box1, box2, eps=1e-7):
    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union_area = bbox_area(box1) + bbox_area(box2) - inter_area
    return inter_area / (union_area + eps)

def bbox_giou(box1, box2, eps=1e-7):
    iou = bbox_iou(box1, box2, eps)

    enclose_x1 = torch.min(box1[:, 0], box2[:, 0])
    enclose_y1 = torch.min(box1[:, 1], box2[:, 1])
    enclose_x2 = torch.max(box1[:, 2], box2[:, 2])
    enclose_y2 = torch.max(box1[:, 3], box2[:, 3])

    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    return iou - (enclose_area - (bbox_area(box1) + bbox_area(box2) - bbox_iou(box1, box2, eps))) / (enclose_area + eps)

def bbox_diou(box1, box2, eps=1e-7):
    iou = bbox_iou(box1, box2, eps)
    center1 = (box1[:, :2] + box1[:, 2:]) / 2
    center2 = (box2[:, :2] + box2[:, 2:]) / 2
    center_dist = ((center1 - center2)**2).sum(dim=1)

    enclose_x1 = torch.min(box1[:, 0], box2[:, 0])
    enclose_y1 = torch.min(box1[:, 1], box2[:, 1])
    enclose_x2 = torch.max(box1[:, 2], box2[:, 2])
    enclose_y2 = torch.max(box1[:, 3], box2[:, 3])

    c2 = ((enclose_x2 - enclose_x1)**2 + (enclose_y2 - enclose_y1)**2).clamp(eps)
    return iou - center_dist / c2

def bbox_ciou(box1, box2, eps=1e-7):
    iou = bbox_iou(box1, box2, eps)
    diou_term = bbox_diou(box1, box2, eps)

    w1, h1 = box1[:, 2] - box1[:, 0], box1[:, 3] - box1[:, 1]
    w2, h2 = box2[:, 2] - box2[:, 0], box2[:, 3] - box2[:, 1]
    v = (4 / (torch.pi ** 2)) * (torch.atan(w1 / h1) - torch.atan(w2 / h2)).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return diou_term - alpha * v

# # === GIoU Trainer ===
# class GIoUTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         outputs = model(
#             input_ids=inputs["input_ids"].to(model.device),
#             attention_mask=inputs["attention_mask"].to(model.device),
#             output_hidden_states=True
#         )
#         last_hidden = outputs.hidden_states[-1]
#         pred_boxes = model.regression_head(last_hidden)
#         true_boxes = inputs["labels"].clone().detach().float().to(model.device)

#         giou = bbox_giou(pred_boxes, true_boxes)
#         loss = 1 - giou.mean()
#         wandb.log({"train/giou_loss": loss.item(), "mean_giou": giou.mean().item()})
#         return (loss, outputs) if return_outputs else loss

# # === DIoU Trainer ===
# class DIoUTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         outputs = model(
#             input_ids=inputs["input_ids"].to(model.device),
#             attention_mask=inputs["attention_mask"].to(model.device),
#             output_hidden_states=True
#         )
#         last_hidden = outputs.hidden_states[-1]
#         pred_boxes = model.regression_head(last_hidden)
#         true_boxes = inputs["labels"].clone().detach().float().to(model.device)

#         diou = bbox_diou(pred_boxes, true_boxes)
#         loss = 1 - diou.mean()
#         wandb.log({"train/diou_loss": loss.item(), "mean_diou": diou.mean().item()})
#         return (loss, outputs) if return_outputs else loss

# # === CIoU Trainer ===
# class CIoUTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         outputs = model(
#             input_ids=inputs["input_ids"].to(model.device),
#             attention_mask=inputs["attention_mask"].to(model.device),
#             output_hidden_states=True
#         )
#         last_hidden = outputs.hidden_states[-1]
#         pred_boxes = model.regression_head(last_hidden)
#         true_boxes = inputs["labels"].clone().detach().float().to(model.device)

#         ciou = bbox_ciou(pred_boxes, true_boxes)
#         loss = 1 - ciou.mean()
#         wandb.log({"train/ciou_loss": loss.item(), "mean_ciou": ciou.mean().item()})
#         return (loss, outputs) if return_outputs else loss

    
    
    
# # ===  Smooth L1 Loss Trainer ===
# class SmoothL1Trainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         outputs = model(
#             input_ids=inputs["input_ids"].to(model.device),
#             attention_mask=inputs["attention_mask"].to(model.device),
#             output_hidden_states=True
#         )
#         last_hidden = outputs.hidden_states[-1]
#         pred_boxes = model.regression_head(last_hidden)

#         # true_boxes = torch.tensor(inputs["labels"], dtype=torch.float32).to(model.device)
#         true_boxes = inputs["labels"].clone().detach().float().to(model.device)

#         loss = F.smooth_l1_loss(pred_boxes, true_boxes)

#         wandb.log({"eval_smoothl1_loss": loss.item()})
#         return (loss, outputs) if return_outputs else loss


# ===  L2 Loss Trainer ===
# class DirectBBoxL2Trainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         outputs = model(
#             input_ids=inputs["input_ids"].to(model.device),
#             attention_mask=inputs["attention_mask"].to(model.device),
#             output_hidden_states=True
#         )
#         last_hidden = outputs.hidden_states[-1]
#         pred_boxes = model.regression_head(last_hidden)

#         try:
#             true_boxes = torch.tensor(inputs["bbox"], dtype=torch.float32).to(model.device)
#         except Exception as e:
#             wandb.log({"malformed_bbox_text": str(inputs.get("bbox", "missing"))})
#             true_boxes = torch.zeros(pred_boxes.shape, device=model.device)

#         l2 = F.mse_loss(pred_boxes, true_boxes)
#         wandb.log({"l2_loss": l2.item()})
#         return (l2, outputs) if return_outputs else l2

# === Clear memory before training ===
torch.cuda.empty_cache()

# === Trainer ===
trainer = SmoothL1Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=None,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics
)

#resume_from_checkpoint=True
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
save_and_push_model(model, processor, "Llama-3.2-11B-finetuned-waveUI-l1", HF_TOKEN)

wandb.finish()