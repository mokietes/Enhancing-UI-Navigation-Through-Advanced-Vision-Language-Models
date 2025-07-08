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
HF_TOKEN = ''
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
