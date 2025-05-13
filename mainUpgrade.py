# === Imports ===
import os
import torch
from datasets import load_dataset
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl
import wandb
import traceback
from huggingface_hub import HfApi
from tqdm import tqdm

# === Configurations ===
HF_TOKEN = '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'
WANDB_PROJECT = "Llama-3.2-11B-finetuned-main"
dataset_path = "/Users/923676946/git-repos/Visual-Data-Mining-AI-Model/training/datasets/wave-ui/data"

# === Initialize environment ===
wandb.init(project=WANDB_PROJECT)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

