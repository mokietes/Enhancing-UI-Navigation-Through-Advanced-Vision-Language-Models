import os
import torch
from datasets import load_dataset, load_dataset_builder 
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl
import wandb
from evaluate import load
import numpy as np
from tqdm import tqdm 
import os
import traceback
from huggingface_hub import HfApi

HF_TOKEN = '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'  # for hugging face.
WANDB_PROJECT = "Llama-3.2-11B-finetuned-main"

# Initialize Weights & Biases
wandb.init(project=WANDB_PROJECT)


# Adjust CUDA memory configuration to avoid fragmentation

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


