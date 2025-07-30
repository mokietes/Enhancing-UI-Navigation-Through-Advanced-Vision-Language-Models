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

# === convert_to_conversation ===
def convert_to_conversation(sample):
    """
    Converts a UI element data sample into a multi-modal instruction format
    compatible with FastVisionModel. Combines global and dynamic context instructions.
    """
    bbox = sample.get("bbox", "[0, 0, 0, 0]")
    ocr_label = sample.get("OCR")
    name = sample.get("name")
    description = sample.get("description")
    element_type = sample.get("type")
    language = sample.get("language")
    platform = sample.get("platform")
    purpose = sample.get("purpose")
    expectation = sample.get("expectation")
    instruction_context = sample.get("instruction")
    resolution = sample.get("resolution")

    global_instruction = (
        "You are given a screenshot of a user interface. "
        "Your task is to locate a button or interactive element based on the provided description. "
        "Return ONLY its bounding box coordinates in [x1, y1, x2, y2] format. Do not add explanation."
    )

    instruction_parts = []

    if name:
        instruction_parts.append(f"The element is named '{name}'.")
    if ocr_label:
        instruction_parts.append(f"It includes the visible text: '{ocr_label}'.")
    if element_type:
        instruction_parts.append(f"This is a '{element_type}' UI component.")
    if description:
        instruction_parts.append(f"Description: {description}.")
    if purpose:
        instruction_parts.append(f"Purpose: {purpose}.")
    if expectation:
        instruction_parts.append(f"Expected behavior: {expectation}.")
    if platform:
        instruction_parts.append(f"The interface belongs to the '{platform}' platform.")
    if language:
        instruction_parts.append(f"The UI is in '{language}' language.")
    if instruction_context:
        instruction_parts.append(f"Additional notes: {instruction_context}")
    if resolution:
        instruction_parts.append(f"Image resolution is {resolution}.")

    instruction_parts.append(
        "Return the coordinates as [x1, y1, x2, y2] where:"
        "\n- x1, y1 = top-left corner"
        "\n- x2, y2 = bottom-right corner"
    )

    dynamic_instruction = " ".join(instruction_parts)
