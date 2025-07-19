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


#1 Conversion instruction
def convert_to_conversation(sample):
    """
    Converts a UI element data sample into a multi-modal instruction format.
    Includes a global instruction and dynamically composes a detailed instruction with proper sentence structure.
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
    instructions = sample.get("instruction")
    resolution = sample.get("resolution")

    # Global instruction
    global_instruction = (
        "You are given a user interface screenshot. "
        "Your task is to identify the target button or text element and return its bounding box "
        "in the format [x1, y1, x2, y2]. Do not provide any explanation—just the coordinates."
    )


    # Build dynamic instruction in full sentences
    sentences = []

    if name:
        sentences.append(f"The element is named '{name}'.")
    if ocr_label:
        sentences.append(f"It contains the text label '{ocr_label}'.")
    if resolution:
        sentences.append(f"The image resolution is {resolution}.")
    if description:
        sentences.append(f"This element is used for {description}.")
    if language:
        sentences.append(f"It is presented in {language}.")
    if purpose:
        sentences.append(f"The purpose of this element is to {purpose}.")
    if expectation:
        sentences.append(f"It is expected to {expectation}.")
    if platform:
        sentences.append(f"This UI is part of the {platform} platform.")
    if instructions:
        sentences.append(f"Additional instruction context: '{instructions}'.")

    sentences.append(
        "Return the bounding box coordinates in the format [x1, y1, x2, y2], where:\n"
        "- x1, y1 is the top-left corner\n"
        "- x2, y2 is the bottom-right corner"
    )

    dynamic_instruction = " ".join(sentences)

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": global_instruction},
                    {"type": "text", "text": dynamic_instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": bbox},
                ],
            },
        ]
    }




#2 Conversion instruction
# def convert_to_conversation(sample):
#     resolution = sample.get("resolution", "Unknown Resolution")
#     ocr_label = sample.get("OCR", "")
#     name = sample.get("name", "Unknown Element")
#     description = sample.get("description", "No description available.")
#     element_type = sample.get("type", "Unknown Type")
#     language = sample.get("language", "Unknown Language")
#     platform = sample.get("platform", "Unknown Platform")
#     purpose = sample.get("purpose", "No specific purpose provided.")
#     expectation = sample.get("expectation", "No expectation specified.")
#     instructions = sample.get("instruction", "No instruction provided.")

#     instruction = f"""You are given a user interface image with a resolution of {resolution}.
#     Your task is to locate a text element with the OCR label "{ocr_label}".
#     Details about this element:
#     - Name: {name}
#     - Description: {description}
#     - Language: {language}
#     - Purpose: {purpose}
#     - Instruction Context: {instructions}
#     - Expected Behavior: {expectation}
#     - Platform: {platform}

#     Identify the precise bounding box of this text element in the image.
#     Return the coordinates in the format: [x1, y1, x2, y2], where:
#     - (x1, y1) is the top-left corner
#     - (x2, y2) is the bottom-right corner

#     The bounding box must tightly enclose only this specific text element.
#     Return only the coordinates — no explanation or extra text."""

#         return {
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "image", "image": sample["image"]},
#                         {"type": "text", "text": instruction},
#                     ],
#                 },
#                 {
#                     "role": "assistant",
#                     "content": [
#                         {"type": "text", "text": sample["bbox"]},
#                     ],
#                 },
#             ]
#         }



