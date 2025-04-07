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
    Converts a UI element data sample into a multi-modal conversation format
    for training a vision-language model to predict bounding boxes of text elements.
    """
    
    resolution = sample.get("resolution", "Unknown Resolution")
    bbox = sample.get("bbox", "[0, 0, 0, 0]") 
    ocr_label = sample.get("OCR", "")
    name = sample.get("name", "Unknown Element")
    description = sample.get("description", "No description available.")
    element_type = sample.get("type", "Unknown Type")
    language = sample.get("language", "Unknown Language")
    platform = sample.get("platform", "Unknown Platform")
    purpose = sample.get("purpose", "No specific purpose provided.")
    expectation = sample.get("expectation", "No expectation specified.")
    instructions = sample.get("instruction", "No instruction provided.")

    instruction = (
        f"In this user interface image with resolution {resolution}, locate the text element {ocr_label} named {name} used for "
        f"{description} in {language} language for the purpose of {purpose} which is used for {instructions}. It is expected "
        f"to make {expectation} in {platform} platform. Determine its precise "
        f"bounding box coordinates. The coordinates should be formatted as [x1, y1, x2, y2] where:"
        f"\n- x1, y1 is the top-left corner"
        f"\n- x2, y2 is the bottom-right corner"
        f"\nThe box should tightly enclose only this specific text element. Return only the coordinates."
    )

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": instruction},
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



#Saving function
def save_and_push_model(model, tokenizer, repo_id: str, HF_TOKEN: str, enable_hf: bool=True):
    """
    Save the fine-tuned quantized model and tokenizer locally and optionally push to Hugging Face Hub with merged 16bit.
    """
    try:
        model_name = f"{repo_id}-Second-Brain-Summarization"
        print(f"Model name: {model_name}")

        # Save the model and tokenizer locally
        model.save_pretrained_merged(
            model_name,
            tokenizer,
            save_method="merged_16bit",
        )

        if enable_hf:
            # Push to Hugging Face Hub
            api = HfApi()
            user_info = api.whoami(token=HF_TOKEN)
            huggingface_user = user_info["name"]
            print(f"Current Hugging Face user: {huggingface_user}")

            model.push_to_hub_merged(
                f"{huggingface_user}/{model_name}",
                tokenizer=tokenizer,
                save_method="merged_16bit",
                token=HF_TOKEN,
            )

            print("✅ Model and tokenizer pushed successfully!")
        else:
            print("Hugging Face upload disabled. Model saved locally.")

        return True  # <-- Success

    except Exception as e:
        print("❌ Error while saving or pushing model:")
        traceback.print_exc()
        return False  # <-- Failure

    

# Load the dataset
dataset_path = "/Users/923676946/git-repos/Visual-Data-Mining-AI-Model/training/datasets/wave-ui/data"  

# Load the dataset with the new path, specifying the wildcard for all .parquet files
dataset = load_dataset("parquet", data_files={
    "train": os.path.join(dataset_path, "train-*.parquet"),  # Wildcard to load all train files
    "validation": os.path.join(dataset_path, "validation-*.parquet"),  # Wildcard for validation files
    "test": os.path.join(dataset_path, "test-*.parquet"),  # Wildcard for test files
})

train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]


# Convert the dataset using tqdm progress bar
print("\nProcessing training dataset...")
train_dataset = [convert_to_conversation(sample) for sample in tqdm(train_dataset, desc="Processing Train Data", unit="sample")]

print("\nProcessing validation dataset...")
val_dataset = [convert_to_conversation(sample) for sample in tqdm(val_dataset, desc="Processing Validation Data", unit="sample")]

print("\nProcessing test dataset...")
# test_dataset = [convert_to_conversation(sample) for sample in tqdm(test_dataset, desc="Processing Test Data", unit="sample")]


# Load the model
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# Setup LoRA
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Training setup
FastVisionModel.for_training(model)

class SaveCheckpointCallback(TrainerCallback):
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"Saving checkpoint at step {state.global_step}")
        model.save_pretrained(f"checkpoint-{state.global_step}")
        tokenizer.save_pretrained(f"checkpoint-{state.global_step}")

training_args = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    
    warmup_steps=500,
    # max_steps=500,  # Adjust as needed
    num_train_epochs=1,
    learning_rate=2e-5,
    fp16=not is_bf16_supported(),
    bf16=is_bf16_supported(),
    logging_steps=500,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="./outputs/check",
    report_to="wandb",
    remove_unused_columns=False,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    dataset_num_proc=4,
    max_seq_length=2048,

    
    # eval_strategy="steps",
    # eval_steps=20,
    
    #eval
    fp16_full_eval = True,
    per_device_eval_batch_size = 4,
    eval_accumulation_steps = 4,
    eval_strategy = "steps",
    eval_steps = 1000,

    
    save_steps=500,  # Save checkpoint every 100 steps
    save_total_limit=5, # Only keep the last 5 checkpoints
    resume_from_checkpoint="./outputs/check",  # Replace XXX with the step number
)


class WandBLoggingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        # Get eval metrics from the kwargs
        eval_metrics = kwargs.get("metrics", {})
        
        # Extract eval_loss if it exists
        eval_loss = eval_metrics.get("eval_loss", None)
        
        # Log to WandB only if eval_loss is available
        if eval_loss is not None:
            epoch = state.epoch
            wandb.log({"epoch": epoch, "eval_loss": eval_loss})
        
    def on_train_end(self, args, state, control, **kwargs):
        # Log final training loss if available
        train_loss = kwargs.get("train_loss", 0)
        wandb.log({"train_loss": train_loss})


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    #train_dataset=converted_dataset,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  
    args=training_args,
    callbacks=[SaveCheckpointCallback(), WandBLoggingCallback()],
)

torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()

# Train the model
#resume_from_checkpoint=True
trainer.train(resume_from_checkpoint=True)

# Save the final model
repo_id = "Llama-3.2-11B-finetuned-waveUI"

success = save_and_push_model(model, tokenizer, repo_id, HF_TOKEN)

if success:
    print("🎉 Model saved and/or pushed successfully!")
else:
    print("⚠️ Model save or push failed.")    

wandb.finish()

