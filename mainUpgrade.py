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

# === Save and Push ===
def save_and_push_model(model, tokenizer, repo_id: str, HF_TOKEN: str, enable_hf: bool=True):
    try:
        model_name = f"{repo_id}-Second-Brain-Summarization"
        print(f"Model name: {model_name}")

        model.save_pretrained_merged(
            model_name,
            tokenizer,
            save_method="merged_16bit",
        )

        if enable_hf:
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
        return True

    except Exception as e:
        print("❌ Error while saving or pushing model:")
        traceback.print_exc()
        return False

# === Dataset Load ===
dataset = load_dataset("parquet", data_files={
    "train": os.path.join(dataset_path, "train-*.parquet"),
    "validation": os.path.join(dataset_path, "validation-*.parquet"),
    "test": os.path.join(dataset_path, "test-*.parquet"),
})

train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]

print("\nProcessing training dataset...")
train_dataset = [convert_to_conversation(sample) for sample in tqdm(train_dataset, desc="Processing Train Data", unit="sample")]

print("\nProcessing validation dataset...")
val_dataset = [convert_to_conversation(sample) for sample in tqdm(val_dataset, desc="Processing Validation Data", unit="sample")]

# === Load Model ===
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

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

FastVisionModel.for_training(model)

# === Callbacks ===
class SaveCheckpointCallback(TrainerCallback):
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"Saving checkpoint at step {state.global_step}")
        model.save_pretrained(f"checkpoint-{state.global_step}")
        tokenizer.save_pretrained(f"checkpoint-{state.global_step}")

class WandBLoggingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        eval_metrics = kwargs.get("metrics", {})
        eval_loss = eval_metrics.get("eval_loss", None)
        if eval_loss is not None:
            wandb.log({"epoch": state.epoch, "eval_loss": eval_loss})

    def on_train_end(self, args, state, control, **kwargs):
        train_loss = kwargs.get("train_loss", 0)
        wandb.log({"train_loss": train_loss})

# === Training Arguments ===
training_args = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    warmup_steps=500,
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
    fp16_full_eval=True,
    per_device_eval_batch_size=4,
    eval_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=1000,
    save_steps=500,
    save_total_limit=5,
    resume_from_checkpoint="./outputs/check",
)

# === Trainer ===
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
    callbacks=[SaveCheckpointCallback(), WandBLoggingCallback()],
)

# === Train Model ===
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()
trainer.train(resume_from_checkpoint=True)

# === Save Model ===
repo_id = "Llama-3.2-11B-finetuned-waveUI"
success = save_and_push_model(model, tokenizer, repo_id, HF_TOKEN)

if success:
    print("🎉 Model saved and/or pushed successfully!")
else:
    print("⚠️ Model save or push failed.")

wandb.finish()
