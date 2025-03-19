import os
from unsloth import FastVisionModel
import torch
from datasets import load_dataset
from transformers import TextStreamer
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import random
import numpy as np
import wandb
import gc
import numpy as np

# Environment setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
HF_TOKEN = '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'  # Replace with your token
WANDB_PROJECT = "Llama-3.2-11B-finetuned-full-wave-ui_3"
RANDOM_SEED = 3407

# Wandb account
# Add this near the top of your script, after the imports
# wandb.login(key="f7bf6b1c9d553e3c69c865dbd823789f9e1b5a5d")  # Replace with your actual API key

# Test configuration
TEST_MODE = True  # Set to False for full training
TEST_SAMPLES = 20  # Number of samples to use in test mode
MAX_STEPS = 10    # Maximum training steps in test mode

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def initialize_wandb(test_mode: bool):
    """Initialize wandb with configuration"""
    config = {
        "test_mode": test_mode,
        "model": "Llama-3.2-11B-Vision",
        "dataset": "Web-filtered-english-wave-ui-25k",
        "lora_rank": 8 if test_mode else 32,
        "lora_alpha": 16 if test_mode else 32,
        "batch_size": 2 if test_mode else 8,
        "learning_rate": 5e-4 if test_mode else 2e-4,
        "epochs": 1 if test_mode else 3,
        "max_steps": MAX_STEPS if test_mode else None,
        "samples": TEST_SAMPLES if test_mode else "full",
    }
    
    run_name = "test-run" if test_mode else "full-training"
    wandb.init(
        project=WANDB_PROJECT,
        config=config,
        name=run_name,
    )

def initialize_model() -> Tuple[FastVisionModel, any]:
    """Initialize and configure the vision model with optimized settings"""
    print("Initializing model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Llama-3.2-11B-Vision-Instruct",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    
    # Store the original dtype before quantization
    original_dtype = model.dtype
    
    model.config.gradient_checkpointing = True
    
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8 if TEST_MODE else 32,
        lora_alpha=16 if TEST_MODE else 32,
        lora_dropout=0.1,
        bias="none",
        random_state=RANDOM_SEED,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Store the dtype in the model config for later use
    model.config.original_dtype = original_dtype
    
    return model, tokenizer

def prepare_dataset(dataset):
    """Enhanced dataset preparation with comprehensive validation"""
    def is_valid_sample(sample):
        required_fields = ['image', 'OCR', 'bbox']
        if not all(field in sample for field in required_fields):
            return False
            
        # Verify bbox format
        bbox = sample['bbox']
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False
        if not all(isinstance(coord, (int, float)) for coord in bbox):
            return False
            
        if not isinstance(sample['OCR'], str) or len(sample['OCR'].strip()) == 0:
            return False
            
        if 'resolution' in sample and sample['resolution']:
            if not isinstance(sample['resolution'], (list, tuple)) or len(sample['resolution']) != 2:
                return False
            if not all(isinstance(dim, (int, float)) for dim in sample['resolution']):
                return False
                
        return True
    
    print("Preparing and validating dataset...")
    return dataset.filter(is_valid_sample)

def convert_to_conversation(sample: Dict) -> Dict:
    """Enhanced conversation format with comprehensive button detection instructions"""
    button_text = sample['OCR']
    button_type = sample.get('type', 'button')
    platform = sample.get('platform', 'UI')
    purpose = sample.get('purpose', '')
    description = sample.get('description', '')
    dataset_instruction = sample.get('instruction', '')
    expectation = sample.get('expectation', '')
    
    context_blocks = []
    
    if description:
        context_blocks.append(f"Task Context: {description}")
    if dataset_instruction:
        context_blocks.append(f"Original Instruction: {dataset_instruction}")
    if expectation:
        context_blocks.append(f"Expected Behavior: {expectation}")
    
    main_instruction = (
        f"Find the exact bounding box coordinates of the {button_type} containing the text '{button_text}' "
        f"in this {platform} interface image. "
        f"{f'This button {purpose}. ' if purpose else ''}"
    )
    
    visual_guidelines = (
        "Look for visual elements that indicate this is a button, such as:"
        "\n- Rectangular or rounded shape"
        "\n- Contrasting background color"
        "\n- Borders or shadows"
        "\n- Clickable appearance"
        "\n- Interactive element styling"
    )
    
    coordinate_spec = (
        "\nReturn ONLY the coordinates as [x1, y1, x2, y2] where:"
        "\n- x1, y1: top-left corner coordinates"
        "\n- x2, y2: bottom-right corner coordinates"
    )
    
    quality_requirements = (
        "\nEnsure the coordinates:"
        "\n1. Capture the entire button including any borders or shadows"
        "\n2. Are precise to the pixel level"
        "\n3. Form a rectangle that completely contains the button"
        "\n4. Include any padding or interactive areas"
    )
    
    resolution_info = ""
    if 'resolution' in sample and sample['resolution']:
        width, height = sample['resolution']
        resolution_info = f"\nImage dimensions: {width}x{height} pixels"
    
    instruction = "\n\n".join(filter(None, [
        "\n".join(context_blocks),
        main_instruction,
        visual_guidelines,
        coordinate_spec,
        quality_requirements,
        resolution_info,
        f"Target Button Text: '{button_text}'"
    ]))
    
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
                    {"type": "text", "text": str(sample["bbox"])},
                ],
            },
        ]
    }

def create_training_config(output_dir: str) -> SFTConfig:
    """Create optimized training configuration"""
    if TEST_MODE:
        return SFTConfig(
            per_device_train_batch_size=2,  # Smaller batch size for testing
            gradient_accumulation_steps=1,
            max_steps=MAX_STEPS,
            learning_rate=5e-4,
            warmup_ratio=0.05,
            bf16=True,
            optim="adamw_8bit",
            logging_steps=1,
            save_steps=5,
            eval_steps=5,
            save_total_limit=1,
            output_dir=output_dir,
            seed=RANDOM_SEED,
            max_seq_length=512,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            report_to=["wandb"],
        )
    else:
        return SFTConfig(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            num_train_epochs=1,
            max_steps=None, 
            learning_rate=1e-4,
            warmup_ratio=0.1,
            bf16=True,
            optim="adamw_torch_fused",
            weight_decay=0.05,
            lr_scheduler_type="cosine",
            logging_steps=20,
            save_steps=500,
            eval_steps=100,
            save_total_limit=3,
            output_dir=output_dir,
            seed=RANDOM_SEED,
            dataset_num_proc=4, #16
            max_seq_length=2048,  #4096
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            report_to=["wandb"],
        )

def save_and_push_model(model, tokenizer, repo_id: str):
    """Save and push the complete merged model to hub"""
    print("Saving model...")
    try:
        print("Disabling gradient checkpointing...")
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
            
        # First get the base model
        print("Loading base model...")
        base_model, _ = FastVisionModel.from_pretrained(  # Unpack the tuple
            "unsloth/Llama-3.2-11B-Vision-Instruct",
            load_in_4bit=False,  # Load in full precision
            device_map="auto",
            trust_remote_code=True
        )
        
        print("Merging model weights...")
        # Get the state dict of your trained model
        trained_state_dict = model.state_dict()
        
        # Update base model's weights with trained weights
        for key in trained_state_dict:
            if key in base_model.state_dict():
                base_model.state_dict()[key].copy_(trained_state_dict[key])
        
        # Save locally first
        local_save_dir = "local_model_save"
        print(f"Saving model locally to: {local_save_dir}")
        base_model.save_pretrained(
            local_save_dir,
            safe_serialization=True,
            # max_shard_size="4.5GB"
        )
        tokenizer.save_pretrained(local_save_dir)
        
        print(f"Pushing complete model to hub: {repo_id}")
        # Push the complete merged model
        base_model.push_to_hub(
            repo_id,
            token=HF_TOKEN,
            safe_serialization=True,
            # max_shard_size="4.5GB",  # This will create ~5GB shards
        )
        
        print("Pushing tokenizer...")
        tokenizer.push_to_hub(
            repo_id,
            token=HF_TOKEN,
            safe_serialization=True
        )
        
        wandb.log({"model_saved": True})
        print("Complete model saved locally and pushed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during model save/push: {str(e)}")
        wandb.log({"model_save_error": str(e)})
        import traceback
        print(f"Full error traceback:\n{traceback.format_exc()}")
        return False

def process_dataset_in_batches(dataset, batch_size=100):
    """Process dataset in batches to manage memory"""
    processed_data = []
    for i in range(0, len(dataset), batch_size):
        # Convert Dataset to list of dictionaries
        batch = dataset[i:min(i + batch_size, len(dataset))]
        if hasattr(batch, 'to_dict'):
            # If it's a Hugging Face Dataset, convert to dict format
            batch = [dict(item) for item in batch]
        batch_processed = [convert_to_conversation(sample) for sample in batch]
        processed_data.extend(batch_processed)
        if i % 1000 == 0:
            torch.cuda.empty_cache()
    return processed_data



def compute_iou(box1, box2):
    """
    box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = max(0, (box1[2] - box1[0])) * max(0, (box1[3] - box1[1]))
    box2_area = max(0, (box2[2] - box2[0])) * max(0, (box2[3] - box2[1]))

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    else:
        return inter_area / union_area

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # If your model outputs logits, decode them here.
    # Assume predictions.shape = (batch_size, 4)
    # and labels.shape = (batch_size, 4)

    ious = []
    for pred_box, true_box in zip(predictions, labels):
        iou = compute_iou(pred_box, true_box)
        ious.append(iou)

    mean_iou = np.mean(ious)
    return {
        "mean_iou": mean_iou
    }



def main():
    # Set random seed
    set_seed(RANDOM_SEED)
    
    # Initialize wandb
    initialize_wandb(TEST_MODE)
    
    # Login to Hugging Face
    print("Logging in to Hugging Face...")
    login(token=HF_TOKEN)
    
    # Initialize model and tokenizer
    model, tokenizer = initialize_model()
    
    # Load and prepare dataset
    print("Loading dataset...")
    dataset = load_dataset("agentsea/wave-ui")
    
    # cleaned_dataset = prepare_dataset(dataset["train"])
    
    # if TEST_MODE:
    #     print(f"Selecting {TEST_SAMPLES} samples for testing...")
    #     cleaned_dataset = cleaned_dataset.select(range(min(TEST_SAMPLES, len(cleaned_dataset))))
    
    # wandb.log({"dataset_size": len(cleaned_dataset)})
    
    # print("Splitting dataset...")
    # total_samples = len(cleaned_dataset)
    # train_size = int(0.8 * total_samples)
    # val_size = int(0.1 * total_samples)
    
    # cleaned_dataset_list = [dict(item) for item in cleaned_dataset]
    
    # train_dataset = cleaned_dataset_list[:train_size]
    # val_dataset = cleaned_dataset_list[train_size:train_size + val_size]

    # /////
    # Prepare and load already-split dataset
    print("Loading and preparing dataset splits...")
    
    train_dataset = prepare_dataset(dataset["train"])
    val_dataset = prepare_dataset(dataset["validation"])
    test_dataset = prepare_dataset(dataset["test"])
    
    # Optional test mode — limit dataset size for fast iteration
    if TEST_MODE:
        print(f"Selecting {TEST_SAMPLES} samples for testing...")
        train_dataset = train_dataset.select(range(min(TEST_SAMPLES, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(TEST_SAMPLES, len(val_dataset))))
        test_dataset = test_dataset.select(range(min(TEST_SAMPLES, len(test_dataset))))
    
    # Log dataset sizes to wandb
    wandb.log({
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
        "test_dataset_size": len(test_dataset)
    })
    
    # Convert to list of dictionaries (for compatibility with custom processing functions)
    train_dataset = [dict(item) for item in train_dataset]
    val_dataset = [dict(item) for item in val_dataset]
    test_dataset = [dict(item) for item in test_dataset]

    # /////

    
    # Process datasets in batches
    print("Processing training dataset...")
    train_dataset = process_dataset_in_batches(train_dataset)
    print("Processing validation dataset...")
    val_dataset = process_dataset_in_batches(val_dataset)

    # Clear memory
    # del cleaned_dataset
    torch.cuda.empty_cache()
    gc.collect()

    # Setup training
    print("Setting up training...")
    FastVisionModel.for_training(model)
    data_collator = UnslothVisionDataCollator(model, tokenizer)
    training_args = create_training_config("outputs")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    print("Starting training...")
    torch.cuda.empty_cache()
    trainer.train()
    
    # Clean up memory before saving
    print("Cleaning up memory before model saving...")
    torch.cuda.empty_cache()
    gc.collect()
    
    repo_id = "miketes/Llama-3.2-11B-finetuned-wave-ui_3"
    if TEST_MODE:
        repo_id += "-test"
        
    success = save_and_push_model(model, tokenizer, repo_id)
    
    if success:
        print("Training and model saving completed successfully!")
    else:
        print("Training completed but model saving encountered errors.")
    
    wandb.finish()
    
    if TEST_MODE:
        print("\nTest run completed! If everything looks good, set TEST_MODE = False for full training.")

if __name__ == "__main__":
    main()