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
from torchvision import transforms  # Added for data augmentation
from torchvision.models.detection import fasterrcnn_resnet50_fpn  # Added for using Faster R-CNN
from torchvision.ops import nms  # Added for Non-Maximum Suppression
from PIL import Image

# Environment setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
HF_TOKEN = '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'
WANDB_PROJECT = "Llama-3.2-11B-finetuned-lora-wave-ui_3"
RANDOM_SEED = 3407

# Test configuration
TEST_MODE = False  # Set to False for full training.
TEST_SAMPLES = 20  # Number of samples to use in test mode
MAX_STEPS = 10    # Maximum training steps in test mode

# Data augmentation transformations
augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # rotate by 10 degrees
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

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

# New function to initialize the Faster R-CNN model
def initialize_model(num_classes: int) -> torch.nn.Module:
    """Initialize and configure the Faster R-CNN model"""
    print("Initializing model...")
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Replace the classifier with a new one for your specific classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)
    
    return model

def prepare_dataset(dataset):
    """Enhanced dataset preparation with comprehensive validation"""
    def is_valid_sample(sample):
        # Check required fields
        required_fields = ['image', 'OCR', 'bbox']
        if not all(field in sample for field in required_fields):
            return False
            
        # Verify bbox format
        bbox = sample['bbox']
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False
        if not all(isinstance(coord, (int, float)) for coord in bbox):
            return False
            
        # Verify OCR text
        if not isinstance(sample['OCR'], str) or len(sample['OCR'].strip()) == 0:
            return False
            
        # Verify resolution if available
        if 'resolution' in sample and sample['resolution']:
            if not isinstance(sample['resolution'], (list, tuple)) or len(sample['resolution']) != 2:
                return False
            if not all(isinstance(dim, (int, float)) for dim in sample['resolution']):
                return False
                
        return True
    
    print("Preparing and validating dataset...")
    return dataset.filter(is_valid_sample)

def extract_image_from_bbox(sample: Dict) -> Image:
    """Extract image from the bounding box coordinates"""
    image = Image.open(sample['image'])
    bbox = sample['bbox']
    extracted_image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    return extracted_image

def convert_to_conversation(sample: Dict) -> Dict:
    """Enhanced conversation format with comprehensive button detection instructions"""
    # Extract all available context
    button_text = sample['OCR']
    button_type = sample.get('type', 'button')
    platform = sample.get('platform', 'UI')
    purpose = sample.get('purpose', '')
    description = sample.get('description', '')
    dataset_instruction = sample.get('instruction', '')
    expectation = sample.get('expectation', '')
    
    # Build comprehensive instruction with context blocks
    context_blocks = []
    
    # Add available context blocks
    if description:
        context_blocks.append(f"Task Context: {description}")
    if dataset_instruction:
        context_blocks.append(f"Original Instruction: {dataset_instruction}")
    if expectation:
        context_blocks.append(f"Expected Behavior: {expectation}")
    
    # Main instruction with detailed guidance
    main_instruction = (
        f"Find the exact bounding box coordinates of the {button_type} containing the text '{button_text}' "
        f"in this {platform} interface image. "
        f"{f'This button {purpose}. ' if purpose else ''}"
    )
    
    # Visual detection guidelines
    visual_guidelines = (
        "Look for visual elements that indicate this is a button, such as:"
        "\n- Rectangular or rounded shape"
        "\n- Contrasting background color"
        "\n- Borders or shadows"
        "\n- Clickable appearance"
        "\n- Interactive element styling"
    )
    
    # Coordinate specification
    coordinate_spec = (
        "\nReturn ONLY the coordinates as [x1, y1, x2, y2] where:"
        "\n- x1, y1: top-left corner coordinates"
        "\n- x2, y2: bottom-right corner coordinates"
    )
    
    # Quality requirements
    quality_requirements = (
        "\nEnsure the coordinates:"
        "\n1. Capture the entire button including any borders or shadows"
        "\n2. Are precise to the pixel level"
        "\n3. Form a rectangle that completely contains the button"
        "\n4. Include any padding or interactive areas"
    )
    
    # Add resolution information if available
    resolution_info = ""
    if 'resolution' in sample and sample['resolution']:
        width, height = sample['resolution']
        resolution_info = f"\nImage dimensions: {width}x{height} pixels"
    
    # Combine all sections
    instruction = "\n\n".join(filter(None, [
        "\n".join(context_blocks),
        main_instruction,
        visual_guidelines,
        coordinate_spec,
        quality_requirements,
        resolution_info,
        f"Target Button Text: '{button_text}'"
    ]))
    
    extracted_image = extract_image_from_bbox(sample)
    
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
                    {"type": "image", "image": extracted_image},
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
            num_train_epochs=3,
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
            dataset_num_proc=16,
            max_seq_length=4096,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            report_to=["wandb"],
        )

def save_and_push_model(model, tokenizer, repo_id: str):
    """Save and push model to hub with proper error handling"""
    print("Saving model...")
    try:
        save_dir = "test_lora_model" if TEST_MODE else "lora_model"
        test_repo_id = f"{repo_id}-test" if TEST_MODE else repo_id
        
        model.save_pretrained(
            save_dir,
            safe_serialization=True,
            save_method="merged_16bit"
        )
        tokenizer.save_pretrained(save_dir)
        
        # Log model artifacts to wandb
        wandb.save(f"{save_dir}/*")
        
        print("Pushing tokenizer to hub...")
        tokenizer.push_to_hub(
            test_repo_id,
            token=HF_TOKEN,
            safe_serialization=True
        )
        
        print("Pushing model to hub...")
        model.push_to_hub(
            test_repo_id,
            token=HF_TOKEN,
            safe_serialization=True
        )
        
        wandb.log({"model_saved": True})
        print("Model saved and pushed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during model save/push: {str(e)}")
        wandb.log({"model_save_error": str(e)})
        return False

def process_dataset_in_batches(dataset, batch_size=100):
    """Process dataset in batches to manage memory"""
    processed_data = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:min(i + batch_size, len(dataset))]
        batch_processed = [convert_to_conversation(sample) for sample in batch]
        processed_data.extend(batch_processed)
        # Clear some memory
        if i % 1000 == 0:
            torch.cuda.empty_cache()
    return processed_data

# New function for post-processing predictions using NMS
def post_process(predictions, iou_threshold=0.5):
    """Apply Non-Maximum Suppression (NMS) to refine bbox predictions"""
    keep = nms(predictions['boxes'], predictions['scores'], iou_threshold)
    return {
        'boxes': predictions['boxes'][keep],
        'scores': predictions['scores'][keep],
        'labels': predictions['labels'][keep]
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
    num_classes = 2  # Background + button
    model = initialize_model(num_classes)  # Updated to use Faster R-CNN model
    
    # Load and prepare dataset
    print("Loading dataset...")
    dataset = load_dataset("miketes/wave-ui_2")
    cleaned_dataset = prepare_dataset(dataset["train"])
    
    # Handle test mode dataset size
    if TEST_MODE:
        print(f"Selecting {TEST_SAMPLES} samples for testing...")
        cleaned_dataset = cleaned_dataset.select(range(min(TEST_SAMPLES, len(cleaned_dataset))))
    
    wandb.log({"dataset_size": len(cleaned_dataset)})
    
    # Split dataset
    print("Splitting dataset...")
    total_samples = len(cleaned_dataset)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    
    # train_dataset = cleaned_dataset.shuffle(seed=RANDOM_SEED).select(range(train_size))
    # val_dataset = cleaned_dataset.shuffle(seed=RANDOM_SEED).select(range(train_size, train_size + val_size))
    
    # Dataset in batches training 
    train_dataset = process_dataset_in_batches(cleaned_dataset)
    val_dataset = process_dataset_in_batches(cleaned_dataset)

    # Clear memory before conversion
    del cleaned_dataset
    torch.cuda.empty_cache()

    # Convert datasets
    print("Converting dataset format...")
    train_dataset = [convert_to_conversation(sample) for sample in train_dataset]
    val_dataset = [convert_to_conversation(sample) for sample in val_dataset]
    
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
    )
    
    # Train model
    print("Starting training...")
    torch.cuda.empty_cache()
    trainer.train()
    
    repo_id = "miketes/Llama-3.2-11B-finetuned-lora-wave-ui_3"
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