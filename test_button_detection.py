import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
import os
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Tuple
import random

# Set random seed for reproducibility
RANDOM_SEED = 3407
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Constants
BUTTON_TYPES = ["button", "link", "input", "select"]  # Add more types as needed
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5

# Data augmentation transformations specifically for UI elements
transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.Lambda(lambda x: x.convert('RGB') if x.mode == 'RGBA' else x),  # Convert RGBA to RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def initialize_model(num_classes: int) -> torch.nn.Module:
    """Initialize the Faster R-CNN model with UI-specific configurations."""
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)
    return model

