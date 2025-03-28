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

def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """Calculate Intersection over Union between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0

def post_process(predictions: Dict[str, torch.Tensor], confidence_threshold: float = CONFIDENCE_THRESHOLD) -> Tuple[List[List[float]], List[int], List[float]]:
    """Post-process model predictions."""
    boxes = predictions["boxes"].detach().cpu()
    scores = predictions["scores"].detach().cpu()
    labels = predictions["labels"].detach().cpu()
    
    # Filter by confidence
    mask = scores > confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    # Apply NMS
    keep = nms(boxes, scores, iou_threshold=IOU_THRESHOLD)
    
    return boxes[keep].tolist(), labels[keep].tolist(), scores[keep].tolist()

def calculate_metrics(pred_boxes: List[List[float]], pred_labels: List[int], 
                     true_boxes: List[List[float]], true_labels: List[int]) -> Dict[str, float]:
    """Calculate detection metrics."""
    metrics = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0
    }
    
    matched_true_boxes = set()
    
    for pred_box, pred_label in zip(pred_boxes, pred_labels):
        matched = False
        for i, (true_box, true_label) in enumerate(zip(true_boxes, true_labels)):
            if i in matched_true_boxes:
                continue
                
            iou = calculate_iou(torch.tensor(pred_box), torch.tensor(true_box))
            if iou >= IOU_THRESHOLD and pred_label == true_label:
                metrics["true_positives"] += 1
                matched_true_boxes.add(i)
                matched = True
                break
