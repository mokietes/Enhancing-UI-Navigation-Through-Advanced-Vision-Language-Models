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
        
        if not matched:
            metrics["false_positives"] += 1
    
    metrics["false_negatives"] = len(true_boxes) - len(matched_true_boxes)
    
    # Calculate precision and recall
    if metrics["true_positives"] + metrics["false_positives"] > 0:
        metrics["precision"] = metrics["true_positives"] / (metrics["true_positives"] + metrics["false_positives"])
    else:
        metrics["precision"] = 0
        
    if metrics["true_positives"] + metrics["false_negatives"] > 0:
        metrics["recall"] = metrics["true_positives"] / (metrics["true_positives"] + metrics["false_negatives"])
    else:
        metrics["recall"] = 0
        
    # Calculate F1 score
    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
    else:
        metrics["f1"] = 0
        
    return metrics

def visualize_detection(image: Image.Image, boxes: List[List[float]], labels: List[int], 
                       scores: List[float], title: str = "Detection Results"):
    """Visualize detection results."""
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red')
        ax.add_patch(rect)
        ax.text(x1, y1, f"{BUTTON_TYPES[label]}: {score:.2f}", 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(title)
    plt.axis('off')
    plt.show()

def evaluate_model(model: torch.nn.Module, dataset, num_samples: int = 5):
    """Evaluate model on test samples."""
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    total_metrics = {
        "precision": 0,
        "recall": 0,
        "f1": 0
    }
    
    for i in range(num_samples):
        sample = dataset[i]
        image = sample['image']
        if isinstance(image, str):  # If image is a path
            image = Image.open(image).convert('RGB')
        
        # Prepare image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get predictions
        with torch.no_grad():
            predictions = model(image_tensor)[0]
        
        # Post-process predictions
        pred_boxes, pred_labels, pred_scores = post_process(predictions)
        
        # Get ground truth
        true_boxes = sample['boxes']
        true_labels = sample['labels']
        
        # Calculate metrics
        metrics = calculate_metrics(pred_boxes, pred_labels, true_boxes, true_labels)
        
        # Update total metrics
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        
        # Visualize results
        visualize_detection(image, pred_boxes, pred_labels, pred_scores,
                          f"Sample {i+1} - Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}")
