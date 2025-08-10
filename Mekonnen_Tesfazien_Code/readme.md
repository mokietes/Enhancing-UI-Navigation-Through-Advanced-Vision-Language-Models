# Vision-Language Fine-Tuning for UI Element Detection

This repository contains training pipelines and evaluation scripts for fine-tuning large vision-language models (e.g., LLaMA 3.2 11B Vision) on UI screenshots to predict bounding boxes for target buttons and text elements. The project compares multiple loss functions and training strategies to optimize detection accuracy on datasets like WaveUI and Rico.

---

## 📁 Project Structure

```
.
├── main.py                  # LoRA fine-tuning using FastVisionModel
├── mainL2.py               # L2 loss-based training with regression head
├── L1LossTraining.py       # Smooth L1 loss decoding via text token prediction
├── sfsuCluster.py          # Combined loss trainer with multiple regression losses
├── mainUpgrade.py          # Enhanced LoRA training with dynamic prompt formatting
├── modelOutputTest.ipynb   # Notebook for testing model outputs
├── plottest.py             # Bounding box visualization using PIL
└── readme.md               # Project documentation
```

---
