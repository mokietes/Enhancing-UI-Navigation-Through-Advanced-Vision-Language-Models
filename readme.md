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

## 🛠️ Setup Instructions

### 1. Environment Setup

#### Local
```bash
git clone https://github.com/mokietes/Visual-Data-Mining-AI-Model.git
cd Visual-Data-Mining-AI-Model
conda env create -f scripts/setup/environment.yml
conda activate vt-spatial
pip install -r scripts/setup/requirements.txt
```

#### HPC Cluster
```bash
module load cuda/11.8
module load anaconda3/2023.03
source scripts/hpc/gpu_activation.sh
```

---

## 📦 Dependencies

- Python 3.8+
- PyTorch 2.0+
- Hugging Face Transformers
- Unsloth
- WandB
- TRL
- tqdm
- numpy
- matplotlib

---

##  Dataset

### Load Dataset
```python
from datasets import load_dataset
dataset = load_dataset("miketes/Web-filtered-english-wave-ui-25k")
```

### Preprocessing
```bash
python src/data/preprocessing.py --input_path /path/to/raw --output_path /path/to/processed
```

---

##  Training

### A. LoRA Fine-Tuning (`main.py` / `mainUpgrade.py`)
```bash
python main.py
```

### B. Smooth L1 Loss Training (`L1LossTraining.py`)
```bash
python L1LossTraining.py
```

### C. L2 Loss Training (`mainL2.py`)
```bash
python mainL2.py
```

### D. Combined Loss with GIoU/L1/L2/SmoothL1 (`sfsuCluster.py`)
```bash
python sfsuCluster.py
```

---

##  Evaluation & Visualization

### Generate Evaluation Metrics
```bash
python src/evaluation/benchmark.py --model_path ./models/full --test_samples 100 --save_visualizations
```

### Visualize Bounding Box Predictions
```bash
python plottest.py
```

---

##  Metrics

- IoU (Intersection over Union)
- Smooth L1 Loss
- L2 Loss
- Pixel-wise bounding box regression error

All metrics are logged to Weights & Biases and optionally visualized using TensorBoard.

---

##  Model Upload

All scripts provide a `save_and_push_model` function for uploading to Hugging Face Hub.

```python
save_and_push_model(model, processor, "your-repo-name", HF_TOKEN)
```

---

##  HPC Workflow

- Remote Jupyter: `bash scripts/hpc/jupyter_remote.sh`
- Job Scheduling: `sbatch scripts/hpc/job_chain.sh`

---

##  Contact

**Author:** Mokie Tesfazien  
**Email:** mokietes@sfsu.edu  
**Project Repo:** [https://github.com/mokietes/Enhancing-UI-Navigation-Through-Advanced-Vision-Language-Models](https://github.com/mokietes/Enhancing-UI-Navigation-Through-Advanced-Vision-Language-Models)
