# Vision Transformer Spatial Reasoning Enhancement

This repository contains the implementation of Vision Transformer fine-tuning approaches for enhanced spatial reasoning capabilities, including both LoRA and full fine-tuning methodologies.

## Setup Instructions

### 1. Environment Setup

#### Local Development
```bash
# Clone the repository
git clone https://https://github.com/mokietes/Visual-Data-Mining-AI-Model.git
cd vision-transformer-enhancement

# Create conda environment
conda env create -f scripts/setup/environment.yml
conda activate vt-spatial

# Install additional dependencies
pip install -r scripts/setup/requirements.txt
```

#### HPC Cluster Setup
```bash
# Load required modules
module load cuda/11.8
module load anaconda3/2023.03

# Activate environment
source scripts/hpc/gpu_activation.sh
```

### 2. Dependencies

#### Core Dependencies
- Python 3.8+
- PyTorch 2.0+
- transformers
- unsloth
- wandb
- numpy
- matplotlib
- tqdm

#### Additional Tools
- CUDA 11.8+
- Jupyter Notebook
- tensorboard

### 3. Data Preparation

#### Dataset Access
```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("miketes/Web-filtered-english-wave-ui-25k")
```

#### Data Preprocessing
```bash
# Run data preprocessing script
python src/data/preprocessing.py --input_path /path/to/data --output_path /path/to/processed
```

### 4. Training

#### LoRA Fine-tuning
```bash
# Run LoRA training
python src/training/lora_training.py \
    --model_path "unsloth/Llama-3.2-11B-Vision-Instruct" \
    --output_dir "./models/lora" \
    --batch_size 14 \
    --learning_rate 5e-4
```

#### Full Fine-tuning
```bash
# Run full fine-tuning
python src/training/full_training.py \
    --model_path "unsloth/Llama-3.2-11B-Vision-Instruct" \
    --output_dir "./models/full" \
    --batch_size 14
```

### 5. Evaluation

#### Running Benchmarks
```bash
# Run evaluation script
python src/evaluation/benchmark.py \
    --model_path "./models/full" \
    --test_samples 100 \
    --save_visualizations
```

### 6. Results

The evaluation results are saved in the following locations:
- Metrics: `results/metrics/`
- Visualizations: `results/visualizations/`
- Training logs: `results/logs/`

#### Key Metrics
- IoU (Intersection over Union)
- Pixel-based accuracy
- Point-specific distances

### 7. HPC Integration

#### GPU Node Activation
```bash
# Activate GPU node
sbatch scripts/hpc/gpu_activation.sh
```

#### Remote Jupyter Setup
```bash
# Start remote Jupyter session
bash scripts/hpc/jupyter_remote.sh
```

#### Job Chaining
```bash
# Submit chained jobs
bash scripts/hpc/job_chain.sh
```

### 8. Monitoring and Visualization

#### WandB Integration
```python
import wandb

# Initialize WandB
wandb.init(project="llama_full_training")
```

#### TensorBoard
```bash
# Launch TensorBoard
tensorboard --logdir results/logs
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Professor Qun Wang for guidance and mentorship
- HPC facility for computational resources

## Contact

Your Name - mokietes@sfsu.edu
Project Link: [https://github.com/mokietes/Visual-Data-Mining-AI-Model](https://github.com/mokietes/Visual-Data-Mining-AI-Model)