# Crop Yield Prediction with Fold-Based Multi-Process Training

This repository contains a PyTorch-based pipeline for training and evaluating a ResNet50 model on crop yield prediction from field imagery. The training is done using **5-fold cross-fitting** with **parallel processing**, synchronized through barriers and shared memory for efficient multi-GPU training.

## 📁 Project Structure
```
project/
├── tools/
│   ├── train_parallel.py
│   ├── train_fold_worker.py  
│   └── aggregate.py
├── config/
│   └── config.yaml
├── models/
│   └── resnet.py
├── data/
│   └── dataset.py
├── notebooks/
│   ├── preprocessing.ipynb  
│   └── postprocessing.ipynb
├── environment.yml
├── README.md
```

## 🧠 Key Features

- **Parallel Fold Training:** Trains 5 folds in parallel using `torch.multiprocessing`, with synchronization barriers for clean coordination.
- **Shared Memory Evaluation:** Each fold shares its validation predictions via a shared `dict`, used to compute global and zone-wise R².
- **Automatic Checkpointing:** Model checkpoints are saved only when validation performance (avg zone-wise R²) improves.

## ⚙️ Configuration

The training configuration is specified in `config/config.yaml`.

## 🖥️ Cluster Usage (SLURM)

This project was trained on a SLURM-based compute cluster using **a single SLURM job** on **one node with multiple GPUs**.

The `train_parallel.py` script internally spawns 5 parallel processes using `torch.multiprocessing`, with each process assigned to a different GPU.