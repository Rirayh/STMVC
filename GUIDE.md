# Experiment Guide for SUMVC

This repository contains the official PyTorch implementation for the paper: **"Enhancing Multi-View Clustering: A Sufficient Information-Theoretic Approach for Consistency Acquisition and Redundancy Elimination"**.

Ensure your experimental environment meets the following requirements:

- Python 3.8+
- PyTorch, NumPy, SciPy, Scikit-learn, Matplotlib, Tqdm, h5py

The codebase is organized into dedicated scripts for different experimental settings:

- `train_MVC.py`: The training script for standard Multi-View Clustering.
- `train_IMVC.py`: The training script for Incomplete Multi-View Clustering.
- `utils.py`: Contains utility functions, dataset loaders, and evaluation metrics.
- `data/`: The directory where all dataset files (`.mat` format) should be placed.
- `results/`: The directory where trained model weights (`.pkl`) will be saved.
- `logs/`: The directory where training logs will be recorded.

## 1. MVC

### Supported Datasets

The script  supports the following datasets:

- `Multi-COIL-20`
- `Multi-COIL-10`
- `Multi-Mnist`
- `Multi-FMnist`

```
# Example 1: Run on Multi-COIL-20
python train_MVC.py --dataset Multi-COIL-20

# Example 2: Run on Multi-Mnist
python train_MVC.py --dataset Multi-Mnist

# Example 3: Run on Multi-COIL-10
python train_MVC.py --dataset Multi-COIL-10
```

## 2. IMVC

### Supported Datasets

The script  supports the following datasets:

- `resized_NoisyMNIST`
- `Multi-COIL-10`
- `Multi-COIL-20`

### Arguments

- `--dataset`: The name of the dataset to evaluate.
- `--missing_rate`: The ratio of missing views (e.g., `0.1`, `0.3`, `0.5`, `0.7`).
- `--gpu`: Specifies the CUDA device ID to use (default is `0`).

### Running the IMVC Experiments


**For resized_NoisyMNIST:**

```
python train_IMVC.py --dataset resized_NoisyMNIST --missing_rate 0.1 --gpu 0
python train_IMVC.py --dataset resized_NoisyMNIST --missing_rate 0.3 --gpu 0
python train_IMVC.py --dataset resized_NoisyMNIST --missing_rate 0.5 --gpu 0
python train_IMVC.py --dataset resized_NoisyMNIST --missing_rate 0.7 --gpu 0
```

**For Multi-COIL-10:**

```
python train_IMVC.py --dataset Multi-COIL-10 --missing_rate 0.1 --gpu 0
python train_IMVC.py --dataset Multi-COIL-10 --missing_rate 0.3 --gpu 0
python train_IMVC.py --dataset Multi-COIL-10 --missing_rate 0.5 --gpu 0
python train_IMVC.py --dataset Multi-COIL-10 --missing_rate 0.7 --gpu 0
```

**For Multi-COIL-20:**

```
python train_IMVC.py --dataset Multi-COIL-20 --missing_rate 0.1 --gpu 0
python train_IMVC.py --dataset Multi-COIL-20 --missing_rate 0.3 --gpu 0
python train_IMVC.py --dataset Multi-COIL-20 --missing_rate 0.5 --gpu 0
python train_IMVC.py --dataset Multi-COIL-20 --missing_rate 0.7 --gpu 0
```

