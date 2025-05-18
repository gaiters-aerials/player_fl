# PLayer-FL: Principled Layer-wise Personalized Federated Learning

This repository contains the official implementation for the paper **"PLayer-FL: A principled approach to personalized layer-wise cross-silo Federated Learning"**

PLayer-FL introduces a novel approach to personalize federated learning models by adapting specific layers based on client data characteristics. This codebase provides the implementation for PLayer-FL, several baseline federated learning algorithms, and tools for layer-wise analysis.


## Repository Structure
```
├── code/                         # Main codebase
│   ├── clients.py                # Client-side implementations for all FL algorithms
│   ├── configs.py                # Global configuration and hyperparameters
│   ├── datasets/                 # Dataset processing and creation
│   │   ├── dataset_creator.py    # Script to generate datasets used in the paper
│   │   └── dataset_processing.py # Dataset loading and preprocessing utilities
│   ├── evaluation/               # Evaluation framework
│   │   ├── losses.py             # Custom loss functions
│   │   ├── optimizers.py         # Custom optimizers for specific algorithms
│   │   ├── performance_logging.py # Logging utilities
│   │   ├── pipeline.py           # Experiment execution pipeline
│   │   ├── results_processing.py # Results analysis and visualization
│   │   └── run.py                # Main entry point for running evaluations
│   ├── helper.py                 # Utility functions
│   ├── layer_metrics/            # Tools for analyzing layer-wise behavior
│   │   ├── analytics_clients.py  # Client implementations for metrics
│   │   ├── analytics_pipeline.py # Pipeline for running analytics
│   │   ├── analytics_results_processing.py # Process analytics results
│   │   ├── analytics_run.py      # Entry point for running metrics analysis
│   │   ├── analytics_server.py   # Server for analytics collection
│   │   └── layer_analytics.py    # Core layer analysis metrics
│   ├── models.py                 # Neural network architectures
│   └── servers.py                # Server-side implementations for all FL algorithms
└── results/                      # Directory for experiment results
    ├── analytics/                # Layer metrics results
    ├── evaluation/               # Algorithm evaluation results
    └── lr_tuning/                # Learning rate tuning results
```

## Implemented Algorithms
PLayer-FL includes implementations of various federated learning algorithms:
- Local: Local training without federation
- FedAvg: Standard Federated Averaging
- FedProx: Federated Proximal optimization with regularization
- pFedMe: Personalized FL with model optimization
- Ditto: Dual-model personalization approach
- LocalAdaptation: Fine-tuning after federation
- BABU: Body Aggregation, Batch Updating (federated body, local head)
- FedLP: Layer-wise probabilistic model aggregation
- FedLAMA: Layer-wise adaptive model aggregation
- pFedLA: Personalized layer-wise aggregation via hypernetwork
- LayerPFL: Our approach with principled layer selection
- LayerPFL_random: Random layer subset selection variant

## Setup and Installation
### 1. Dependencies
*   Python 3.8+ recommended.
*   Create and activate a virtual environment (e.g., using `conda` or `venv`).
*   Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Dataset Setup
The project requires several datasets that need to be downloaded and placed in a `data` folder. Below are the required datasets and their sources.

### Directory Structure
```
data/
├── FMNIST/       # Fashion-MNIST (downloaded automatically)
├── EMNIST/       # Extended MNIST (downloaded automatically)
├── CIFAR10/      # CIFAR-10 (downloaded automatically)
├── Heart/        # FLamby heart disease data
├── ISIC/         # ISIC skin lesion dataset
├── mimic_iii/    # MIMIC-III clinical notes
└── Sentiment/    # Twitter Sentiment140 dataset
```

### Standard ML Datasets (via PyTorch)
- **Fashion-MNIST (FMNIST)**
  - Available through `torchvision.datasets`
  - Will be downloaded automatically during first run

- **EMNIST**
  - Available through `torchvision.datasets`
  - Will be downloaded automatically during first run

- **CIFAR-10**
  - Available through `torchvision.datasets`
  - Will be downloaded automatically during first run

### Federated Learning Datasets (via FLamby)

- **FED-HEART**
  - **Source**: [FLamby Heart Disease Dataset](https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_heart_disease)
  - **Installation**: Follow FLamby installation instructions
  - **Directory**: `data/Heart/`
  - **Required Files**:
    - `processed.cleveland.data`: Cleveland data
    - `processed.hungarian.data`: Hungarian data
    - `processed.switzerland.data`: Switzerland data
    - `processed.va.data`: VA data

- **FED-ISIC-2019**
  - **Source**: [FLamby ISIC 2019 Dataset](https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_isic2019)
  - **Installation**: Follow FLamby installation instructions
  - **Directory**: `data/ISIC/`
  - **Required Files**:
    - `ISIC_2019_Training_GroundTruth.csv`: Ground truth labels from the ISIC 2019 challenge.
    - `ISIC_2019_Training_Metadata_FL.csv`: Associated metadata.
    - `ISIC_2019_Training_Input_preprocessed/`: Folder containing preprocessed JPEG image files.

---

### Healthcare Datasets

- **MIMIC-III**
  - **Source**: [PhysioNet MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/)
  - **Access**: Requires credentialed approval via PhysioNet
  - **Directory**: `data/mimic_iii/`
  - **Required Files**:
    - `ADMISSIONS.csv`: Admissions table
    - `NOTEEVENTS.csv`: Clinical notes table
    - `DIAGNOSIS_ICD.csv`: Diagnosis table
    - `ICUSTAYS.csv`: Stays table

---

### NLP Datasets

- **Sentiment140 (Sent-140)**
  - **Source**: [Kaggle Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
  - **Directory**: `data/sentiment140/`
  - **Required File**:
    - `data/Sentiment/sentiment.zip`: .

---

#### Access Requirements
- MIMIC-III requires PhysioNet credentialed access
- Other datasets are publicly available

### 3. Dataset preprocessing
After downloading the required datasets, run:

```bash
bash ./code/datasets/submit_dataset_creation.sh [OPTIONS]
```
Available Options:
- `datasets`: Comma-separated list of datasets to process (default: isic,sentiment,mimic,benchmark)
- `process-all`: Flag to process all datasets regardless of individual selection
- `dir`: Root directory (defaults to project location)
- `env-path`: Environment activation path
- `env-name`: Environment name
- `memory`: Memory allocation for job (default: 64G)
- `time`: Time allocation for job (default: 48:00:00)
- `help`: Show help message


This script creates a SLURM submission that handles the dataset creation process, with appropriate resource allocations for these computationally intensive tasks.


## Running Epxeriments
The codebase supports two main types of experiments: Model Evaluation and Layer Metrics.

### Layer Metrics Analysis

This pipeline runs specific algorithms (currently 'local' and 'fedavg' as per analytics_pipeline.py) and collects detailed layer-wise metrics (gradient norms, Hessian-vector products, activation similarity) during the initial and final rounds of training.

Run using the provided SLURM script:
```bash
bash ./code/layer_metrics/submit_layer_metrics.sh [OPTIONS]
```
- `datasets`: Comma-separated list of datasets (default: Heart,FMNIST,EMNIST,CIFAR,Sentiment,ISIC,mimic)
- `env-path`: Environment activation path
- `env-name`: Environment name
- `help`: Show help message

Results (raw metrics per client, per layer, per run) are saved in `results/analytics/<DATASET_NAME>_analytics_results.pkl.`


#### Results processing
 Use the `code/layer_metrics/analytics_results_processing.py` script to generate plots (gradient importance, variance, Hessian SVD sum, activation similarity) from the .pkl files in results/analytics/.

 ```python
# Example
python code/layer_metrics/analytics_results_processing.py --dataset FMNIST --server_type local
python code/layer_metrics/analytics_results_processing.py --dataset FMNIST --server_type fedavg
```

### Model Evaluations

This pipeline trains and evaluates the performance of PLayer-FL and baseline algorithms. It involves two steps:
1. Learning Rate Tuning
2. Evaluation

Both are run using the same bash script `submit_evaluation.sh`. As evaluation automatically selects the best learning rate, learning rate tuning for a dataset must be completed prior to running evalutation

#### Running Model Evaluations

```bash
bash ./code/evaluation/submit_evaluation.sh [OPTIONS]
```
Available Options
- `datasets`: Comma-separated list of datasets to process (default: Heart,FMNIST,EMNIST,CIFAR,Sentiment,ISIC,mimic)
- `exp-types`: Comma-separated list of experiment types (default: evaluation, alternative: learning_rate)
- `dir`: Root directory
- `env-path`: Environment activation path
- `env-name`: Environment name
- `help`: Show help message

Results are saved in `results/lr_tuning/<DATASET_NAME>_lr_tuning.pkl` and `results/evaluation/<DATASET_NAME>_evaluation.pkl`, respectively.

#### Results Processing
Use the `code/evaluation/results_processing.py` script to generate summary tables (median performance with CIs, fairness metrics) from the .pkl files in results/evaluation/.
```python
# Example 
python code/evaluation/results_processing.py
```
This will generate tables and figures summarizing the performance of different algorithms across datasets.

## Configuration

The main configuration file is `code/configs.py`. It defines:

- Default hyperparameters (learning rates, rounds, epochs, batch sizes) per dataset.
- Paths to data and results directories.
- Lists of supported algorithms and datasets.
- Algorithm-specific parameters:
  - Layers to federate (LAYERS_TO_FEDERATE_DICT) for LayerPFL, BABU.
  - Regularization parameters (REG_PARAMS) for FedProx, pFedMe, Ditto.
  - Layer preservation rate (LAYER_PRESERVATION_RATES) for FedLP.
  - FedLAMA parameters (LAMA_RATES).
  - pFedLA HyperNetwork parameters (HYPERNETWORK_PARAMS).

Modify this file to change experiment settings, add new datasets, or adjust algorithm parameters.
