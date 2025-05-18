# Datasets README

This directory contains scripts and classes related to dataset acquisition, processing, and loading for the PLayer-FL project.

## Purpose

The goal is to provide a unified way to handle various datasets (image, NLP, tabular) and prepare them for use in the federated learning pipelines. This involves:

1.  **Acquisition/Processing:** Downloading publicly available datasets or processing manually downloaded ones (like MIMIC-III, Sentiment140).
2.  **Formatting:** Converting raw data into a consistent format, potentially generating embeddings for NLP tasks.
3.  **Partitioning:** Splitting data based on natural client divisions (e.g., ISIC centers, MIMIC diagnoses) or simulating non-IID distributions using Dirichlet partitioning for benchmark datasets.
4.  **DataLoader Creation:** Providing standard PyTorch `Dataset` and `DataLoader` objects with appropriate transformations and augmentations for training, validation, and testing.

## Key Files

*   **`dataset_creator.py`**: This is the primary script to run *after* manually acquiring datasets that require it (MIMIC-III, Sentiment140, ISIC, Heart). It performs tasks like:
    *   Downloading benchmark datasets (FMNIST, EMNIST, CIFAR10) via torchvision.
    *   Unzipping the Sentiment140 dataset.
    *   Processing MIMIC-III notes, generating embeddings using ClinicalBERT, and saving data split by diagnosis group.
    *   Processing ISIC metadata and linking it to image paths.
    *   Saving processed data and embeddings into the respective `data/<DatasetName>/` folders.
    *   **Direct Usage:** `python code/datasets/dataset_creator.py [OPTIONS]` (see main README for options).
*   **`dataset_processing.py`**: Contains the core classes for data handling within the FL pipelines:
    *   **`UnifiedDataLoader`**: Loads data processed by `dataset_creator.py` (or directly from torchvision for benchmarks) into a standardized pandas DataFrame format.
    *   **`DataPartitioner`**: Splits the loaded DataFrame data across simulated or natural clients based on dataset type (Dirichlet or natural splits).
    *   **`BaseDataset`**: Abstract base class for PyTorch datasets.
    *   **Dataset-specific classes (`EMNISTDataset`, `CIFARDataset`, ..., `HeartDataset`)**: Inherit from `BaseDataset` and implement data loading and transformations.
    *   **`DataPreprocessor`**: Orchestrates the creation of client-specific `DataLoader` objects.

## Workflow

1.  **Acquire Raw Data:** Download required datasets and place them in the correct subfolders within the main `data/` directory (see main README).
2.  **Run Creator Script:** Execute `python code/datasets/dataset_creator.py` (with appropriate flags if needed) to process the raw data.
3.  **Use in Pipelines:** The evaluation and layer metrics pipelines use `DataPreprocessor` to automatically load and prepare the client data loaders.

## SLURM Job Submission (`submit_dataset_creation.sh`)

Included in this directory is a convenience script `submit_dataset_creation.sh` designed for submitting the `dataset_creator.py` script as a job to a SLURM-managed cluster.

*   **Purpose:** Automates the creation of an SBATCH script and submits it to the SLURM scheduler.
*   **Configuration:** The script allows customization of datasets to process, SLURM resource allocation (memory, time, CPU, GPU), the Conda environment, and project directories via command-line arguments.
*   **Usage (from project root):**
    ```bash
    bash code/datasets/submit_dataset_creation.sh [OPTIONS]
    ```
*   **Key Options:**
    *   `--datasets=<comma-separated-list>`: Specify datasets (e.g., `isic,mimic`). Defaults to all.
    *   `--process-all`: Process all datasets defined in the script defaults.
    *   `--memory=<MEM>`: Set job memory (e.g., `64G`).
    *   `--time=<HH:MM:SS>`: Set job time limit (e.g., `48:00:00`).
    *   `--env-name=<name>`: Specify the Conda environment name.
    *   `--dir=<path>`: Specify the project root directory.
    *   `--help`: Display detailed usage instructions.
*   **Output:** SLURM log files are saved in `code/datasets/logs/outputs/` and `code/datasets/logs/errors/`.