# Layer Metrics README

This directory contains the code for performing detailed layer-wise analysis of models during federated learning.

## Purpose

The goal of this pipeline is to gain insights into the internal workings of models trained with federated learning, specifically focusing on layer behavior. It calculates metrics such as:

*   Gradient norms and variance per layer.
*   Hessian properties (approximated via Hessian-vector products) like dominant eigenvalues (SVD).
*   Activation similarity between clients using techniques like Centered Kernel Alignment (CKA) via the `netrep` library.

This analysis helps understand how different layers adapt during federated training and how personalization strategies affect layer representations.

## Key Files

*   **`analytics_run.py`**: The main entry point script to launch layer analytics experiments.
    *   **Direct Usage:**
        ```bash
        python code/layer_metrics/analytics_run.py --dataset FMNIST
        ```
*   **`analytics_pipeline.py`**: Defines the `AnalyticsExperiment` class which orchestrates the analytics workflow.
*   **`layer_analytics.py`**: Contains the core functions for calculating the layer-wise metrics.
*   **`analytics_server.py`**: Defines specialized server classes with hooks for metric collection.
*   **`analytics_clients.py`**: Defines a specialized `AnalyticsClient` that performs local metric calculations.
*   **`analytics_results_processing.py`**: Loads and processes analytics results, generating plots.
*   **`logs/`**: Directory where detailed log files for each analytics run are stored.

## Workflow

1.  **Configure:** Adjust settings in `code/configs.py`. Note: algorithms analyzed are currently set in `analytics_pipeline.py`.
2.  **Prepare Data:** Run `code/datasets/dataset_creator.py`.
3.  **Run Analytics:** Execute `code/layer_metrics/analytics_run.py` with the desired dataset.
4.  **Analyze Results:** Use `code/layer_metrics/analytics_results_processing.py` to generate plots from the saved `.pkl` files in `results/analytics/`.

## SLURM Job Submission (`submit_layer_metrics.sh`)

Included in this directory is a convenience script `submit_layer_metrics.sh` designed for submitting layer analytics jobs to a SLURM-managed cluster.

*   **Purpose:** Automates the creation and submission of SBATCH jobs, one for each specified dataset, running `analytics_run.py`.
*   **Configuration:** Allows customization of datasets, SLURM resources (defaults: 6 CPUs, 60G memory, 48h time), Conda environment, and project directory via command-line arguments.
*   **Usage (from project root):**
    ```bash
    bash code/layer_metrics/submit_layer_metrics.sh [OPTIONS]
    ```
*   **Key Options:**
    *   `--datasets=<comma-separated-list>`: Specify datasets (e.g., `FMNIST,Heart`). Defaults to all defined.
    *   `--env-name=<name>`: Specify the Conda environment name.
    *   `--dir=<path>`: Specify the project root directory.
    *   `--help`: Display detailed usage instructions.
*   **Output:** SLURM log files for each submitted job are saved in `code/layer_metrics/logs/outputs/` and `code/layer_metrics/logs/errors/`, named according to the dataset (e.g., `FMNIST_analytics.txt`). Python-specific logs are saved in `code/layer_metrics/logs/python_logs/`.