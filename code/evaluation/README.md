# Evaluation Pipeline README

This directory contains the code necessary to run standard federated learning experiments, including hyperparameter tuning and final model evaluation.

## Purpose

The goal of this pipeline is to train and evaluate the performance (e.g., accuracy, F1-score) of the implemented federated learning algorithms (PLayer-FL variants and baselines) on various datasets. It supports a two-phase process:

1.  **Learning Rate Tuning:** Find the optimal learning rate for each algorithm on a given dataset.
2.  **Final Evaluation:** Run multiple independent trials using the best hyperparameters found during tuning to obtain robust performance estimates.

## Key Files

*   **`run.py`**: The main entry point script to launch experiments. Use this script from the command line.
    *   **Direct Usage:**
        ```bash
        # Tune learning rate for FMNIST
        python code/evaluation/run.py --dataset FMNIST --experiment_type learning_rate

        # Run final evaluation for FMNIST (after tuning)
        python code/evaluation/run.py --dataset FMNIST --experiment_type evaluation
        ```
*   **`pipeline.py`**: Defines the `Experiment` class which orchestrates the entire workflow.
*   **`results_processing.py`**: Contains functions and classes (`ResultAnalyzer`) to load and analyze `.pkl` results files.
*   **`losses.py`**: Defines custom loss functions (e.g., `MulticlassFocalLoss`).
*   **`optimizers.py`**: Defines custom optimizers (e.g., `pFedMeOptimizer`).
*   **`performance_logging.py`**: Sets up the logging infrastructure.
*   **`logs/`**: Directory where detailed log files for each experiment run are stored.

## Workflow

1.  **Configure:** Adjust settings in `code/configs.py`.
2.  **Prepare Data:** Run `code/datasets/dataset_creator.py`.
3.  **Tune Hyperparameters (Optional):** Run `code/evaluation/run.py` with `--experiment_type learning_rate`.
4.  **Evaluate:** Run `code/evaluation/run.py` with `--experiment_type evaluation`.
5.  **Analyze Results:** Use `code/evaluation/results_processing.py` to generate summary tables.

## SLURM Job Submission (`submit_evaluation.sh`)

Included in this directory is a convenience script `submit_evaluation.sh` designed for submitting evaluation or learning rate tuning jobs to a SLURM-managed cluster.

*   **Purpose:** Automates the creation and submission of *multiple* SBATCH jobs, typically one for each combination of dataset and experiment type specified. Each job runs `run.py`.
*   **Configuration:** Allows customization of datasets, experiment types, SLURM resources (defaults: 6 CPUs, 40G memory, 30h time), Conda environment, and project directory via command-line arguments.
*   **Usage (from project root):**
    ```bash
    bash code/evaluation/submit_evaluation.sh [OPTIONS]
    ```
*   **Key Options:**
    *   `--datasets=<comma-separated-list>`: Specify datasets (e.g., `FMNIST,CIFAR`). Defaults to all defined.
    *   `--exp-types=<comma-separated-list>`: Specify experiment types (e.g., `learning_rate,evaluation`). Defaults to `evaluation`.
    *   `--env-name=<name>`: Specify the Conda environment name.
    *   `--dir=<path>`: Specify the project root directory.
    *   `--help`: Display detailed usage instructions.
*   **Output:** SLURM log files for each submitted job are saved in `code/evaluation/logs/outputs/` and `code/evaluation/logs/errors/`, named according to the dataset and experiment type (e.g., `FMNIST_evaluation.txt`). Python-specific logs are saved in `code/evaluation/logs/python_logs/`.