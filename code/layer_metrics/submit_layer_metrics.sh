#!/bin/bash

# --- Default Values ---
DEFAULT_DATASETS=("Heart" "FMNIST" "EMNIST" "CIFAR" "Sentiment" "ISIC" "mimic")
DEFAULT_DIR='/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
DEFAULT_ENV_PATH='/gpfs/commons/home/aelhussein/anaconda3/bin/activate'
DEFAULT_ENV_NAME='cuda_env_ne1'

# --- Function to display usage ---
show_usage() {
    echo "Usage: $0 [options]"
    echo "Submit SLURM jobs for the Analytics Pipeline."
    echo ""
    echo "Options:"
    echo "  --datasets=DATASET_LIST   Comma-separated list of datasets (default: all defined)"
    echo "  --env-path=ENV_PATH       Environment activation path (default: $DEFAULT_ENV_PATH)"
    echo "  --env-name=ENV_NAME       Environment name (default: $DEFAULT_ENV_NAME)"
    echo "  --help                    Show this help message"
}

# --- Parse named arguments ---
datasets=() # Initialize as empty array
DIR="${DEFAULT_DIR}"
ENV_PATH="${DEFAULT_ENV_PATH}"
ENV_NAME="${DEFAULT_ENV_NAME}"

while [ $# -gt 0 ]; do
    case "$1" in
        --datasets=*)
            IFS=',' read -ra datasets <<< "${1#*=}"
            ;;
        --env-path=*)
            ENV_PATH="${1#*=}"
            ;;
        --env-name=*)
            ENV_NAME="${1#*=}"
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            show_usage
            exit 1
            ;;
    esac
    shift
done

# --- Use defaults if arrays/variables are empty ---
if [ ${#datasets[@]} -eq 0 ]; then
    datasets=("${DEFAULT_DATASETS[@]}")
fi
# Note: num_runs remains DEFAULT_NUM_RUNS ("") if not specified, Python handles its default

# --- Define Python script path ---
# IMPORTANT: Adjust this path if run_analytics.py is located elsewhere relative to $DIR
PYTHON_SCRIPT_PATH="${DIR}/code/layer_metrics/analytics_run.py"

# --- Create log directories ---
LOG_OUTPUT_DIR="logs/outputs"
LOG_ERROR_DIR="logs/errors"
PYTHON_LOG_DIR="logs/python_logs"
mkdir -p "${LOG_OUTPUT_DIR}" "${LOG_ERROR_DIR}" "${PYTHON_LOG_DIR}"

# --- Echo configuration ---
echo "Running Analytics Pipeline with configuration:"
echo "Datasets: ${datasets[*]}"
echo "Directory: $DIR"
echo "Environment path: $ENV_PATH"
echo "Environment name: $ENV_NAME"
echo "Python Script: $PYTHON_SCRIPT_PATH"
echo

# --- Submit jobs ---
for dataset in "${datasets[@]}"; do
    job_name="${dataset}_analytics"

    # Construct the python command with conditional arguments
    python_command="python ${PYTHON_SCRIPT_PATH} -d ${dataset}"

    echo "Constructed command for ${dataset}: ${python_command}"

    # Create temporary submission script
    cat << EOF > "temp_submit_${job_name}.sh"
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aelhussein@nygenome.org # Change email if needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6      # Adjust based on netrep parallelization needs
#SBATCH --mem=60G              # Increased memory assumption for analytics
#SBATCH --gres=gpu:1           # Analytics might need GPU for model runs/backprop
#SBATCH --time=48:00:00        # Increased time assumption for analytics
#SBATCH --output=${LOG_OUTPUT_DIR}/${job_name}.txt  
#SBATCH --error=${LOG_ERROR_DIR}/${job_name}.txt  

echo "Starting job ${job_name} on \$(hostname)"
echo "SLURM Job ID: \${SLURM_JOB_ID}"
echo "Activation path: ${ENV_PATH}"
echo "Environment name: ${ENV_NAME}"
echo "Python script: ${PYTHON_SCRIPT_PATH}"
echo "Full command: ${python_command}"
echo "----------------------------------------"

# Activate the environment
source "${ENV_PATH}" "${ENV_NAME}"
if [ \$? -ne 0 ]; then
    echo "ERROR: Failed to activate environment ${ENV_NAME}"
    exit 1
fi
echo "Environment activated."

export PYTHONUNBUFFERED=1
export PYTHON_LOG_DIR="${PYTHON_LOG_DIR}" # For performance_logger if it uses this

# Run the Python script
echo "Executing Python script..."
time ${python_command} # Add time for basic timing
EXIT_CODE=\$?
echo "----------------------------------------"
echo "Python script finished with exit code: \${EXIT_CODE}"

exit \${EXIT_CODE}
EOF

    echo "Submitting SLURM job for dataset: ${dataset}"

    sbatch "temp_submit_${job_name}.sh"
    sbatch_exit_code=$?
    if [ $sbatch_exit_code -eq 0 ]; then
        echo "Job submitted successfully."
    else
        echo "ERROR: sbatch submission failed with code ${sbatch_exit_code}."
    fi
    rm "temp_submit_${job_name}.sh" # Clean up temporary script
    sleep 1 # Avoid overwhelming the scheduler
done

echo "All analytics jobs submitted."