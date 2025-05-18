#!/bin/bash

# Default values
DEFAULT_DATASETS=("isic" "sentiment" "mimic" "benchmark")
DEFAULT_DIR='/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
DEFAULT_ENV_PATH='/gpfs/commons/home/aelhussein/anaconda3/bin/activate'
DEFAULT_ENV_NAME='cuda_env_ne1'
DEFAULT_MEMORY='64G'  # More memory for dataset creation
DEFAULT_TIME='48:00:00'  # Longer time for dataset processing

# Function to display usage
show_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --datasets     Comma-separated list of datasets to process (default: isic,sentiment,mimic,benchmark)"
    echo "  --process-all  Flag to process all datasets regardless of individual selection"
    echo "  --dir          Root directory (default: $DEFAULT_DIR)"
    echo "  --env-path     Environment activation path (default: $DEFAULT_ENV_PATH)"
    echo "  --env-name     Environment name (default: $DEFAULT_ENV_NAME)"
    echo "  --memory       Memory allocation for job (default: $DEFAULT_MEMORY)"
    echo "  --time         Time allocation for job (default: $DEFAULT_TIME)"
    echo "  --help         Show this help message"
}

# Parse named arguments
PROCESS_ALL=false
while [ $# -gt 0 ]; do
    case "$1" in
        --datasets=*)
            IFS=',' read -ra datasets <<< "${1#*=}"
            ;;
        --process-all)
            PROCESS_ALL=true
            ;;
        --dir=*)
            DIR="${1#*=}"
            ;;
        --env-path=*)
            ENV_PATH="${1#*=}"
            ;;
        --env-name=*)
            ENV_NAME="${1#*=}"
            ;;
        --memory=*)
            MEMORY="${1#*=}"
            ;;
        --time=*)
            TIME="${1#*=}"
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

# Use defaults if not specified
datasets=("${datasets[@]:-${DEFAULT_DATASETS[@]}}")
DIR="${DIR:-$DEFAULT_DIR}"
ENV_PATH="${ENV_PATH:-$DEFAULT_ENV_PATH}"
ENV_NAME="${ENV_NAME:-$DEFAULT_ENV_NAME}"
MEMORY="${MEMORY:-$DEFAULT_MEMORY}"
TIME="${TIME:-$DEFAULT_TIME}"

# Create log directories
mkdir -p logs/outputs logs/errors

# Echo configuration
echo "Running with configuration:"
echo "Datasets to process: ${datasets[*]}"
echo "Process all datasets: $PROCESS_ALL"
echo "Directory: $DIR"
echo "Environment path: $ENV_PATH"
echo "Environment name: $ENV_NAME"
echo "Memory allocation: $MEMORY"
echo "Time allocation: $TIME"
echo

# Generate dataset flags
DATASET_FLAGS=""
if [ "$PROCESS_ALL" = true ]; then
    DATASET_FLAGS=""  # No flags will process all datasets by default
else
    for dataset in "${datasets[@]}"; do
        DATASET_FLAGS="$DATASET_FLAGS --$dataset"
    done
fi

# Set job name
if [ "$PROCESS_ALL" = true ]; then
    job_name="all_datasets_creation"
else
    job_name="dataset_creation_$(IFS=_; echo "${datasets[*]}")"
fi

# Create submission script
cat << EOF > temp_submit_dataset_creation.sh
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aelhussein@nygenome.org
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=${MEMORY}
#SBATCH --gres=gpu:1
#SBATCH --time=${TIME}
#SBATCH --output=logs/outputs/${job_name}.txt
#SBATCH --error=logs/errors/${job_name}.txt

# Activate the environment
source ${ENV_PATH} ${ENV_NAME}
conda activate ${ENV_NAME}

# Run the Python script
echo "Starting dataset creation process at \$(date)"
echo "Processing datasets with flags: ${DATASET_FLAGS}"

# Execute the dataset creator script
module load cuda
python ${DIR}/code/datasets/dataset_creator.py ${DATASET_FLAGS}

echo "Dataset creation process completed at \$(date)"
EOF

echo "Submitting job for dataset creation: ${job_name}"
sbatch temp_submit_dataset_creation.sh
rm temp_submit_dataset_creation.sh