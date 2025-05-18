#!/bin/bash

# Default values
DEFAULT_DATASETS=("Heart" "FMNIST" "EMNIST" "CIFAR" "Sentiment" "ISIC" "mimic")
DEFAULT_EXP_TYPES=("evaluation")
DEFAULT_DIR='/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
DEFAULT_ENV_PATH='/gpfs/commons/home/aelhussein/anaconda3/bin/activate'
DEFAULT_ENV_NAME='cuda_env_ne1'

# Function to display usage
show_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --datasets    Comma-separated list of datasets (default: Heart,FMNIST,EMNIST,CIFAR,Sentiment,ISIC,mimic)"
    echo "  --exp-types   Comma-separated list of experiment types (default: evaluation)"
    echo "  --dir        Root directory (default: $DEFAULT_DIR)"
    echo "  --env-path   Environment activation path (default: $DEFAULT_ENV_PATH)"
    echo "  --env-name   Environment name (default: $DEFAULT_ENV_NAME)"
    echo "  --help       Show this help message"
}

# Parse named arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --datasets=*)
            IFS=',' read -ra datasets <<< "${1#*=}"
            ;;
        --exp-types=*)
            IFS=',' read -ra experiment_types <<< "${1#*=}"
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
experiment_types=("${experiment_types[@]:-${DEFAULT_EXP_TYPES[@]}}")
DIR="${DIR:-$DEFAULT_DIR}"
ENV_PATH="${ENV_PATH:-$DEFAULT_ENV_PATH}"
ENV_NAME="${ENV_NAME:-$DEFAULT_ENV_NAME}"

# Create log directories
mkdir -p logs/outputs logs/errors

# Echo configuration
echo "Running with configuration:"
echo "Datasets: ${datasets[*]}"
echo "Experiment types: ${experiment_types[*]}"
echo "Directory: $DIR"
echo "Environment path: $ENV_PATH"
echo "Environment name: $ENV_NAME"
echo

# Submit jobs
for dataset in "${datasets[@]}"; do
    for exp_type in "${experiment_types[@]}"; do
        job_name="${dataset}_${exp_type}"
        cat << EOF > temp_submit_${job_name}.sh
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aelhussein@nygenome.org
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --output=logs/outputs/${job_name}.txt
#SBATCH --error=logs/errors/${job_name}.txt

# Activate the environment
source ${ENV_PATH} ${ENV_NAME}

export PYTHONUNBUFFERED=1
export PYTHON_LOG_DIR="logs/python_logs"
# Run the Python script
python ${DIR}/code/evaluation/run.py -ds ${dataset} -exp ${exp_type}
EOF

        echo "Submitted job for dataset: ${dataset} and experiment type: ${exp_type}"

        sbatch temp_submit_${job_name}.sh
        rm temp_submit_${job_name}.sh
        sleep 1
    done
done