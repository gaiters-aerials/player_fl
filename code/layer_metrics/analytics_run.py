import argparse
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # -> ./layer_pfl/code/layer_metrics
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR) # -> (...)/layer_pfl/code
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _CURRENT_DIR)

from analytics_pipeline import AnalyticsConfig, AnalyticsExperiment
from configs import *
from helper import get_parameters_for_dataset # To get default runs

def main():
    parser = argparse.ArgumentParser(description="Run Federated Learning Analytics Pipeline")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="Name of the dataset (e.g., CIFAR, FMNIST).")

    args = parser.parse_args()
    num_runs = get_parameters_for_dataset(args.dataset)['runs_layers']

    # Create configuration object using the analytics-specific config class
    config = AnalyticsConfig(
        dataset=args.dataset,
        num_runs=num_runs,
        results_dir=RESULTS_DIR
    )

    # Instantiate and run the experiment using the refactored class
    experiment = AnalyticsExperiment(config)
    experiment.run_experiment()

    print(f"\nAnalytics pipeline finished for dataset: {args.dataset}.")

if __name__ == "__main__":
    main()