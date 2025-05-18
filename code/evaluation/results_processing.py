"""
Tools for loading, analyzing, and formatting experimental results.

Includes:
- Functions to load results from pickle files (`load_lr_results`, `load_eval_results`).
- `bootstrap_ci`: Calculates bootstrap confidence intervals for medians.
- `ResultAnalyzer`: A class to perform statistical analysis (median, CI, variance)
  on results across datasets and algorithms, including fairness metrics.
- `analyze_experiment_results`: Top-level function to orchestrate the analysis
  and formatting of results into tables.
"""
from configs import RESULTS_DIR, ALGORITHMS, pickle, np, pd, sys, stats
from typing import Dict, Optional, List, Tuple, Union

def load_lr_results(DATASET: str) -> Optional[Dict]:
    """
    Loads learning rate tuning results from the corresponding pickle file.

    Args:
        DATASET (str): The name of the dataset whose LR tuning results to load.

    Returns:
        Optional[Dict]: The loaded results dictionary, or None if the file is not found
                        or cannot be loaded.
    """
    filepath = f'{RESULTS_DIR}/lr_tuning/{DATASET}_lr_tuning.pkl' # Adjusted path
    try:
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        print(f"Successfully loaded LR tuning results from: {filepath}")
        return results
    except FileNotFoundError:
        print(f"Error: LR tuning results file not found at: {filepath}")
        return None
    except (pickle.UnpicklingError, EOFError) as e:
        print(f"Error loading or unpickling LR results from {filepath}: {e}")
        return None

def get_lr_results(DATASET: str) -> Optional[pd.DataFrame]:
    """
    Analyzes loaded learning rate tuning results to find the best round and performance
    for each algorithm at each tested learning rate.

    Calculates the mean loss across runs for each round, finds the round with the
    minimum mean loss, and reports the metrics (mean accuracy, mean F1) from that best round.

    Args:
        DATASET (str): The name of the dataset whose LR tuning results to analyze.

    Returns:
        Optional[pd.DataFrame]: A DataFrame summarizing the best performance for each
                                algorithm-LR combination, sorted by the best median loss achieved.
                                Returns None if results cannot be loaded.
    """
    results = load_lr_results(DATASET)
    if results is None:
        return None

    results_output = []
    # Structure: results = {lr1: {'algoA': {'global': {'losses': [[run1_r1,...], [run2_r1,...]], 'scores': [...]}}}}
    for lr, algorithms_data in results.items():
        for algo, metrics_data in algorithms_data.items():
            # Check if required keys exist
            if 'global' not in metrics_data or 'losses' not in metrics_data['global'] or 'scores' not in metrics_data['global']:
                print(f"Warning: Skipping LR={lr}, Algo={algo} due to missing 'global' metrics.")
                continue
            if not metrics_data['global']['losses'] or not metrics_data['global']['scores']:
                 print(f"Warning: Skipping LR={lr}, Algo={algo} due to empty 'losses' or 'scores' list.")
                 continue

            global_metrics = metrics_data['global']
            losses_all_runs = global_metrics['losses'] # [[run1_r1,...], [run2_r1,...]]
            scores_all_runs = global_metrics['scores'] # [[run1_r1_dict,...], [run2_r1_dict,...]]

            try:
                num_runs = len(losses_all_runs)
                num_rounds = len(losses_all_runs[0])
                # Verify consistency
                if not all(len(run) == num_rounds for run in losses_all_runs) or \
                   not all(len(run) == num_rounds for run in scores_all_runs) or \
                   len(scores_all_runs) != num_runs:
                    print(f"Warning: Inconsistent number of runs/rounds for LR={lr}, Algo={algo}. Skipping.")
                    continue
            except IndexError:
                print(f"Warning: Empty run data for LR={lr}, Algo={algo}. Skipping.")
                continue

            mean_losses_per_round = []
            for round_idx in range(num_rounds):
                # Get losses for this round across all runs
                try:
                    round_losses = [losses_all_runs[run_idx][round_idx] for run_idx in range(num_runs)]
                    mean_loss = np.mean(round_losses)
                    mean_losses_per_round.append((round_idx, mean_loss))
                except IndexError:
                     print(f"Warning: IndexError accessing losses at round {round_idx} for LR={lr}, Algo={algo}. Skipping round.")
                     continue # Skip this round if data is incomplete

            if not mean_losses_per_round:
                 print(f"Warning: No valid rounds found for LR={lr}, Algo={algo}. Skipping combo.")
                 continue

            # Find the round index with the minimum mean loss
            best_round_idx, best_mean_loss = min(mean_losses_per_round, key=lambda item: item[1])

            # Get metrics (accuracy, F1) from that specific best round across all runs
            try:
                accuracies_at_best_round = [scores_all_runs[run_idx][best_round_idx]['accuracy'] for run_idx in range(num_runs)]
                f1_scores_at_best_round = [scores_all_runs[run_idx][best_round_idx]['f1_macro'] for run_idx in range(num_runs)]
            except (IndexError, KeyError) as e:
                 print(f"Warning: Could not extract metrics at best round {best_round_idx} for LR={lr}, Algo={algo}. Error: {e}. Skipping combo.")
                 continue

            # Calculate mean of metrics at the best round
            mean_accuracy = np.mean(accuracies_at_best_round)
            mean_f1 = np.mean(f1_scores_at_best_round)

            results_output.append({
                'learning_rate': lr,
                'algorithm': algo,
                'best_round': best_round_idx,
                'mean_loss_at_best_round': best_mean_loss, 
                'mean_accuracy_at_best_round': mean_accuracy, 
                'mean_f1_at_best_round': mean_f1
            })

    if not results_output:
        print("No results were processed successfully.")
        return None

    df_results = pd.DataFrame(results_output)
    # Find the row corresponding to the minimum mean loss for each algorithm
    best_lr_indices = df_results.loc[df_results.groupby('algorithm')['mean_loss_at_best_round'].idxmin()]
    # Sort these best rows by loss
    return best_lr_indices.sort_values(by='mean_loss_at_best_round', ascending=True)


def load_eval_results(DATASET: str) -> Optional[Dict]:
    """
    Loads final evaluation results from the corresponding pickle file.

    Args:
        DATASET (str): The name of the dataset whose evaluation results to load.

    Returns:
        Optional[Dict]: The loaded results dictionary, or None if the file is not found
                        or cannot be loaded.
    """
    filepath = f'{RESULTS_DIR}/evaluation/{DATASET}_evaluation.pkl' # Adjusted path
    try:
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        print(f"Successfully loaded evaluation results from: {filepath}")
        return results
    except FileNotFoundError:
        print(f"Error: Evaluation results file not found at: {filepath}")
        return None
    except (pickle.UnpicklingError, EOFError) as e:
        print(f"Error loading or unpickling evaluation results from {filepath}: {e}")
        return None

def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000,
                 confidence: float = 0.95, min_samples: int = 5) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for the median using vectorized operations.

    Generates bootstrap samples by resampling with replacement, calculates the
    median for each sample, and determines the confidence interval from the
    percentiles of the bootstrap median distribution.

    Args:
        data (np.ndarray): 1D array of sample data.
        n_bootstrap (int): Number of bootstrap samples to generate. Defaults to 1000.
        confidence (float): Confidence level (e.g., 0.95 for 95% CI). Defaults to 0.95.
        min_samples (int): Minimum number of samples required in the data. Defaults to 5.

    Returns:
        Tuple[float, float]: A tuple containing the lower and upper bounds of the
                             confidence interval. Returns (median, median) if CI
                             cannot be computed (e.g., too few samples).

    Raises:
        ValueError: If the number of samples is less than `min_samples`.
    """
    if len(data) < min_samples:
        # Instead of raising ValueError, return median as both bounds for robustness in tables
        median_val = np.median(data) if len(data) > 0 else np.nan
        print(f"Warning: Need at least {min_samples} samples for bootstrap CI, got {len(data)}. Returning median ({median_val}) as bounds.")
        return (median_val, median_val)
        # raise ValueError(f"Need at least {min_samples} samples for bootstrap CI, got {len(data)}")

    size = len(data)
    # Use default_rng for better random number generation
    rng = np.random.default_rng()

    # Generate indices for bootstrap samples: shape (n_bootstrap, size)
    indices = rng.integers(0, size, size=(n_bootstrap, size))
    # Create bootstrap samples using fancy indexing: shape (n_bootstrap, size)
    bootstrap_samples = data[indices]
    # Calculate median along the sample dimension (axis=1): shape (n_bootstrap,)
    bootstrap_medians = np.median(bootstrap_samples, axis=1)

    # Calculate percentiles for the confidence interval
    lower_percentile = (1.0 - confidence) / 2.0 * 100
    upper_percentile = (1.0 - (1.0 - confidence) / 2.0) * 100

    # Compute the percentile values from the bootstrap median distribution
    ci_lower, ci_upper = np.percentile(bootstrap_medians, [lower_percentile, upper_percentile])

    return ci_lower, ci_upper


class ResultAnalyzer:
    """
    Analyzes processed experimental results (loaded from pickle files).

    Provides methods to calculate summary statistics (median, CI) for global metrics,
    analyze fairness metrics (variance across clients, percentage of clients
    outperforming baselines), and format these analyses into pandas DataFrames
    and presentation-ready tables.
    """

    def __init__(self, all_dataset_results: Dict[str, Dict]):
        """
        Initializes the ResultAnalyzer with results loaded for multiple datasets.

        Args:
            all_dataset_results (Dict[str, Dict]): A dictionary where keys are dataset names
                                                   and values are the corresponding loaded
                                                   results dictionaries (typically from evaluation runs).
                                                   Example: {'FMNIST': eval_results_fmnist, 'CIFAR': eval_results_cifar}
        """
        if not all_dataset_results:
             raise ValueError("Input `all_dataset_results` dictionary cannot be empty.")
        self.all_dataset_results = all_dataset_results

        # Infer the list of metrics from the structure of the first valid result entry
        try:
            # Navigate through the nested structure to find a sample metric dictionary
            first_dataset_name = next(iter(all_dataset_results))
            first_dataset_data = all_dataset_results[first_dataset_name]
            first_run_key = next(iter(first_dataset_data)) # e.g., 'run_1'
            first_algo_key = next(iter(first_dataset_data[first_run_key])) # e.g., 'fedavg'
            first_scores_list = first_dataset_data[first_run_key][first_algo_key]['global']['scores']
            first_run_final_scores_dict = first_scores_list[0][0] # Get scores from the first run
            self.metrics = list(first_run_final_scores_dict.keys()) + ['loss'] # Add 'loss' manually
            print(f"Inferred metrics: {self.metrics}")
        except (StopIteration, KeyError, IndexError, TypeError) as e:
            print(f"Error inferring metrics from results structure: {e}")
            # Fallback or default metrics list
            self.metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted', 'mcc', 'loss']
            print(f"Using default metrics list: {self.metrics}")


    def analyze_dataset(self, results: Dict) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Analyzes global performance metrics for a single dataset's results.

        Calculates the median and bootstrap confidence interval for each metric
        (including loss) across all runs for each algorithm.

        Args:
            results (Dict): The loaded results dictionary for one dataset.
                            Expected structure includes runs as top-level keys, then algorithms.
                            e.g., {'run_1': {'fedavg': {'global':...}}, 'run_2':{...}}

        Returns:
            Dict[str, Dict[str, Dict[str, float]]]: A nested dictionary:
                {algorithm: {metric_name: {'median': X, 'ci_lower': Y, 'ci_upper': Z}}}
        """
        # --- Step 1: Aggregate metrics across runs for each algorithm ---
        aggregated_metrics = {}
        # Expected structure: results = {'run_1': {'algoA': {'global':...}, 'algoB':...}, 'run_2': ...}

        # Find all algorithms present across runs
        algorithms = set()
        for run_key in results:
            algorithms.update(results[run_key].keys())
        algorithms = list(algorithms)

        for algo in algorithms:
            aggregated_metrics[algo] = {metric: [] for metric in self.metrics} # Initialize lists for each metric

            for run_key in results:
                if algo in results[run_key]:
                    run_data = results[run_key][algo]
                    if 'global' not in run_data or 'scores' not in run_data['global'] or 'losses' not in run_data['global']:
                        print(f"Warning: Missing global data for algo '{algo}' in run '{run_key}'. Skipping run.")
                        continue

                    try:
                        # Handle the new structure: 'scores' -> List with 1 items -> List with 1 items -> Dict 
                        # Extract final score for this run
                        final_score_dict = None
                        if run_data['global']['scores'] and isinstance(run_data['global']['scores'][0], list) and run_data['global']['scores'][0]:
                            final_score_dict = run_data['global']['scores'][0][0]
                        
                        # Handle the new structure: 'losses' -> List with 1 items -> List with 1 items -> float
                        # Extract final loss for this run
                        final_loss = None
                        if run_data['global']['losses'] and isinstance(run_data['global']['losses'][0], list) and run_data['global']['losses'][0]:
                            final_loss = run_data['global']['losses'][0][0]

                        if final_score_dict is not None:
                            for metric_name in self.metrics[:-1]: # Exclude 'loss' here
                                if metric_name in final_score_dict:
                                    aggregated_metrics[algo][metric_name].append(final_score_dict[metric_name])
                                else:
                                    print(f"Warning: Metric '{metric_name}' not found in scores for algo '{algo}', run '{run_key}'.")
                        else:
                            print(f"Warning: Missing final scores for algo '{algo}', run '{run_key}'.")

                        if final_loss is not None:
                            aggregated_metrics[algo]['loss'].append(final_loss)
                        else:
                            print(f"Warning: Missing final loss for algo '{algo}', run '{run_key}'.")
                    
                    except (IndexError, TypeError) as e:
                        print(f"Error extracting scores/losses for algo '{algo}', run '{run_key}': {e}")
                        continue

        # --- Step 2: Calculate statistics (median, CI) for each algorithm and metric ---
        analysis_output = {}
        for algo, metrics_values in aggregated_metrics.items():
            analysis_output[algo] = {}
            for metric, values_list in metrics_values.items():
                if not values_list:
                    print(f"Warning: No data found for metric '{metric}' for algorithm '{algo}'. Setting stats to NaN.")
                    analysis_output[algo][metric] = {'median': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}
                    continue

                scores_np = np.array(values_list)
                median = np.median(scores_np)
                try:
                    ci_lower, ci_upper = bootstrap_ci(scores_np)
                    analysis_output[algo][metric] = {
                        'median': median,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper
                    }
                except ValueError as e:
                    # Handle cases where bootstrap_ci fails (e.g., < min_samples)
                    print(f"Warning: Could not compute CI for {algo}, {metric}: {e}. Using median as bounds.")
                    analysis_output[algo][metric] = {
                        'median': median,
                        'ci_lower': median, # Use median as CI bounds if calculation fails
                        'ci_upper': median
                    }

        return analysis_output

    def _get_client_metrics(self, results_all_runs: Dict, algorithm: str, client_id: str) -> Dict[str, float]:
        """
        Extracts the median metric values for a specific client across all runs.

        Args:
            results_all_runs (Dict): The loaded results dictionary for a dataset (containing multiple runs).
            algorithm (str): The algorithm identifier.
            client_id (str): The client identifier.

        Returns:
            Dict[str, float]: Dictionary mapping metric names to their median value
                            across runs for the specified client and algorithm.
                            Returns NaNs if data is missing.
        """
        client_metrics_all_runs = {metric: [] for metric in self.metrics}

        for run_key in results_all_runs:
            if algorithm in results_all_runs[run_key]:
                algo_data = results_all_runs[run_key][algorithm]
                if 'sites' in algo_data and client_id in algo_data['sites']:
                    client_run_data = algo_data['sites'][client_id]

                    try:
                        # Handle the new structure: 'scores' -> List with 1 items -> List with 1 items -> Dict
                        # Extract final score for this client in this run
                        final_score_dict = None
                        if client_run_data.get('scores') and isinstance(client_run_data['scores'][0], list) and client_run_data['scores'][0]:
                            final_score_dict = client_run_data['scores'][0][0]
                        
                        # Handle the new structure: 'losses' -> List with 1 items -> List with 1 items -> float
                        # Extract final loss for this client in this run
                        final_loss = None
                        if client_run_data.get('losses') and isinstance(client_run_data['losses'][0], list) and client_run_data['losses'][0]:
                            final_loss = client_run_data['losses'][0][0]

                        if final_score_dict:
                            for metric_name in self.metrics[:-1]:
                                if metric_name in final_score_dict:
                                    client_metrics_all_runs[metric_name].append(final_score_dict[metric_name])
                        
                        if final_loss is not None:
                            client_metrics_all_runs['loss'].append(final_loss)
                    
                    except (IndexError, TypeError) as e:
                        print(f"Error extracting client scores/losses for {client_id}, {algorithm}, {run_key}: {e}")
                        continue

        # Calculate median for each metric across runs
        median_metrics = {}
        for metric, values in client_metrics_all_runs.items():
            if values:
                median_metrics[metric] = np.median(values)
            else:
                median_metrics[metric] = np.nan # Use NaN if no data collected

        return median_metrics

    def analyze_fairness(self, results_all_runs: Dict) -> pd.DataFrame:
        """
        Analyzes fairness metrics for a single dataset across multiple runs.

        Calculates:
        1. Variance: The variance of the median client performance for each metric.
           Lower variance suggests higher fairness (more uniform performance).
        2. Pct_Better: The percentage of clients whose median performance with a given
           algorithm is better than their median performance with *both* 'local'
           and 'fedavg' baselines.

        Args:
            results_all_runs (Dict): The loaded results dictionary for one dataset,
                                     containing multiple runs.

        Returns:
            pd.DataFrame: DataFrame summarizing fairness metrics (Variance, Pct_Better)
                          for each algorithm (excluding baselines) and metric.
        """
        # Identify all algorithms and clients present across runs
        algorithms = set()
        client_ids = set()
        for run_key in results_all_runs:
             for algo, algo_data in results_all_runs[run_key].items():
                 algorithms.add(algo)
                 if 'sites' in algo_data:
                     client_ids.update(algo_data['sites'].keys())
        algorithms = list(algorithms)
        client_ids = list(client_ids)

        if 'local' not in algorithms or 'fedavg' not in algorithms:
            print("Warning: 'local' or 'fedavg' baseline results missing. Cannot calculate Pct_Better fairness metric.")
            # Return empty dataframe or dataframe with only variance
            # return pd.DataFrame(columns=['Algorithm', 'Metric', 'Variance', 'Pct_Better'])

        # Algorithms to compare against baselines
        pfl_algorithms = [algo for algo in algorithms if algo not in ['local', 'fedavg']]

        # --- Get Median Client Metrics for Baselines and PFL Algorithms ---
        all_median_client_metrics = {} # Structure: {algo: {client_id: {metric: median_value}}}
        for algo in algorithms:
            all_median_client_metrics[algo] = {}
            for client_id in client_ids:
                 # Calculate median metrics across runs for this client/algo combo
                 all_median_client_metrics[algo][client_id] = self._get_client_metrics(
                     results_all_runs, algo, client_id
                 )

        # Store baseline metrics separately for easier access
        baselines = {}
        if 'local' in all_median_client_metrics:
             baselines['local'] = all_median_client_metrics['local']
        if 'fedavg' in all_median_client_metrics:
             baselines['fedavg'] = all_median_client_metrics['fedavg']


        # --- Calculate Fairness Metrics ---
        fairness_data_list = []
        for algo in pfl_algorithms:
             if algo not in all_median_client_metrics: continue # Skip if algo had no results

             algo_client_medians = all_median_client_metrics[algo] # {client_id: {metric: median}}

             for metric in self.metrics:
                 # Calculate Variance across clients for this algo/metric
                 median_metric_values = np.array([
                     client_metrics[metric] for client_metrics in algo_client_medians.values() if metric in client_metrics and not np.isnan(client_metrics[metric])
                 ])
                 variance = np.var(median_metric_values) if len(median_metric_values) > 0 else np.nan

                 # Calculate Pct_Better compared to baselines
                 better_count = 0
                 valid_clients_for_comparison = 0
                 if 'local' in baselines and 'fedavg' in baselines: # Check if baselines exist
                     for client_id in client_ids:
                         if client_id in algo_client_medians and \
                            client_id in baselines['local'] and \
                            client_id in baselines['fedavg']:

                             algo_metric = algo_client_medians[client_id].get(metric)
                             local_metric = baselines['local'][client_id].get(metric)
                             fedavg_metric = baselines['fedavg'][client_id].get(metric)

                             # Ensure all metrics are valid numbers for comparison
                             if algo_metric is not None and not np.isnan(algo_metric) and \
                                local_metric is not None and not np.isnan(local_metric) and \
                                fedavg_metric is not None and not np.isnan(fedavg_metric):

                                 valid_clients_for_comparison += 1
                                 is_loss = metric == 'loss'
                                 # Check if algo performs better than *both* baselines
                                 if is_loss:
                                     if algo_metric < local_metric and algo_metric < fedavg_metric:
                                         better_count += 1
                                 else: # Higher is better for other metrics
                                     if algo_metric > local_metric and algo_metric > fedavg_metric:
                                         better_count += 1
                     pct_better = (better_count / valid_clients_for_comparison * 100) if valid_clients_for_comparison > 0 else 0.0
                 else:
                     pct_better = np.nan # Cannot calculate if baselines missing


                 fairness_data_list.append({
                     'Algorithm': algo,
                     'Metric': metric,
                     'Variance': variance,
                     'Pct_Better': pct_better
                 })

        return pd.DataFrame(fairness_data_list)


    def analyze_all(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Performs analysis (global metrics and fairness) for all datasets provided at initialization.

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: A dictionary containing the analysis results,
                structured as: {'metrics': {metric_name: DataFrame},
                                'fairness': {metric_name: DataFrame}}
                Each DataFrame summarizes results across all datasets for that specific metric.
        """
        results = {
            'metrics': {}, # Store global metric summaries
            'fairness': {} # Store fairness metric summaries
        }

        # Initialize dictionaries for each metric type
        for metric in self.metrics:
             results['metrics'][metric] = []
             results['fairness'][f'variance_{metric}'] = [] # Store variance results
             results['fairness'][f'pct_better_{metric}'] = [] # Store pct_better results


        # Analyze each dataset
        for dataset_name, dataset_results in self.all_dataset_results.items():
            print(f"\n--- Analyzing Dataset: {dataset_name} ---")
            if not dataset_results:
                 print(f"Warning: No results found for dataset {dataset_name}. Skipping analysis.")
                 continue

            # --- Analyze Global Metrics ---
            global_metrics_data = self.analyze_dataset(dataset_results)
            for algo, algo_metrics in global_metrics_data.items():
                for metric, stats in algo_metrics.items():
                     results['metrics'][metric].append({
                         'Dataset': dataset_name,
                         'Algorithm': algo,
                         'Median': stats['median'],
                         'CI_Lower': stats['ci_lower'],
                         'CI_Upper': stats['ci_upper']
                     })

            # --- Analyze Fairness Metrics ---
            try:
                fairness_df = self.analyze_fairness(dataset_results)
                # Append fairness results to the respective lists
                for _, row in fairness_df.iterrows():
                     metric = row['Metric']
                     results['fairness'][f'variance_{metric}'].append({
                         'Dataset': dataset_name,
                         'Algorithm': row['Algorithm'],
                         'Value': row['Variance'] # Use generic 'Value' column name
                     })
                     results['fairness'][f'pct_better_{metric}'].append({
                         'Dataset': dataset_name,
                         'Algorithm': row['Algorithm'],
                         'Value': row['Pct_Better'] # Use generic 'Value' column name
                     })
            except Exception as e:
                # Catch errors during fairness analysis (e.g., missing baselines)
                print(f"Warning: Fairness analysis failed for dataset {dataset_name}: {e}")


        # --- Convert accumulated results into DataFrames ---
        final_results = {'metrics': {}, 'fairness': {}}
        for metric in results['metrics']:
            if results['metrics'][metric]:
                 final_results['metrics'][metric] = pd.DataFrame(results['metrics'][metric])
            else:
                 print(f"Warning: No global metrics data collected for metric '{metric}'.")

        for fairness_key in results['fairness']:
             if results['fairness'][fairness_key]:
                 final_results['fairness'][fairness_key] = pd.DataFrame(results['fairness'][fairness_key])
             # else: # No need to warn if no fairness data was generated (e.g., due to missing baselines)
             #    print(f"Warning: No fairness data collected for '{fairness_key}'.")


        return final_results # Return dict containing DataFrames


    def _perform_friedman_test(self, pivot_table: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        Performs the Friedman test on performance data arranged in a pivot table.

        The Friedman test is a non-parametric test to detect differences in treatments
        across multiple test attempts (datasets in this case). It ranks the algorithms
        within each dataset and checks if the mean ranks are significantly different.

        Args:
            pivot_table (pd.DataFrame): A DataFrame where rows are algorithms, columns
                                        are datasets (or other blocking factor), and
                                        values are the performance metric (e.g., median).
                                        Should *not* include 'Mean Rank' column yet.

        Returns:
            Optional[Dict[str, float]]: A dictionary with 'statistic' (chi-squared) and 'pvalue',
                                        or None if the test cannot be performed (e.g., < 2 algorithms).
        """
        # Identify the data columns (datasets) excluding potential rank columns
        data_columns = [col for col in pivot_table.columns if col not in ['Mean Rank', 'Friedman Test']]
        valid_data = pivot_table[data_columns].dropna(axis=1, how='all').dropna(axis=0, how='any') # Drop datasets with all NaNs, drop algos with any NaN

        # Need at least 2 groups (algorithms) and 2 blocks (datasets) to compare
        if valid_data.shape[0] < 2 or valid_data.shape[1] < 2:
            print(f"Warning: Friedman test requires at least 2 algorithms and 2 datasets with non-NaN values. Found {valid_data.shape}. Skipping test.")
            return None

        try:
            # Perform the Friedman test on the valid data
            statistic, pvalue = stats.friedmanchisquare(
                *[valid_data[col].values for col in valid_data.columns]
            )
            return {'statistic': statistic, 'pvalue': pvalue}
        except Exception as e:
            print(f"Error performing Friedman test: {e}")
            return None


    def format_tables(self, datasets_order: List[str], analysis_results: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, Union[pd.DataFrame, str]]]:
        """
        Formats the analyzed results (medians, CIs, fairness metrics) into
        presentation-ready pandas DataFrames with specific ordering and formatting.

        Calculates mean ranks across datasets and performs Friedman tests.

        Args:
            datasets_order (List[str]): The desired order of dataset columns in the tables.
            analysis_results (Dict): The dictionary of DataFrames produced by `analyze_all`.
                                    {'metrics': {metric: df}, 'fairness': {fairness_key: df}}

        Returns:
            Dict[str, Dict[str, Union[pd.DataFrame, str]]]: A dictionary where keys are descriptive names
                (e.g., 'accuracy_summary', 'fairness_variance_loss') and values are dicts
                containing the formatted DataFrame ('table') and Friedman test results ('friedman').
        """
        formatted_tables = {}

        # --- Define Algorithm and Dataset Ordering ---
        # Use ALGORITHMS and DATASETS from configs.py for consistency
        algorithm_order = ALGORITHMS
        dataset_order_with_rank = datasets_order + ['Mean Rank'] # Add Mean Rank column

        # Define order for fairness tables (excluding baselines)
        fairness_algorithm_order = [algo for algo in algorithm_order if algo not in ['local', 'fedavg']]

        # --- Format Global Metric Summaries ---
        if 'metrics' in analysis_results:
            for metric, df in analysis_results['metrics'].items():
                if df.empty: continue # Skip empty dataframes

                try:
                    # Ensure numeric values are actually numeric
                    for col in ['Median', 'CI_Lower', 'CI_Upper']:
                        # Convert string representations to float if necessary
                        if df[col].dtype == 'object':
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Create pivot tables for median and CI bounds
                    pivot_median = df.pivot(index='Algorithm', columns='Dataset', values='Median')
                    pivot_ci_lower = df.pivot(index='Algorithm', columns='Dataset', values='CI_Lower')
                    pivot_ci_upper = df.pivot(index='Algorithm', columns='Dataset', values='CI_Upper')

                    # --- Calculate Ranks ---
                    ranks_df = df.copy()
                    # Determine ranking direction (lower is better for loss, higher otherwise)
                    ascending_rank = (metric == 'loss')
                    # Rank algorithms within each dataset based on median performance
                    ranks_df['Rank'] = ranks_df.groupby('Dataset')['Median'].rank(method='average', ascending=ascending_rank)
                    # Calculate mean rank across datasets for each algorithm
                    mean_ranks = ranks_df.groupby('Algorithm')['Rank'].mean()

                    # --- Combine into Final Pivot Table ---
                    pivot_final = pivot_median.copy()
                    pivot_final['Mean Rank'] = mean_ranks

                    # --- Format Numbers with Confidence Intervals ---
                    # Calculate CI width / 2 for the ± notation
                    ci_half_width = (pivot_ci_upper - pivot_ci_lower) / 2.0
                    # Iterate through dataset columns to format
                    for col in datasets_order: # Only format dataset columns
                        if col in pivot_final.columns:
                            # Create a new column with formatted strings
                            formatted_col = []
                            for idx in pivot_final.index:
                                if pd.notna(pivot_final.loc[idx, col]) and pd.notna(ci_half_width.loc[idx, col]):
                                    try:
                                        # Use explicit float conversion to ensure numeric values
                                        median_val = float(pivot_final.loc[idx, col])
                                        half_width_val = float(ci_half_width.loc[idx, col])
                                        formatted_col.append(f"{median_val:.3f} ± {half_width_val:.3f}")
                                    except (ValueError, TypeError):
                                        formatted_col.append("N/A")
                                else:
                                    formatted_col.append("N/A")
                            # Replace the column with the formatted values
                            pivot_final[col] = formatted_col

                    # Format Mean Rank column
                    if 'Mean Rank' in pivot_final.columns:
                        formatted_ranks = []
                        for idx in pivot_final.index:
                            if pd.notna(pivot_final.loc[idx, 'Mean Rank']):
                                try:
                                    rank_val = float(pivot_final.loc[idx, 'Mean Rank'])
                                    formatted_ranks.append(f"{rank_val:.2f}")
                                except (ValueError, TypeError):
                                    formatted_ranks.append("N/A")
                            else:
                                formatted_ranks.append("N/A")
                        pivot_final['Mean Rank'] = formatted_ranks

                    # --- Reorder Rows and Columns ---
                    # Reindex rows (algorithms) based on defined order, drop missing
                    pivot_final = pivot_final.reindex(index=algorithm_order).dropna(how='all')
                    # Reindex columns (datasets + Mean Rank), drop missing
                    pivot_final = pivot_final.reindex(columns=dataset_order_with_rank).dropna(axis=1, how='all')

                    # --- Perform Friedman Test ---
                    # Use the original median pivot table for the test
                    friedman_result = self._perform_friedman_test(pivot_median.reindex(index=algorithm_order).dropna(how='all'))
                    friedman_str = f"Friedman test p-value: {friedman_result['pvalue']:.5f}" if friedman_result and 'pvalue' in friedman_result else "Friedman test not applicable or failed"

                    # Store formatted table and Friedman result
                    formatted_tables[f'{metric}_summary'] = {
                        'table': pivot_final,
                        'friedman': friedman_str
                    }
                except Exception as e:
                    print(f"Error formatting global metrics table for '{metric}': {e}")
                    import traceback
                    traceback.print_exc()

        # --- Format Fairness Metric Summaries ---
        if 'fairness' in analysis_results:
            # Process variance and pct_better separately
            for fairness_key, df in analysis_results['fairness'].items():
                if df.empty: continue

                try:
                    # Ensure Value column is numeric
                    if df['Value'].dtype == 'object':
                        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
                    
                    # Determine if higher is better (Pct_Better) or lower is better (Variance)
                    is_variance = 'variance' in fairness_key
                    metric_name = fairness_key.replace('variance_', '').replace('pct_better_', '')
                    table_type = 'Variance' if is_variance else 'Pct_Better'

                    # Create pivot table for the fairness value
                    pivot_value = df.pivot(index='Algorithm', columns='Dataset', values='Value')

                    # --- Calculate Ranks ---
                    ranks_df = df.copy()
                    # Lower variance is better, higher Pct_Better is better
                    ascending_rank = is_variance
                    ranks_df['Rank'] = ranks_df.groupby('Dataset')['Value'].rank(method='average', ascending=ascending_rank)
                    mean_ranks = ranks_df.groupby('Algorithm')['Rank'].mean()

                    # --- Combine into Final Pivot Table ---
                    pivot_final = pivot_value.copy()
                    pivot_final['Mean Rank'] = mean_ranks

                    # --- Format Numbers ---
                    value_format = "{:.6f}" if is_variance else "{:.1f}" # Different precision
                    for col in datasets_order: # Only format dataset columns
                        if col in pivot_final.columns:
                            formatted_col = []
                            for idx in pivot_final.index:
                                if pd.notna(pivot_final.loc[idx, col]):
                                    try:
                                        # Explicit float conversion
                                        val = float(pivot_final.loc[idx, col])
                                        formatted_col.append(value_format.format(val))
                                    except (ValueError, TypeError):
                                        formatted_col.append("N/A")
                                else:
                                    formatted_col.append("N/A")
                            pivot_final[col] = formatted_col

                    # Format Mean Rank column
                    if 'Mean Rank' in pivot_final.columns:
                        formatted_ranks = []
                        for idx in pivot_final.index:
                            if pd.notna(pivot_final.loc[idx, 'Mean Rank']):
                                try:
                                    rank_val = float(pivot_final.loc[idx, 'Mean Rank'])
                                    formatted_ranks.append(f"{rank_val:.2f}")
                                except (ValueError, TypeError):
                                    formatted_ranks.append("N/A")
                            else:
                                formatted_ranks.append("N/A")
                        pivot_final['Mean Rank'] = formatted_ranks

                    # --- Reorder Rows and Columns ---
                    pivot_final = pivot_final.reindex(index=fairness_algorithm_order).dropna(how='all')
                    pivot_final = pivot_final.reindex(columns=dataset_order_with_rank).dropna(axis=1, how='all')

                    # --- Perform Friedman Test ---
                    friedman_result = self._perform_friedman_test(pivot_value.reindex(index=fairness_algorithm_order).dropna(how='all'))
                    friedman_str = f"Friedman test p-value: {friedman_result['pvalue']:.5f}" if friedman_result and 'pvalue' in friedman_result else "Friedman test not applicable or failed"

                    # Store formatted table and Friedman result
                    formatted_tables[f'fairness_{table_type.lower()}_{metric_name}'] = {
                        'table': pivot_final,
                        'friedman': friedman_str
                    }
                except Exception as e:
                    print(f"Error formatting fairness table for '{fairness_key}': {e}")
                    import traceback
                    traceback.print_exc()

        return formatted_tables


def analyze_experiment_results(datasets_to_analyze: List[str]) -> Dict:
    """
    Top-level function to load evaluation results for specified datasets,
    analyze them using `ResultAnalyzer`, and format them into tables.

    Args:
        datasets_to_analyze (List[str]): A list of dataset names for which to
                                         load and analyze evaluation results.

    Returns:
        Dict: A dictionary containing the formatted tables and Friedman test results,
              as generated by `ResultAnalyzer.format_tables`. Returns an empty
              dictionary if major errors occur.
    """
    print(f"Starting analysis for datasets: {datasets_to_analyze}")
    try:
        # Load results for all specified datasets
        all_results = {}
        loaded_datasets = []
        for dataset in datasets_to_analyze:
            eval_results = load_eval_results(dataset)
            if eval_results is not None:
                all_results[dataset] = eval_results
                loaded_datasets.append(dataset)
            else:
                print(f"Warning: Could not load evaluation results for dataset '{dataset}'. It will be excluded from analysis.")

        if not all_results:
             print("Error: No evaluation results could be loaded for any specified dataset. Aborting analysis.")
             return {}

        # Initialize analyzer with the loaded results
        analyzer = ResultAnalyzer(all_results)

        # Perform the analysis (calculates stats, fairness)
        print("\nPerforming comprehensive analysis...")
        analysis_results = analyzer.analyze_all() # Returns {'metrics': {m: df}, 'fairness': {f: df}}

        # Format the analysis into tables
        print("\nFormatting results into tables...")
        # Use the list of datasets that were actually loaded for table ordering
        formatted_tables = analyzer.format_tables(loaded_datasets, analysis_results)

        print("\nAnalysis complete.")
        return formatted_tables

    except Exception as e:
        print(f"FATAL ERROR during results analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return {} # Return empty dict on major failure