"""
Utility functions and classes used across the codebase

Includes functions for setting random seeds, retrieving configuration parameters,
managing GPU resources, moving data to devices, and managing experiment results.
"""
from configs import * # Import all configurations



def set_seeds(seed_value=1):
    """
    Set random seeds for PyTorch, NumPy, and Python's random module.

    Ensures reproducibility of experiments. Also configures CUDA operations
    to be deterministic.

    Args:
        seed_value (int): The seed value to use. Defaults to 1.
    """
    torch.manual_seed(seed_value)
    # Sets seeds for all GPUs if available
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    # Disables benchmark mode which can introduce non-determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_parameters_for_dataset(DATASET: str) -> Dict:
    """
    Retrieve the default configuration parameters for a given dataset.

    Args:
        DATASET (str): The name of the dataset (e.g., 'FMNIST', 'CIFAR').
                       Must be one of the keys in `DEFAULT_PARAMS`.

    Returns:
        Dict: A dictionary containing the default parameters for the dataset.

    Raises:
        ValueError: If the specified DATASET is not found in `DEFAULT_PARAMS`.
    """
    params = DEFAULT_PARAMS.get(DATASET)
    if not params:
        raise ValueError(f"Dataset {DATASET} is not supported or has no default parameters defined.")
    return params


def get_algorithm_config(server_type: str, dataset_name: str) -> Dict:
    """
    Get algorithm-specific parameters based on the server type and dataset.

    This function retrieves parameters like layers to federate, regularization
    constants, hypernetwork settings, etc., specific to certain FL algorithms.

    Args:
        server_type (str): The name of the FL algorithm (e.g., 'fedprox', 'layerpfl').
                           Should be one of the values in `ALGORITHMS`.
        dataset_name (str): The name of the dataset (e.g., 'FMNIST', 'CIFAR').

    Returns:
        Dict: A dictionary containing the specific parameters for the
              given algorithm and dataset combination. Returns an empty
              dictionary if no specific parameters are needed for the algorithm.
    """
    params = {}

    # --- Layer-based Methods ---
    # These methods involve selecting specific layers for federation.
    if server_type == 'babu':
        # BABU federates all layers except the final classification head.
        params['layers_to_include'] = LAYERS_TO_FEDERATE_DICT[server_type][dataset_name]

    if server_type in ['playerfl']:
        # LayerPFL federates a predefined, fixed subset of layers based on layer metrics.
        params['layers_to_include'] = LAYERS_TO_FEDERATE_DICT[server_type][dataset_name]
        # Note: Regularization parameter is defined but not used in the standard LayerPFL setup.
        params['reg_param'] = REG_PARAMS.get('playerfl', {}).get(dataset_name)

    elif server_type == 'playerfl_random':
        # LayerPFL_random selects a random prefix of the available layers for federation.
        # It ensures the selected subset is different from the fixed 'layerpfl' set
        # and also not the entire set of layers defined for 'playerfl_random'.
        fixed_layers = LAYERS_TO_FEDERATE_DICT['playerfl'][dataset_name]
        all_possible_layers = LAYERS_TO_FEDERATE_DICT[server_type][dataset_name] # The pool to choose from

        # Keep selecting a random prefix until it's valid
        while True:
            # Choose a random index (from 0 to len-1)
            idx = np.random.randint(len(all_possible_layers))
            # Select layers up to and including the chosen index
            selected_layers = all_possible_layers[:idx + 1]
            # Ensure the selection is non-empty, not identical to fixed LayerPFL,
            # and not identical to the full set which is just fedavg (meaning at least one layer is excluded).
            if selected_layers and (selected_layers != fixed_layers) and (selected_layers != all_possible_layers):
                break
            # If the full set is small (e.g., 1 layer) or identical to fixed set,
            if len(all_possible_layers) <= 1 or all_possible_layers == fixed_layers:
                 # Fallback or warning: Use the fixed set or the full set if no valid random subset found
                 selected_layers = fixed_layers if fixed_layers else all_possible_layers
                 print(f"Warning: Could not find a distinct random subset for playerfl_random on {dataset_name}. Using fallback.")
                 break
        params['layers_to_include'] = selected_layers

    # --- FedLP Specific Parameters ---
    elif server_type == 'fedlp':
        # FedLP uses a probability rate for layer participation in aggregation.
        params['layer_preserving_rate'] = LAYER_PRESERVATION_RATES[dataset_name]

    # --- Regularization-based Methods ---
    # These methods add a regularization term to the local objective function.
    elif server_type in ['fedprox', 'pfedme', 'ditto']:
        params['reg_param'] = REG_PARAMS[server_type][dataset_name]

    # --- FedLAMA Specific Parameters ---
    elif server_type == 'fedlama':
        # FedLAMA uses parameters for adaptive aggregation frequency.
        params['tau_prime'] = LAMA_RATES['tau_prime']  # Base aggregation interval
        params['phi'] = LAMA_RATES['phi']        # Interval increase factor

    # --- pFedLA Specific Parameters ---
    elif server_type == 'pfedla':
        # pFedLA uses a hypernetwork for personalized layer aggregation.
        params['embedding_dim'] = HYPERNETWORK_PARAMS['embedding_dim'][dataset_name]
        params['hidden_dim'] = HYPERNETWORK_PARAMS['hidden_dim'][dataset_name]
        params['hn_lr'] = HYPERNETWORK_PARAMS['hn_lr'][dataset_name]

    return params


def cleanup_gpu():
    """
    Release unused GPU memory cached by PyTorch and trigger garbage collection.

    Helpful to call after deleting large tensors or models to free up memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Clears PyTorch's cache
        gc.collect()             # Triggers Python's garbage collector


def move_to_device(batch: Union[Tensor, List[Optional[Tensor]], Tuple[Optional[Tensor],]],
                   device: str) -> Union[Tensor, List[Optional[Tensor]], Tuple[Optional[Tensor]]]:
    """
    Recursively move a batch of data (tensors or structures of tensors) to the specified device.

    Handles single tensors, lists of tensors, and tuples of tensors.
    Preserves None values within lists/tuples.

    Args:
        batch: The data batch to move. Can be a Tensor, list, or tuple
               containing Tensors or None.
        device: The target torch device (e.g., 'cuda', 'cpu', torch.device('cuda:0')).

    Returns:
        The batch with all tensors moved to the target device, maintaining the
        original structure (tensor, list, or tuple).
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        # Recursively move elements in the list/tuple
        return [move_to_device(x, device) if x is not None else None for x in batch]
    elif isinstance(batch, dict):
        # Handle dictionaries if necessary (move values)
         return {k: move_to_device(v, device) for k, v in batch.items()}
    else:
        # If the batch element is not a tensor, list, tuple, or dict, return as is
        return batch


class ExperimentType:
    """Enumerates the types of experiments that can be run."""
    LEARNING_RATE = 'learning_rate' # Experiment to tune the learning rate
    EVALUATION = 'evaluation'       # Experiment for final evaluation using best hyperparameters


class ExperimentConfig:
    """
    Configuration settings for a single experiment run.

    Attributes:
        dataset (str): Name of the dataset being used.
        experiment_type (str): The type of experiment (from ExperimentType).
    """
    def __init__(self, dataset: str, experiment_type: str):
        """
        Initializes the ExperimentConfig.

        Args:
            dataset (str): The dataset name (e.g., 'FMNIST').
            experiment_type (str): The type of experiment (e.g., ExperimentType.LEARNING_RATE).
        """
        self.dataset = dataset
        if experiment_type not in [ExperimentType.LEARNING_RATE, ExperimentType.EVALUATION]:
            raise ValueError(f"Invalid experiment_type: {experiment_type}")
        self.experiment_type = experiment_type


class ResultsManager:
    """
    Manages loading, saving, and processing of experiment results stored in pickle files.

    Organizes results based on experiment type (e.g., learning rate tuning, evaluation)
    and dataset. Provides methods to load existing results, save new results,
    append metrics, and find the best hyperparameters from tuning runs.
    """
    def __init__(self, dataset: str, experiment_type: str):
        """
        Initializes the ResultsManager.

        Args:
            dataset (str): The name of the dataset.
            experiment_type (str): The type of the current experiment.
        """
        self.dir = RESULTS_DIR
        self.dataset = dataset
        self.experiment_type = experiment_type
        # Defines the directory structure and filenames for saving results.
        self.results_structure = {
            ExperimentType.LEARNING_RATE: {
                'directory': 'lr_tuning',
                'filename_template': f'{dataset}_lr_tuning.pkl',
            },
            ExperimentType.EVALUATION: {
                'directory': 'evaluation',
                'filename_template': f'{dataset}_evaluation.pkl'
            },
            # Add other experiment types here if needed
        }

    def _get_results_path(self, experiment_type: str) -> str:
        """
        Constructs the full path to the results file for a given experiment type.

        Args:
            experiment_type (str): The type of experiment (from ExperimentType).

        Returns:
            str: The absolute path to the results pickle file.

        Raises:
            KeyError: If the experiment_type is not defined in `results_structure`.
        """
        if experiment_type not in self.results_structure:
            raise KeyError(f"Results structure not defined for experiment type: {experiment_type}")
        experiment_info = self.results_structure[experiment_type]
        # Ensure the directory exists before returning the path
        dir_path = os.path.join(self.dir, experiment_info['directory'])
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, experiment_info['filename_template'])

    def load_results(self, experiment_type: str) -> Optional[Dict]:
        """
        Loads results from a pickle file for the specified experiment type.

        Args:
            experiment_type (str): The type of experiment to load results for.

        Returns:
            Optional[Dict]: The loaded results dictionary, or None if the file
                            does not exist or an error occurs during loading.
        """
        path = self._get_results_path(experiment_type)
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
                print(f"Warning: Could not load results from {path}. Error: {e}. Returning None.")
                # Optionally, back up the corrupted file here
                return None
        return None # File doesn't exist

    def save_results(self, results: Dict, experiment_type: str):
        """
        Saves the results dictionary to a pickle file for the specified experiment type.

        Args:
            results (Dict): The dictionary containing the experiment results.
            experiment_type (str): The type of experiment these results belong to.
        """
        path = self._get_results_path(experiment_type)
        try:
            with open(path, 'wb') as f:
                pickle.dump(results, f)
        except (IOError, pickle.PicklingError) as e:
             print(f"Error: Failed to save results to {path}. Error: {e}")


    def append_or_create_metric_lists(self, existing_dict: Optional[Dict], new_dict: Dict) -> Dict:
        """
        Recursively merges new results into an existing results dictionary.

        If a key exists in both dictionaries and its value is not a dictionary,
        the new value is appended to a list associated with that key in the existing
        dictionary. If the key only exists in the new dictionary, it's added to the
        existing dictionary, with its value wrapped in a list.
        If a value is itself a dictionary, the function recurses.

        This is useful for accumulating results from multiple runs.

        Example:
            existing = {'algoA': {'loss': [0.5], 'acc': [0.9]}}
            new =      {'algoA': {'loss': 0.4, 'acc': 0.95}, 'algoB': {'loss': 0.6}}
            result =   {'algoA': {'loss': [0.5, 0.4], 'acc': [0.9, 0.95]}, 'algoB': {'loss': [0.6]}}

        Args:
            existing_dict (Optional[Dict]): The dictionary to append to (or None if starting fresh).
            new_dict (Dict): The new dictionary containing results to merge.

        Returns:
            Dict: The updated dictionary with appended results.
        """
        if existing_dict is None:
            # If starting fresh, create lists for all values in the new dictionary
            # unless the value is already a dict (then recurse).
            return {k: [v] if not isinstance(v, dict) else
                    self.append_or_create_metric_lists(None, v)
                    for k, v in new_dict.items()}

        output_dict = copy.deepcopy(existing_dict) # Avoid modifying the original dict directly

        for key, new_value in new_dict.items():
            if key not in output_dict:
                # Key is new, add it with the value wrapped in a list (or recurse if dict)
                output_dict[key] = [new_value] if not isinstance(new_value, dict) else \
                                   self.append_or_create_metric_lists(None, new_value)
            else:
                # Key exists
                if isinstance(new_value, dict) and isinstance(output_dict[key], dict):
                     # Both values are dicts, recurse
                     output_dict[key] = self.append_or_create_metric_lists(
                         output_dict[key], new_value)
                elif isinstance(output_dict[key], list):
                     # Existing value is a list, append the new value
                     output_dict[key].append(new_value)
                else:
                     if not isinstance(new_value, dict):
                         output_dict[key] = [output_dict[key], new_value]
                     else:
                         print(f"Warning: Merging conflict for key '{key}'. Existing value is not a list or dict, but new value is a dict. Overwriting with new structure.")
                         output_dict[key] = self.append_or_create_metric_lists(None, new_value)


        return output_dict

    def get_best_parameters(self, param_type: str, server_type: str) -> Optional[Union[float, int, str]]:
        """
        Retrieves the best hyperparameter value for a given server type based on saved tuning results.

        Currently supports finding the best learning rate (ExperimentType.LEARNING_RATE).
        The "best" is determined by the hyperparameter value that resulted in the
        lowest mean validation loss across all rounds, averaged over multiple runs.

        Args:
            param_type (str): The type of hyperparameter tuning results to load
                              (e.g., ExperimentType.LEARNING_RATE).
            server_type (str): The algorithm/server type for which to find the best parameter.

        Returns:
            Optional[Union[float, int, str]]: The best hyperparameter value found,
                                             or None if no results are available for
                                             this server type or parameter type.
        """
        results = self.load_results(param_type) # e.g., load lr_tuning results
        if results is None:
            print(f"Warning: No results found for parameter type '{param_type}' for dataset '{self.dataset}'.")
            return None

        # Structure of results (example for LR tuning):
        # results = {
        #     lr1: {
        #         'fedavg': {'global': {'losses': [[run1_r1, run1_r2,...], [run2_r1,...]], 'scores': [...]}, ...},
        #         'fedprox': {...}
        #     },
        #     lr2: {...}
        # }

        # Collect all results for the specific server_type across different hyperparameter values
        server_results_by_param = {}
        for param_value, algorithms_results in results.items():
            if server_type in algorithms_results:
                server_results_by_param[param_value] = algorithms_results[server_type]

        if not server_results_by_param:
            print(f"Warning: No results found for server type '{server_type}' within '{param_type}' results for dataset '{self.dataset}'.")
            return None

        # Determine the best hyperparameter based on performance
        return self._select_best_hyperparameter(server_results_by_param)

    def _select_best_hyperparameter(self, param_results: Dict) -> Optional[Union[float, int, str]]:
        """
        Internal helper to select the best hyperparameter from collected results.

        Finds the hyperparameter value that yields the minimum mean validation loss
        over all rounds, averaged across runs.

        Args:
            param_results (Dict): A dictionary where keys are hyperparameter values
                                 (e.g., learning rates) and values are the corresponding
                                 metric dictionaries for a specific server type.
                                 Example: {lr1: {'global': {'losses': [...], ...}}, lr2: {...}}

        Returns:
            Optional[Union[float, int, str]]: The best hyperparameter value, or None if input is empty.
        """
        best_overall_loss = float('inf')
        best_param_value = None

        for param_value, metrics in param_results.items():
            # Check if 'global' metrics and 'losses' are present and valid
            if 'global' not in metrics or 'losses' not in metrics['global'] or not metrics['global']['losses']:
                print(f"Warning: Skipping param {param_value} due to missing or empty 'global' losses.")
                continue

            global_losses_per_run = metrics['global']['losses'] # List of lists: [[run1_r1, run1_r2,...], [run2_r1,...]]

            # Ensure all runs have the same number of rounds, find num_rounds
            try:
                num_rounds = len(global_losses_per_run[0])
                if not all(len(run_losses) == num_rounds for run_losses in global_losses_per_run):
                    print(f"Warning: Inconsistent number of rounds for param {param_value}. Skipping.")
                    continue
            except IndexError:
                print(f"Warning: Empty loss list found for param {param_value}. Skipping.")
                continue

            # Calculate the mean loss for each round across all runs
            mean_losses_per_round = []
            for round_idx in range(num_rounds):
                # Get losses for this round from all runs
                losses_this_round = [run_losses[round_idx] for run_losses in global_losses_per_run]
                # Calculate the mean loss for this round
                mean_round_loss = np.mean(losses_this_round)
                mean_losses_per_round.append(mean_round_loss)

            # Find the minimum mean loss achieved across all rounds for this hyperparameter value
            min_mean_loss_for_param = min(mean_losses_per_round) if mean_losses_per_round else float('inf')

            # Update the best parameter if this one performed better
            if min_mean_loss_for_param < best_overall_loss:
                best_overall_loss = min_mean_loss_for_param
                best_param_value = param_value

        if best_param_value is None:
             print(f"Warning: Could not determine the best hyperparameter. No valid results processed.")

        return best_param_value