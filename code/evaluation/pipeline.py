"""
Orchestrates the federated learning experiment execution.

Defines the `Experiment` class which manages the overall workflow,
including:
- Loading configuration and data.
- Setting up the server and clients based on the chosen algorithm.
- Running hyperparameter tuning or final evaluation phases.
- Coordinating training rounds and evaluation.
- Saving results using `ResultsManager`.
- Provides helper functions for initializing loggers.
"""
from helper import *            # Utility functions (seeds, device, cleanup, etc.)
from configs import *           # Global configurations (paths, algorithms, datasets, params)
from dataset_processing import * # Data loading and preprocessing classes
import models                   # Model definitions
from losses import MulticlassFocalLoss # Custom loss functions if used
from clients import *           # Client implementations for different algorithms
from servers import *           # Server implementations for different algorithms
from performance_logging import * # Logging setup and utilities
import time

# Instantiate the global performance logger
performance_logger = PerformanceLogger()

def get_client_logger(client_id: str, algorithm_type: Optional[str] = None) -> logging.Logger:
    """
    Gets a dedicated logger instance for a specific client.

    Args:
        client_id (str): The unique identifier of the client (e.g., 'client_1').
        algorithm_type (Optional[str]): An optional identifier for the algorithm
                                        being run, used in the logger name.

    Returns:
        logging.Logger: The configured logger for the client.
    """
    # Use client_id as the 'dataset' part and algorithm_type as the 'name' part for get_logger
    logger_name_suffix = algorithm_type if algorithm_type else "run"
    return performance_logger.get_logger(dataset=client_id, name=logger_name_suffix)

def get_server_logger(algorithm_type: Optional[str] = None) -> logging.Logger:
    """
    Gets a dedicated logger instance for the central server.

    Args:
        algorithm_type (Optional[str]): An optional identifier for the algorithm
                                        being run, used in the logger name.

    Returns:
        logging.Logger: The configured logger for the server.
    """
    logger_name_suffix = algorithm_type if algorithm_type else "run"
    return performance_logger.get_logger(dataset="server", name=logger_name_suffix)


class Experiment:
    """
    Manages the setup and execution of a federated learning experiment.

    Handles both hyperparameter tuning and final evaluation phases based on the
    provided configuration. Coordinates data loading, server/client creation,
    training loop, evaluation, and results saving.
    """
    def __init__(self, config: ExperimentConfig):
        """
        Initializes the Experiment.

        Args:
            config (ExperimentConfig): Configuration object specifying the dataset
                                       and experiment type.
        """
        self.config = config
        self.root_dir = ROOT_DIR
        self.results_manager = ResultsManager(
            dataset=self.config.dataset,
            experiment_type=self.config.experiment_type
        )
        # Load default parameters (like LR range, rounds, etc.) for the specified dataset
        self.default_params = get_parameters_for_dataset(self.config.dataset)
        # Get a logger specific to this dataset and experiment type
        self.logger = performance_logger.get_logger(
            dataset=self.config.dataset,
            name=self.config.experiment_type
        )
        self.logger.info(f"Initializing Experiment: Dataset='{config.dataset}', Type='{config.experiment_type}'")
        set_seeds() # Ensure seeds are set at the start of the experiment

    def run_experiment(self) -> Dict:
        """
        Executes the configured experiment.

        Delegates to specific methods for hyperparameter tuning or final evaluation.

        Returns:
            Dict: The results collected during the experiment execution.
        """
        self.logger.info(f"Starting experiment run for dataset {self.config.dataset}, type {self.config.experiment_type}")
        if self.config.experiment_type == ExperimentType.EVALUATION:
            results = self._run_final_evaluation()
        elif self.config.experiment_type == ExperimentType.LEARNING_RATE:
            results = self._run_hyperparameter_tuning()
        else:
            # Handle other experiment types if added later
            raise NotImplementedError(f"Experiment type '{self.config.experiment_type}' not implemented.")
        self.logger.info(f"Experiment run finished for dataset {self.config.dataset}, type {self.config.experiment_type}")
        return results

    def _check_existing_results(self, server_types: List[str]) -> Tuple[Optional[Dict], Dict[str, int]]:
        """
        Checks for previously saved results for the current experiment type.

        Determines how many runs have already been completed for each server type
        to allow resuming experiments.

        Args:
            server_types (List[str]): A list of algorithm/server types being run
                                      in this experiment phase.

        Returns:
            Tuple[Optional[Dict], Dict[str, int]]: A tuple containing:
                - The loaded results dictionary (or None if no file exists).
                - A dictionary mapping each server type to the number of completed runs found.
        """
        results = self.results_manager.load_results(self.config.experiment_type)
        completed_runs = {server_type: 0 for server_type in server_types}

        if results is not None:
            self.logger.info(f"Loaded existing results for {self.config.experiment_type}.")
            # Determine completed runs based on the structure: results[param_value][server_type]['global']['losses']
            # Assumes 'losses' is a list where each element represents a run.
            for param_value in results: # e.g., loop through learning rates
                for server_type in server_types:
                    if server_type in results[param_value]:
                        try:
                            # Count runs based on the length of the global losses list
                            num_runs = len(results[param_value][server_type]['global']['losses'])
                            # Update if this param_value shows more completed runs for this server_type
                            completed_runs[server_type] = max(completed_runs[server_type], num_runs)
                        except (KeyError, TypeError, IndexError):
                            # Handle cases where the structure might be incomplete or corrupt
                            self.logger.warning(f"Could not determine completed runs for '{server_type}' under param '{param_value}'. Structure might be incomplete.")
                            pass # Keep completed_runs[server_type] as it was
        else:
             self.logger.info(f"No existing results found for {self.config.experiment_type}. Starting fresh.")

        self.logger.info(f"Current completed runs: {completed_runs}")
        return results, completed_runs

    def _run_hyperparameter_tuning(self) -> Dict:
        """
        Manages the hyperparameter tuning process over multiple runs.

        Iterates through specified hyperparameter values (e.g., learning rates),
        runs experiments for each value across all relevant server types, and
        accumulates results, allowing for resumption.

        Returns:
            Dict: The accumulated results from all tuning runs.
        """
        self.logger.info(f"Starting hyperparameter tuning ({self.config.experiment_type}) for dataset {self.config.dataset}")
        server_types = ALGORITHMS # Tune all algorithms

        # --- Determine Hyperparameters to Test ---
        if self.config.experiment_type == ExperimentType.LEARNING_RATE:
            hyperparams_to_try = self.default_params['learning_rates_try']
            # Create list of dicts, e.g., [{'learning_rate': 0.01}, {'learning_rate': 0.001}]
            hyperparams_list = [{'learning_rate': hp} for hp in hyperparams_to_try]
            num_total_runs = self.default_params['runs_lr']
            param_name = 'learning_rate'
            self.logger.info(f"Tuning {param_name} values: {hyperparams_to_try}. Total runs per setting: {num_total_runs}")
        else:
            raise NotImplementedError(f"Hyperparameter tuning not implemented for type: {self.config.experiment_type}")

        # --- Load Existing Results and Check Completion Status ---
        results, completed_runs_dict = self._check_existing_results(server_types)

        # --- Run Tuning Loop ---
        min_completed = min(completed_runs_dict.values()) if completed_runs_dict else 0
        current_run_number = min_completed + 1

        while current_run_number <= num_total_runs:
            self.logger.info(f"--- Starting Hyperparameter Tuning Run {current_run_number}/{num_total_runs} ---")
            # Set seed for this specific run to ensure reproducibility if resumed
            set_seeds(seed_value=current_run_number)
            self.logger.info(f"Seeds set to {current_run_number} for this run.")

            # Dictionary to store results *for this specific run*
            results_this_run = {}

            # Iterate through each hyperparameter setting
            for hyperparams_setting in hyperparams_list:
                param_value = list(hyperparams_setting.values())[0] # Get the value (e.g., 0.01)
                self.logger.info(f"--> Testing {param_name} = {param_value}")

                # Initialize entry for this param value in the run's results
                results_this_run[param_value] = {}

                # Identify server types that still need this run completed
                server_types_for_this_run = [
                    st for st in server_types if completed_runs_dict[st] < current_run_number
                ]

                if not server_types_for_this_run:
                    self.logger.info(f"    All server types completed run {current_run_number} for {param_name}={param_value}. Skipping.")
                    continue

                # Run the actual tuning logic for the required server types
                try:
                    # _hyperparameter_tuning runs *one* round of experiments for the given hyperparams and servers
                    tuning_results = self._hyperparameter_tuning(hyperparams_setting, server_types_for_this_run)
                    # Store the results obtained for this parameter value in this run
                    results_this_run[param_value].update(tuning_results)
                except Exception as e:
                     self.logger.error(f"    Error during tuning run {current_run_number} for {param_name}={param_value}: {e}", exc_info=True)
                     # Decide how to handle: continue to next param, stop run, etc.
                     # For now, continue to next param value.

            # --- Merge and Save Results for the Completed Run ---
            if results_this_run: # Only merge if any results were generated
                self.logger.info(f"--- Merging results for Run {current_run_number} ---")
                # Append results from this run to the main results dictionary
                results = self.results_manager.append_or_create_metric_lists(results, results_this_run)
                # Save the updated results after each run completes
                self.results_manager.save_results(results, self.config.experiment_type)
                self.logger.info(f"Results saved after Run {current_run_number}.")

                # Update the completion count for servers that ran in this iteration
                for param_value, server_results in results_this_run.items():
                    for server_type in server_results:
                        completed_runs_dict[server_type] = max(completed_runs_dict[server_type], current_run_number)
            else:
                self.logger.info(f"No new results generated in Run {current_run_number}.")


            # Prepare for the next run
            current_run_number += 1
            cleanup_gpu()
            # Small pause for cleanup
            time.sleep(2)


        self.logger.info(f"Hyperparameter tuning ({self.config.experiment_type}) finished after {num_total_runs} runs.")
        return results

    def _hyperparameter_tuning(self, hyperparams: Dict, server_types: List[str]) -> Dict:
        """
        Executes a single tuning trial for a given set of hyperparameters and server types.

        Initializes the dataset and clients, then creates and runs each specified
        server type with the provided hyperparameters for one full cycle (all rounds).

        Args:
            hyperparams (Dict): The specific hyperparameter setting to test
                                (e.g., {'learning_rate': 0.01}).
            server_types (List[str]): The list of server/algorithm types to run
                                      with these hyperparameters.

        Returns:
            Dict: A dictionary mapping each server type run to its collected metrics.
                  Example: {'fedavg': {'global': {...}, 'sites': {...}}, 'fedprox': {...}}
        """
        # Initialize dataset loaders once for this trial
        client_dataloaders = self._initialize_experiment(self.default_params['batch_size'])
        # Dictionary to store metrics for this specific trial (one hyperparam setting, multiple servers)
        tracking_trial = {}

        for server_type in server_types:
            self.logger.info(f"    Running server type: {server_type} with params: {hyperparams}")
            server = None # Ensure server is reset/deleted
            try:
                # Create server instance with current hyperparameters, marking as 'tuning' run
                server = self._create_server_instance(server_type, hyperparams, tuning=True)
                # Add clients (with their data) to the server
                self._add_clients_to_server(server, client_dataloaders)
                # Train for the configured number of rounds and collect metrics
                metrics = self._train_and_evaluate(server, server.config.rounds)
                # Store metrics for this server type
                tracking_trial[server_type] = metrics
                self.logger.info(f"    Finished server type: {server_type}")

            except Exception as e:
                self.logger.error(f"    Error running server type {server_type}: {e}", exc_info=True)
                # Store partial results or error indicators if needed
                tracking_trial[server_type] = {'error': str(e)} # Example error tracking
            finally:
                # Ensure cleanup happens even if errors occur
                if server is not None:
                    del server # Explicitly delete server object
                cleanup_gpu() # Clean GPU memory after each server type finishes
                time.sleep(1) # Small pause

        return tracking_trial

    def _run_final_evaluation(self) -> Dict:
        """
        Manages the final evaluation phase over multiple independent runs.

        For each run, it retrieves the best hyperparameters found during tuning
        for each algorithm, sets up the experiment, runs the training and
        evaluation, and accumulates results.

        Returns:
            Dict: The accumulated results from all final evaluation runs.
        """
        self.logger.info(f"Starting final evaluation phase for dataset {self.config.dataset}")
        num_total_runs = self.default_params['runs']
        server_types = ALGORITHMS # Evaluate all algorithms

        # --- Load Existing Results and Check Completion Status ---
        # Use the EVALUATION type to load/save results
        results, completed_runs_dict = self._check_existing_results(server_types)

        # --- Run Evaluation Loop ---
        min_completed = min(completed_runs_dict.values()) if completed_runs_dict else 0
        current_run_number = min_completed + 1

        while current_run_number <= num_total_runs:
            self.logger.info(f"--- Starting Final Evaluation Run {current_run_number}/{num_total_runs} ---")
            # Set seed for this specific run
            set_seeds(seed_value=current_run_number)
            self.logger.info(f"Seeds set to {current_run_number} for this run.")

            # Dictionary to store results *for this specific run*
            results_this_run = {}

            # Run the evaluation logic for all algorithms
            try:
                # _final_evaluation runs *one* cycle of experiments using best HPs
                evaluation_results = self._final_evaluation(server_types_to_run=server_types)
                 # Store the results obtained in this run
                results_this_run.update(evaluation_results)
            except Exception as e:
                 self.logger.error(f"    Error during evaluation run {current_run_number}: {e}", exc_info=True)


            # --- Merge and Save Results for the Completed Run ---
            if results_this_run:
                self.logger.info(f"--- Merging results for Run {current_run_number} ---")
                wrapped_results = {f"run_{current_run_number}": results_this_run}
                results = self.results_manager.append_or_create_metric_lists(results, wrapped_results)

                # Save the updated results after each run completes
                self.results_manager.save_results(results, self.config.experiment_type)
                self.logger.info(f"Results saved after Run {current_run_number}.")

                # Update completion count - need careful handling based on saved structure
                for server_type in results_this_run:
                     completed_runs_dict[server_type] = max(completed_runs_dict[server_type], current_run_number)
            else:
                self.logger.info(f"No new results generated in Run {current_run_number}.")

            # Prepare for the next run
            current_run_number += 1
            time.sleep(2)
            cleanup_gpu()

        self.logger.info(f"Final evaluation finished after {num_total_runs} runs.")
        return results


    def _final_evaluation(self, server_types_to_run: List[str]) -> Dict:
        """
        Executes a single final evaluation trial for specified server types using best hyperparameters.

        Retrieves the best learning rate (and potentially other params) from tuning results,
        initializes dataset/clients, creates and runs each server, and collects metrics.

        Args:
            server_types_to_run (List[str]): List of server types to evaluate in this trial.

        Returns:
            Dict: Dictionary mapping each server type evaluated to its collected metrics.
        """
        tracking_evaluation = {}
        # Initialize dataset loaders once for this evaluation trial
        client_dataloaders = self._initialize_experiment(self.default_params['batch_size'])

        for server_type in server_types_to_run:
            self.logger.info(f"  Evaluating {server_type} with best hyperparameters")
            print(f"  Evaluating {server_type} with best hyperparameters") # Also print to console
            server = None
            try:
                # --- Get Best Hyperparameters ---
                # Retrieve best LR from tuning results
                best_lr = self.results_manager.get_best_parameters(
                    ExperimentType.LEARNING_RATE, server_type
                )
                if best_lr is None:
                    self.logger.warning(f"    Could not find best learning rate for {server_type}. Using dataset default: {self.default_params['learning_rate']}.")
                    best_lr = self.default_params['learning_rate']
                else:
                    self.logger.info(f"    Using best learning rate found: {best_lr}")

                # Consolidate hyperparameters
                hyperparams = {'learning_rate': best_lr}

                # --- Create and Run Server ---
                # Create server instance with best hyperparameters, marking as non-tuning run
                server = self._create_server_instance(server_type, hyperparams, tuning=False)
                # Add clients
                self._add_clients_to_server(server, client_dataloaders)
                # Train and evaluate (includes final test phase because tuning=False)
                metrics = self._train_and_evaluate(server, server.config.rounds)
                # Store metrics for this server type
                tracking_evaluation[server_type] = metrics
                self.logger.info(f"  Completed {server_type} evaluation trial.")

            except Exception as e:
                self.logger.error(f"    Error evaluating server type {server_type}: {e}", exc_info=True)
                tracking_evaluation[server_type] = {'error': str(e)}
            finally:
                if server is not None:
                    del server
                cleanup_gpu()
                time.sleep(1)

        return tracking_evaluation

    def _initialize_experiment(self, batch_size: int) -> Dict[str, Tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Initializes data loading and preprocessing for the experiment.

        Loads the dataset using `UnifiedDataLoader`, preprocesses it using
        `DataPreprocessor` to create train, validation, and test splits and
        DataLoaders for each client.

        Args:
            batch_size (int): The batch size to use for the DataLoaders.

        Returns:
            Dict[str, Tuple[DataLoader, DataLoader, DataLoader]]: A dictionary where keys
                are client IDs (e.g., 'client_1') and values are tuples containing
                (train_loader, val_loader, test_loader) for that client.
        """
        self.logger.info(f"Initializing dataset: {self.config.dataset} with batch size: {batch_size}")
        # Initialize preprocessor (handles transformations, splitting, DataLoader creation)
        # Pass dataset-specific alpha if needed for non-IID simulation during preprocessing
        alpha = DATASET_ALPHA.get(self.config.dataset)
        preprocessor = DataPreprocessor(self.config.dataset, batch_size)

        # Use UnifiedDataLoader to load the raw dataset (potentially pre-split by site)
        loader = UnifiedDataLoader(root_dir=self.root_dir, dataset_name=self.config.dataset)
        dataset_df = loader.load() # Returns DataFrame with 'data', 'label', 'site'

        # Process the raw data into client-specific train/val/test DataLoaders
        # The preprocessor handles splitting by site, train/val/test split, and DataLoader creation.
        client_dataloaders_map = preprocessor.process_client_data(dataset_df)
        self.logger.info(f"Data loaded and preprocessed for {len(client_dataloaders_map)} clients.")
        return client_dataloaders_map


    def _get_client_ids(self) -> List[str]:
        """
        Generates a list of client IDs based on the configured number of clients.

        Returns:
            List[str]: A list of client IDs, e.g., ['client_1', 'client_2', ...].
        """
        # Ensure num_clients is available in default_params
        num_clients = self.default_params.get('num_clients')
        if num_clients is None:
             raise ValueError("Configuration error: 'num_clients' not defined in default_params.")
        return [f'client_{i}' for i in range(1, num_clients + 1)]

    def _create_trainer_config(self, server_type: str, learning_rate: float, algorithm_params: Optional[Dict] = None) -> TrainerConfig:
        """
        Creates a TrainerConfig object for clients.

        Populates the configuration with default parameters and specific settings
        for the current experiment, including whether a personal model is required.

        Args:
            server_type (str): The algorithm/server type being used.
            learning_rate (float): The learning rate for this trial.
            algorithm_params (Optional[Dict]): Algorithm-specific parameters
                                               (e.g., reg_param, layers_to_include).

        Returns:
            TrainerConfig: The populated trainer configuration object.
        """
        # Determine if this algorithm requires a separate personal model state
        requires_personal = server_type in ['pfedme', 'ditto']
        # pFedLA manages personalization server-side, client doesn't need separate state flag here.
        # BABU, LocalAdaptation modify training but use the main model state.

        return TrainerConfig(
            dataset_name=self.config.dataset,
            device=DEVICE,
            learning_rate=learning_rate,
            batch_size=self.default_params['batch_size'],
            epochs=self.default_params['epochs_per_round'],
            rounds=self.default_params['rounds'],
            num_clients=self.default_params['num_clients'],
            requires_personal_model=requires_personal,
            algorithm_params=algorithm_params if algorithm_params else {} # Ensure it's a dict
        )

    def _create_model(self, learning_rate: float) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
        """
        Instantiates the model, loss function, and optimizer for the experiment.

        Selects the appropriate model architecture and loss function based on the
        dataset name specified in the configuration.

        Args:
            learning_rate (float): The learning rate to configure the optimizer with.

        Returns:
            Tuple[nn.Module, nn.Module, torch.optim.Optimizer]: A tuple containing:
                - The instantiated PyTorch model.
                - The instantiated loss function (criterion).
                - The instantiated optimizer.

        Raises:
            AttributeError: If the dataset name does not correspond to a defined model
                            class in the `models` module.
            ValueError: If no loss function is defined for the dataset.
        """
        num_classes = self.default_params['classes']

        # --- Instantiate Model ---
        try:
            model_class = getattr(models, self.config.dataset)
            model = model_class(num_classes)
            model.to(DEVICE) # Move model to the designated device
        except AttributeError:
            self.logger.error(f"Model class '{self.config.dataset}' not found in models.py")
            raise AttributeError(f"Model class '{self.config.dataset}' not found in models.py")

        # --- Instantiate Loss Function ---
        # Define loss functions per dataset, handling class imbalance if necessary
        loss_weights_isic = torch.tensor([0.87868852, 0.88131148, 0.82793443, 0.41206557], device=DEVICE)
        loss_weights_heart = torch.tensor([0.12939189, 0.18108108, 0.22331081, 0.22364865, 0.24256757], device=DEVICE) 
        loss_weights_mimic = torch.tensor([0.15, 0.85], device=DEVICE)

        criterion_map = {
            'EMNIST': nn.CrossEntropyLoss(),
            'CIFAR': nn.CrossEntropyLoss(),
            "FMNIST": nn.CrossEntropyLoss(),
            # Use Focal Loss for imbalanced datasets
            "ISIC": MulticlassFocalLoss(num_classes=num_classes, alpha=loss_weights_isic, gamma=1).to(DEVICE),
            "Sentiment": nn.CrossEntropyLoss(),
            "Heart": MulticlassFocalLoss(num_classes=num_classes, alpha=loss_weights_heart, gamma=3).to(DEVICE),
            "mimic": MulticlassFocalLoss(num_classes=num_classes, alpha=loss_weights_mimic, gamma=1).to(DEVICE),
        }
        criterion = criterion_map.get(self.config.dataset)
        if criterion is None:
             self.logger.error(f"Loss function (criterion) not defined for dataset: {self.config.dataset}")
             raise ValueError(f"Loss function (criterion) not defined for dataset: {self.config.dataset}")

        # --- Instantiate Optimizer ---
        # Use Adam optimizer with specified settings
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), # Optimize only trainable parameters
            lr=learning_rate,
            amsgrad=True,      # Use AMSGrad variant
            weight_decay=1e-4  # Apply L2 regularization
        )

        self.logger.info(f"Model, Criterion, Optimizer created for {self.config.dataset}. LR={learning_rate}")
        return model, criterion, optimizer

    def _create_server_instance(self, server_type: str, hyperparams: Dict, tuning: bool) -> Server:
        """
        Creates and configures a server instance for the specified algorithm.

        Args:
            server_type (str): The algorithm/server type (e.g., 'fedavg', 'fedprox').
            hyperparams (Dict): Dictionary of hyperparameters for this trial (e.g., {'learning_rate': 0.01}).
            tuning (bool): Flag indicating if this is a tuning run (influences test evaluation).

        Returns:
            Server: The instantiated and configured server object.

        Raises:
            ValueError: If the server_type is not recognized.
        """
        # Get the learning rate from hyperparameters, fall back to default if needed
        lr = hyperparams.get('learning_rate', self.default_params['learning_rate'])
        if lr is None: # Handle case where get returns None explicitly
             lr = self.default_params['learning_rate']

        # Get algorithm-specific configuration (e.g., layers, reg_param)
        algorithm_params = get_algorithm_config(server_type, self.config.dataset)
        # Add any hyperparameters passed directly (like reg_param if tuned) to algorithm_params
        for key, value in hyperparams.items():
             if key != 'learning_rate': # Avoid overwriting LR used for optimizer
                 algorithm_params[key] = value

        # Create the base configuration for the trainer/clients
        config = self._create_trainer_config(server_type, lr, algorithm_params)

        # Instantiate the global model, criterion, and optimizer
        model, criterion, optimizer = self._create_model(config.learning_rate)

        # Create the initial global model state
        global_model_state = ModelState(
            model=model,
            optimizer=optimizer, # Note: Server doesn't use optimizer, but clients copy it
            criterion=criterion
        )

        # Map server type string to the corresponding server class
        server_mapping = {
            'local': Server, # 'local' uses the base Server for simple local training simulation
            'fedavg': FedAvgServer,
            'fedprox': FedProxServer,
            'pfedme': PFedMeServer,
            'ditto': DittoServer,
            'localadaptation': LocalAdaptationServer,
            'babu': BABUServer,
            'fedlp': FedLPServer,
            'fedlama': FedLAMAServer,
            'pfedla': pFedLAServer,
            'playerfl': LayerPFLServer,
            'playerfl_random': LayerPFLServer, # Uses the same server class as layerpfl
        }

        server_class = server_mapping.get(server_type)
        if server_class is None:
            self.logger.error(f"Unknown server type: {server_type}")
            raise ValueError(f"Unknown server type: {server_type}")

        # Instantiate the server
        server = server_class(config=config, globalmodelstate=global_model_state)
        # Inform the server about its type and whether it's a tuning run
        server.set_server_type(server_type, tuning)

        self.logger.info(f"Created server instance for type: {server_type}")
        return server

    def _add_clients_to_server(self, server: Server, client_dataloaders: Dict[str, Tuple[DataLoader, DataLoader, DataLoader]]):
        """
        Adds clients to the server instance, providing each with its data.

        Args:
            server (Server): The server instance to add clients to.
            client_dataloaders (Dict): A dictionary mapping client IDs to their
                                       (train, val, test) DataLoaders.
        """
        if not client_dataloaders:
             self.logger.warning("Client dataloaders dictionary is empty. No clients will be added.")
             return

        self.logger.info(f"Adding {len(client_dataloaders)} clients to the server...")
        for client_id, loaders in client_dataloaders.items():
            # Create SiteData object encapsulating the loaders for the client
            client_data = self._create_site_data(client_id, loaders)
            # Server's add_client method handles creating the actual Client object
            server.add_client(clientdata=client_data)
        self.logger.info("Finished adding clients.")

    def _create_site_data(self, client_id: str, loaders: Tuple[DataLoader, DataLoader, DataLoader]) -> SiteData:
        """
        Creates a SiteData object for a client.

        Args:
            client_id (str): The unique identifier for the client.
            loaders (Tuple[DataLoader, DataLoader, DataLoader]): A tuple containing the
                train, validation, and test DataLoaders for the client.

        Returns:
            SiteData: The created SiteData object.
        """
        if len(loaders) != 3:
             raise ValueError(f"Expected 3 DataLoaders (train, val, test) for client {client_id}, but got {len(loaders)}.")
        train_loader, val_loader, test_loader = loaders
        return SiteData(
            site_id=client_id,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )


    def _train_and_evaluate(self, server: Server, rounds: int) -> Dict:
        """
        Runs the main federated learning loop (training and validation) for a
        specified number of rounds, followed by final testing if not a tuning run.

        Also handles special final round logic for algorithms like BABU and LocalAdaptation.

        Args:
            server (Server): The configured server instance.
            rounds (int): The total number of communication rounds to perform.

        Returns:
            Dict: A dictionary containing aggregated global metrics and per-site metrics
                  collected during the process. Structure:
                  {'global': {'losses': [...], 'scores': [...]},
                   'sites': {client_id: {'losses': [...], 'scores': [...]}, ...}}
        """
        self.logger.info(f"Starting training and evaluation for {server.server_type} over {rounds} rounds.")
        start_time = time.time()

        # --- Training Rounds ---
        for round_num in range(rounds):
            round_start_time = time.time()
            self.logger.info(f"--- Round {round_num + 1}/{rounds} ---")
            train_loss, val_loss, val_score = server.train_round(round_num)
            round_end_time = time.time()
            self.logger.info(f"Round {round_num + 1} completed. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_score.get('accuracy', -1):.4f}. Time: {round_end_time - round_start_time:.2f}s")
            log_gpu_stats(self.logger, prefix=f"End of Round {round_num + 1}: ") # Log GPU stats periodically

        total_training_time = time.time() - start_time
        self.logger.info(f"All {rounds} training rounds completed in {total_training_time:.2f}s.")

        # --- Final Testing (only if not a tuning run) ---
        if not server.tuning:
            self.logger.info("Performing final testing on the best models...")
            test_start_time = time.time()
            # Server's test_global() evaluates best model on each client's test set
            test_loss, test_score = server.test_global()
            test_end_time = time.time()
            self.logger.info(f"Final Testing completed. Test Loss: {test_loss:.4f}, Test Acc: {test_score.get('accuracy', -1):.4f}. Time: {test_end_time - test_start_time:.2f}s")
        else:
             self.logger.info("Skipping final testing phase (tuning run).")


        # --- Collect Metrics ---
        self.logger.info("Collecting metrics from server and clients...")
        # Determine which metrics to collect (validation for tuning, test for evaluation)
        state_to_use = server.serverstate # Global state
        if server.tuning:
            global_losses, global_scores = state_to_use.val_losses, state_to_use.val_scores
            metric_type = "Validation"
        else:
            global_losses, global_scores = state_to_use.test_losses, state_to_use.test_scores
            metric_type = "Test"

        # Structure to hold all metrics
        metrics = {
            'global': {
                'losses': copy.deepcopy(global_losses), # Use copies to avoid modifying state
                'scores': copy.deepcopy(global_scores)
            },
            'sites': {}
        }

        # Collect per-client metrics
        for client_id, client in server.clients.items():
            # Choose the appropriate client state (personal or global)
            client_state = client.personal_state if client.personal_state is not None else client.global_state

            if server.tuning:
                client_losses, client_scores = client_state.val_losses, client_state.val_scores
            else:
                client_losses, client_scores = client_state.test_losses, client_state.test_scores

            metrics['sites'][client_id] = {
                'losses': copy.deepcopy(client_losses),
                'scores': copy.deepcopy(client_scores)
            }

        self.logger.info(f"Metrics collected ({metric_type}).")
        return metrics
