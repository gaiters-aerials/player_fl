from configs import *
from helper import *
from dataset_processing import DataPreprocessor, UnifiedDataLoader
import models
from losses import MulticlassFocalLoss
from performance_logging import PerformanceLogger
from analytics_server import *
from analytics_clients import *


performance_logger = PerformanceLogger(log_dir='code/layer_metrics/logs/python_logs')

def get_client_logger(client_id, algorithm_type=None):
    """Get client logger with optional algorithm type."""
    return performance_logger.get_logger(f"client_{client_id}", algorithm_type)

def get_server_logger(algorithm_type=None):
    """Get server logger with optional algorithm type."""
    return performance_logger.get_logger("server", algorithm_type)

# Define which algorithms to run analytics on
ALGORITHMS_FOR_ANALYSIS = ['local', 'fedavg'] # Example subset

@dataclass
class AnalyticsConfig:
    """Configuration for the Analytics Pipeline."""
    dataset: str
    num_runs: int = 1              
    results_dir: str = f'{RESULTS_DIR}'
    cpus_for_analysis: int = field(default_factory=lambda: int(os.getenv('SLURM_CPUS_PER_TASK', 4)))


class AnalyticsResultsManager: # Use the standard name
    """Manages loading and saving of analytics experiment results."""
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.analytics_dir = os.path.join(RESULTS_DIR, 'analytics') # Specific subdir
        self.results_filename = f'{dataset}_analytics_results.pkl' # Specific filename
        self.results_path = os.path.join(self.analytics_dir, self.results_filename)
        os.makedirs(self.analytics_dir, exist_ok=True)

    def load_results(self) -> dict:
        """Loads the existing analytics results dictionary."""
        if os.path.exists(self.results_path):
            try:
                with open(self.results_path, 'rb') as f:
                    results = pickle.load(f)
                    return results if isinstance(results, dict) else {}
            except (EOFError, pickle.UnpicklingError):
                print(f"Warning: Analytics results file {self.results_path} is empty/corrupted.")
        return {}

    def save_results(self, results_data: dict):
        """Saves the complete analytics results dictionary."""
        try:
            with open(self.results_path, 'wb') as f:
                pickle.dump(results_data, f)
            print(f"Analytics results saved to {self.results_path}")
        except Exception as e:
            print(f"Error saving analytics results to {self.results_path}: {e}")

    def update_run_results(self, run_number: int, run_results: dict):
        """Loads existing results, updates with the current run, and saves back."""
        existing_results = self.load_results()
        existing_results[f'run_{run_number}'] = run_results
        self.save_results(existing_results)

    def get_completed_runs_count(self) -> int:
        """Determines how many runs are already saved."""
        results = self.load_results()
        run_keys = [k for k in results.keys() if k.startswith('run_')]
        return len(run_keys)

class AnalyticsExperiment:
    """Runs the Federated Learning Analytics Experiment."""

    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.root_dir = ROOT_DIR
        # Use the specialized results manager for analytics
        self.results_manager = AnalyticsResultsManager(
            dataset=self.config.dataset,
        )
        self.default_params = get_parameters_for_dataset(self.config.dataset)
        # Logger specific to analytics
        self.logger = performance_logger.get_logger(config.dataset, 'analytics')

    def run_experiment(self):
        """Main entry point for running the analytics experiment."""
        # No branching needed, directly call the analytics evaluation logic
        self.logger.info(f"Starting Analytics Experiment for dataset: {self.config.dataset}")
        self.logger.info(f"Configuration: {self.config}")
        results = self._run_analytics_evaluation()
        self.logger.info("Analytics Experiment Finished.")
        return results

    def _run_analytics_evaluation(self):
        """Handles running the analytics evaluation for the configured number of runs."""
        existing_results = self.results_manager.load_results()
        completed_runs_count = self.results_manager.get_completed_runs_count()
        total_runs_needed = self.config.num_runs

        self.logger.info(f"Found {completed_runs_count} completed runs. Total runs needed: {total_runs_needed}")

        if completed_runs_count >= total_runs_needed:
            self.logger.info("Target number of analytics runs already completed. Exiting.")
            print("Target number of analytics runs already completed.")
            return existing_results

        # Loop through the required number of runs
        for run in range(completed_runs_count + 1, total_runs_needed + 1):
            run_data = {}
            try:
                set_seeds(run) # Set seed for reproducibility per run
                self.logger.info(f"Starting Analytics Run {run}/{total_runs_needed}")
                print(f"--- Starting Analytics Run {run}/{total_runs_needed} for {self.config.dataset} ---")

                # Call the method for a single run's logic
                run_data = self._analytics_evaluation(run_number=run)

                # Save results for this run incrementally
                # Use update_run_results which handles loading/saving
                self.results_manager.update_run_results(run, run_data)
                self.logger.info(f"Successfully completed and saved Analytics Run {run}")
                print(f"--- Finished Analytics Run {run}/{total_runs_needed} ---")

            except Exception as e:
                self.logger.error(f"Analytics Run {run} failed: {str(e)}", exc_info=True)
                print(f"!!! Analytics Run {run} failed: {e} !!!")
                # Save partial data if available
                if run_data:
                     pass
                     #self.results_manager.update_run_results(run, run_data)
                     #self.logger.warning(f"Saved partial results for failed run {run}.")
                break # Stop experiment if a run fails critically
            finally:
                cleanup_gpu()

        # Return the final dictionary containing all run results
        return self.results_manager.load_results()

    def _analytics_evaluation(self, run_number: int) -> dict:
        """Performs the analysis logic for a single run across specified algorithms."""
        run_tracking = {}
        # Initialize data once for this run
        client_dataloaders = self._initialize_experiment(self.default_params['batch_size'])

        for server_type in ALGORITHMS_FOR_ANALYSIS:
            self.logger.info(f"Run {run_number}: Analyzing {server_type}")
            print(f"  Analyzing {server_type}...")
            server = None # Ensure server is reset
            try:
                lr = self.default_params['learning_rate']
                hyperparams = {'learning_rate': lr}

                # 2. Create Server Instance (using consistent naming)
                # This method now needs to create AnalyticsServer
                server = self._create_server_instance(server_type, hyperparams, tuning=False) # False indicates not tuning LR

                # 3. Add Clients
                self._add_clients_to_server(server, client_dataloaders)

                # 4. Train and Collect Analytics (replaces _train_and_evaluate)
                analysis_results = self._train_and_analyze(server, run_number)

                # Store the collected analytics results
                run_tracking[server_type] = analysis_results
                self.logger.info(f"Run {run_number}: Completed analysis for {server_type}")

            except Exception as e:
                 self.logger.error(f"Run {run_number}: Analysis failed for {server_type}: {str(e)}", exc_info=True)
                 print(f"    Analysis for {server_type} failed: {e}")
            finally:
                if server: del server
                cleanup_gpu()
                time.sleep(1) # Small delay

        return run_tracking # Return results dict for this run
    
    def _train_and_analyze(self, server: AnalyticsServer, seed: int) -> dict:
        """Handles training and running analysis hooks for a given server."""
        self.logger.info(f"Starting training for {server.server_type} ({server.config.rounds} rounds)...")
        num_rounds = server.config.rounds
        for round_num in range(num_rounds):
            try:
                _ = server.train_round(round_num)

                if (round_num + 1) % 5 == 0: # Log progress less frequently
                    print(f"      Training round {round_num + 1}/{num_rounds} completed for {server.server_type}.")

            except Exception as e:
                 self.logger.error(f"Training failed at round {round_num + 1} for {server.server_type}: {e}", exc_info=True)
                 print(f"      ERROR in training round {round_num + 1}. Stopping training.")
                 # Decide if partial analysis is useful or just raise error
                 raise # Re-raise to indicate failure for this server_type

        self.logger.info(f"Training finished for {server.server_type}.")
        # Return the collected analysis results
        return copy.deepcopy(server.analysis_results)


    def _initialize_experiment(self, batch_size: int) -> dict:
        """Initializes data loading and preprocessing."""
        self.logger.debug("Initializing data pipeline...")
        preprocessor = DataPreprocessor(self.config.dataset, batch_size)
        loader = UnifiedDataLoader(root_dir=self.root_dir, dataset_name=self.config.dataset)
        dataset_df = loader.load()
        client_data_loaders = preprocessor.process_client_data(dataset_df)
        self.logger.debug("Data pipeline initialized.")
        return client_data_loaders

    def _create_trainer_config(self, learning_rate: float, algorithm_params: dict = None) -> TrainerConfig:
        """Creates the TrainerConfig based on defaults and inputs."""
        rounds = self.default_params.get('rounds_analytics', self.default_params['rounds'])

        return TrainerConfig(
            dataset_name=self.config.dataset,
            device=DEVICE,
            learning_rate=learning_rate,
            batch_size=self.default_params['batch_size'],
            epochs=self.default_params['epochs_per_round'],
            rounds=rounds,
            num_clients=self.default_params['num_clients'],
            requires_personal_model=False,
            algorithm_params=algorithm_params,
            num_cpus=self.config.cpus_for_analysis
        )

    def _create_model_essentials(self, learning_rate: float):
        """Creates model, criterion, optimizer."""
        classes = self.default_params['classes']
        model = getattr(models, self.config.dataset)(classes)

        criterion_map = {
             'EMNIST': nn.CrossEntropyLoss(), 'CIFAR': nn.CrossEntropyLoss(),
             "FMNIST": nn.CrossEntropyLoss(),
             "ISIC": MulticlassFocalLoss(num_classes=classes, alpha=[0.87868852, 0.88131148, 0.82793443, 0.41206557], gamma=1),
             "Sentiment": nn.CrossEntropyLoss(),
             "Heart": MulticlassFocalLoss(num_classes=classes, alpha=[0.12939189, 0.18108108, 0.22331081, 0.22364865, 0.24256757], gamma=3),
             "mimic": MulticlassFocalLoss(num_classes=classes, alpha=[0.15, 0.85], gamma=1),
        }
        criterion = criterion_map.get(self.config.dataset, nn.CrossEntropyLoss())

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=1e-4
        )
        return model, criterion, optimizer

    def _create_server_instance(self, server_type: str, hyperparams: dict, tuning: bool) -> AnalyticsServer:
        """Creates an AnalyticsServer instance configured for the specific analysis task."""
        lr = hyperparams.get('learning_rate')
        algorithm_params = get_algorithm_config(server_type, self.config.dataset)

        # Create config (uses analytics rounds etc.)
        config = self._create_trainer_config(lr, algorithm_params)
        self.logger.debug(f"Creating server {server_type} with config: {config}")

        # Create model state
        model, criterion, optimizer = self._create_model_essentials(config.learning_rate)
        globalmodelstate = ModelState(model=model, optimizer=optimizer, criterion=criterion)

        server_mapping = {
            'local': AnalyticsServer,
            'fedavg': AnalyticsFedAvgServer}
        server_class = server_mapping[server_type]

        server = server_class(
            config=config,
            globalmodelstate=globalmodelstate,
        )

        # Set server type attributes (useful for internal logic/logging)
        if hasattr(server, 'set_server_type'):
             server.set_server_type(server_type, tuning=tuning) # Pass tuning flag
        else:
             self.logger.warning(f"AnalyticsServer instance for {server_type} lacks set_server_type method.")

        self.logger.info(f"Created AnalyticsServer for base algorithm: {server_type}")
        return server

    def _add_clients_to_server(self, server: AnalyticsServer, client_dataloaders: dict):
        """Adds clients to the server (AnalyticsClients will be created)."""
        self.logger.debug(f"Adding {len(client_dataloaders)} clients...")
        for client_id, loaders in client_dataloaders.items():
            client_site_data = self._create_site_data(client_id, loaders)
            server.add_client(clientdata=client_site_data)
        self.logger.debug("Clients added.")

    def _create_site_data(self, client_id, loaders):
        """Creates SiteData object from data loaders."""
        # Assuming loaders is a tuple/list: (train_loader, val_loader, test_loader)
        return SiteData(
            site_id=client_id,
            train_loader=loaders[0],
            val_loader=loaders[1],
            test_loader=loaders[2]
        )