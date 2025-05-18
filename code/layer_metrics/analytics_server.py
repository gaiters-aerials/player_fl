ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
import sys
sys.path.append(f'{ROOT_DIR}/code')
from helper import move_to_device, cleanup_gpu 
from configs import *
from clients import SiteData, ModelState, MetricsCalculator
from analytics_clients import TrainerConfig, AnalyticsClient 
from servers import Server
from layer_analytics  import *


# Base Server classes (keep as is unless they interfere)
class AnalyticsServer(Server):
    """
    Extends a Federated Learning Server to orchestrate model analysis across clients.
    FIX: Uses RandomSampler and fractional sampling for probe data.
    """
    def __init__(self, config: TrainerConfig, globalmodelstate: ModelState):
        super().__init__(config, globalmodelstate)
        self.analysis_results = {'grad_hess': {}, 'similarity': {}}
        self.probe_data_batch = None
        self.num_cpus_analysis = config.num_cpus if hasattr(config, 'num_cpus') else 4
        # Store config parameters needed later
        self.num_sites_config = config.num_clients # Assuming num_clients is the total expected
        self.batch_size_config = config.batch_size

    def _create_client(self, clientdata: SiteData, modelstate: ModelState, personal_model=False) -> AnalyticsClient:
        """Creates the fixed AnalyticsClient."""
        print(f"AnalyticsServer: Creating AnalyticsClient for site {clientdata.site_id}")
        # Ensure the config passed has requires_personal_model if needed by client __init__
        if not hasattr(self.config, 'requires_personal_model'):
            # Add a default if missing from original config for AnalyticsClient init
             print("Warning: TrainerConfig missing 'requires_personal_model', defaulting to False for client creation.")
             self.config.requires_personal_model = False

        return AnalyticsClient(
            config=self.config,
            data=clientdata,
            modelstate=modelstate.copy(),
            metrics_calculator=MetricsCalculator(self.config.dataset_name),
            personal_model=self.config.requires_personal_model # Pass the flag
        )

    # --- MODIFIED Probe Data Preparation ---
    def _prepare_probe_data(self, seed: int = 42):
        """
        Creates a consistent data batch for activation similarity analysis by
        sampling from each client's training data using RandomSampler and fractional sampling.
        """
        if self.probe_data_batch is not None:
            print("Server: Probe data batch already prepared.")
            return

        print(f"Server: Preparing probe data batch for similarity analysis (seed={seed})...")
        if not self.clients:
            print("Warning: No clients available to prepare probe data.")
            return

        all_features_list = []
        all_labels_list = []
        feature_type = None
        num_active_clients = len(self.clients) # Use actual number of clients connected

        # Determine samples per site based on actual clients and configured batch size
        # Mimic: samples = len(batch_label) // self.num_sites
        # We use the configured batch_size as a proxy for the batch length
        samples_per_site = 32 // num_active_clients if num_active_clients > 0 else 0
        if samples_per_site == 0:
            print(f"Warning: Calculated samples_per_site is 0 (batch_size={32}, clients={num_active_clients}). Setting to 1.")
            samples_per_site = 1

        print(f"Server: Aiming for {samples_per_site} samples per client for probe data.")

        for client_id, client in self.clients.items():
            if client.data.train_loader is None or client.data.train_loader.dataset is None or len(client.data.train_loader.dataset) == 0:
                 print(f"Warning: Client {client_id} train_loader or dataset empty/None. Skipping for probe data.")
                 continue

            try:
                sampler = RandomSampler(client.data.train_loader.dataset, replacement=True, num_samples=client.data.train_loader.batch_size)
                # Use a consistent generator for the sampler across clients for this specific probe data prep
                g = torch.Generator()
                g.manual_seed(seed + int(client.data.site_id.split('_')[-1])) # Seed per client offset by site_id

                random_loader = DataLoader(
                    dataset=client.data.train_loader.dataset,
                    batch_size=32, # Limit batch size for probing due to memory
                    sampler=RandomSampler(client.data.train_loader.dataset, replacement=True, generator=g), # Seeded sampler
                    collate_fn=client.data.train_loader.collate_fn,
                    num_workers=0,
                    pin_memory=client.data.train_loader.pin_memory,
                    generator=g # Pass generator to loader too
                )
                batch_features, batch_labels = next(iter(random_loader))
                del random_loader, sampler, g # Cleanup

                # Determine feature type from first successful client
                if feature_type is None:
                    feature_type = type(batch_features)

                # Take specified number of samples (fraction of the random batch)
                num_samples_to_take = min(samples_per_site, len(batch_labels))
                if num_samples_to_take == 0:
                    print(f"Warning: Client {client_id} yielded 0 samples to take for probe data.")
                    continue

                # Slice the features and labels
                if feature_type in (tuple, list):
                    # Handle tuple features (e.g., input_ids, attention_mask)
                    sampled_features = tuple(f[:num_samples_to_take].cpu() for f in batch_features) # Move to CPU
                elif feature_type == torch.Tensor:
                    sampled_features = batch_features[:num_samples_to_take].cpu() # Move to CPU
                else:
                    print(f"Warning: Unsupported feature type {feature_type} for client {client_id}. Skipping.")
                    continue

                sampled_labels = batch_labels[:num_samples_to_take].cpu() # Move to CPU

                all_features_list.append(sampled_features)
                all_labels_list.append(sampled_labels)

            except StopIteration:
                print(f"Warning: Client {client_id} random_loader failed (StopIteration) during probe data prep.")
                continue
            except Exception as e:
                print(f"Error getting probe data from client {client_id}: {e}")
                traceback.print_exc()
                continue

        if not all_labels_list:
             print("Error: Could not collect any probe data samples from clients.")
             return

        # Combine samples into a single batch (on CPU)
        combined_labels = torch.cat(all_labels_list, dim=0)

        if feature_type in (tuple, list):
            num_feature_elements = len(all_features_list[0])
            combined_features_tuple = []
            for i in range(num_feature_elements):
                # Ensure all tensors in the tuple element are concatenated
                combined_features_tuple.append(torch.cat([f[i] for f in all_features_list], dim=0))
            combined_features = tuple(combined_features_tuple)
        elif feature_type == torch.Tensor:
            combined_features = torch.cat(all_features_list, dim=0)
        else:
            print("Error: Could not combine features due to unsupported type.")
            return

        self.probe_data_batch = (combined_features, combined_labels)
        print(f"Server: Probe data batch created with {len(combined_labels)} samples from {len(all_labels_list)} clients.")
    
    def layer_metrics_hook(self, round_num: int):
        is_first_round = (round_num == 0)
        is_last_round = (round_num == self.config.rounds - 1) 
        if is_first_round or is_last_round:
            round_identifier = 'first' if is_first_round else 'final'
            seed = round_num
            self.run_analysis(round_identifier=round_identifier, seed=seed)
        return

    def run_analysis(self, round_identifier: str, seed: int): # Require seed
        """
        Runs the analysis pipeline for a given round/identifier.
        """
        print(f"--- Server: Starting Analysis for '{round_identifier}' (seed={seed}) ---")
        start_time = time.time()

        if not self.clients:
            print("Server: No clients to run analysis on.")
            return

        # --- 1. Prepare Probe Data (using seed) ---
        # Only prepares if self.probe_data_batch is None
        self._prepare_probe_data(seed=seed)

        # --- 2. Local Metrics Calculation (Hessian etc.) ---
        print(f"Server: Requesting local metrics from {len(self.clients)} clients...")
        current_local_metrics = {}
        for client_id, client in self.clients.items():
            try:
                # Pass the SAME seed to each client for this analysis round
                # This seed controls both the data sample AND the HVP vector 'v'
                metrics_df = client.run_local_metrics_calculation(
                    seed=seed
                )
                current_local_metrics[client_id] = metrics_df if metrics_df is not None else pd.DataFrame()
            except Exception as e:
                print(f"Error getting local metrics from client {client_id}: {e}")
                traceback.print_exc()
                current_local_metrics[client_id] = pd.DataFrame()

        # Store results for this round
        self.analysis_results['grad_hess'][round_identifier] = current_local_metrics
        print("Server: Local metrics collected.")

        # --- 3. Activation Similarity Calculation ---
        if self.probe_data_batch is None:
            print("Server: Skipping similarity calculation - probe data not available.")
            self.analysis_results['similarity'][round_identifier] = {} # Store empty dict
        else:
            print(f"Server: Requesting activations from {len(self.clients)} clients...")
            all_client_activations: Dict[str, List[Tuple[str, np.ndarray]]] = {}
            for client_id, client in self.clients.items():
                try:
                    # The probe data batch itself is now consistent due to _prepare_probe_data
                    activations = client.run_activation_extraction(
                        probe_data_batch=self.probe_data_batch,
                    )
                    # Filter out potential None or empty lists robustly
                    if activations: # Check if list is not empty
                         all_client_activations[client_id] = activations
                    else:
                         print(f"Warning: Client {client_id} returned no activations.")
                         all_client_activations[client_id] = [] # Ensure key exists but is empty list

                except Exception as e:
                    print(f"Error getting activations from client {client_id}: {e}")
                    traceback.print_exc()
                    all_client_activations[client_id] = []

            print("Server: Calculating activation similarity...")
            # Filter dict for clients that actually provided activations
            valid_activations = {cid: act for cid, act in all_client_activations.items() if act}

            if len(valid_activations) >= 2:
                # Pass the probe batch ONLY for potential mask extraction inside calculate_activation_similarity
                similarity_results = calculate_activation_similarity(
                    activations_dict=valid_activations,
                    probe_data_batch=self.probe_data_batch,
                    cpus=self.num_cpus_analysis
                    )
                self.analysis_results['similarity'][round_identifier] = similarity_results
                print("Server: Activation similarity calculated.")
            else:
                print(f"Server: Skipping similarity calculation - only {len(valid_activations)} clients provided valid activations.")
                self.analysis_results['similarity'][round_identifier] = {} # Store empty dict

        end_time = time.time()
        print(f"--- Server: Analysis for '{round_identifier}' finished ({end_time - start_time:.2f} sec) ---")


    def save_analysis_results(self, filepath: str):
        """Saves the collected analysis results to a pickle file."""
        print(f"Server: Saving analysis results to {filepath}...")
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # Save the original structure (with DataFrames)
            with open(filepath, 'wb') as f:
                 pickle.dump(self.analysis_results, f)

            print("Server: Analysis results saved successfully.")
        except Exception as e:
            print(f"Error saving analysis results: {e}")
            traceback.print_exc()


class AnalyticsFLServer(AnalyticsServer):
    """Base federated learning server with FedAvg implementation."""
    def aggregate_models(self):
        """Standard FedAvg aggregation."""
        # Reset global model parameters
        for param in self.serverstate.model.parameters():
            param.data.zero_()
            
        # Aggregate parameters
        for client in self.clients.values():
            client_model = client.personal_state.model if self.personal else client.global_state.model
            for g_param, c_param in zip(self.serverstate.model.parameters(), client_model.parameters()):
                g_param.data.add_(c_param.data * client.data.weight)

    def distribute_global_model(self):
        """Distribute global model to all clients."""
        global_state = self.serverstate.model.state_dict()
        for client in self.clients.values():
            client.set_model_state(global_state)

class AnalyticsFedAvgServer(AnalyticsFLServer):
    """FedAvg server implementation."""
    pass