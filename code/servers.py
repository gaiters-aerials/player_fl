"""
Defines the server-side logic for various federated learning algorithms.

Includes:
- A base `Server` class providing core functionalities like client management,
  round orchestration (training, validation, aggregation, distribution),
  and metric tracking.
- An `FLServer` class inheriting from `Server` and implementing standard
  FedAvg aggregation and distribution.
- Specialized server classes inheriting from `FLServer` or `Server` to implement
  algorithm-specific aggregation, distribution, or round logic
  (e.g., `FedProxServer`, `DittoServer`, `LayerServer`, `pFedLAServer`).
"""
from helper import *       # Utility functions
from configs import *      # Global configurations
from clients import *      # Client classes (needed for instantiation and type hints)
from models import HyperNetwork # HyperNetwork model used by pFedLAServer


class Server:
    """
    Base class for the central server in a federated learning system.

    Manages clients, orchestrates communication rounds, holds the global model state,
    and tracks aggregated performance metrics. Subclasses implement specific
    aggregation and distribution strategies for different FL algorithms.
    """
    def __init__(self, config: TrainerConfig, globalmodelstate: ModelState):
        """
        Initializes the base Server.

        Args:
            config (TrainerConfig): Configuration object shared across server and clients.
            globalmodelstate (ModelState): The initial state of the global model, including
                                           the model architecture, optimizer template, and criterion.
        """
        self.config = config
        self.device = config.device
        # Flag indicating if clients for this algorithm need a separate personal model state
        self.personal = config.requires_personal_model
        self.clients: Dict[str, Client] = {} # Dictionary mapping client_id to Client object
        # Server's perspective of the global model state
        self.serverstate = globalmodelstate
        # Ensure server's model and best model snapshot are on the correct device
        if self.serverstate.model is not None:
            self.serverstate.model = self.serverstate.model.to(self.device)
        if self.serverstate.best_model is not None:
            self.serverstate.best_model = self.serverstate.best_model.to(self.device)
        else:
             # Initialize best model if None
             self.serverstate.best_model = copy.deepcopy(self.serverstate.model).to(self.device)

        # Placeholder attributes set by set_server_type
        self.server_type: Optional[str] = None
        self.tuning: Optional[bool] = None


    def set_server_type(self, name: str, tuning: bool):
        """
        Sets the server type name and tuning status.

        Args:
            name (str): The name of the FL algorithm being run (e.g., 'fedavg').
            tuning (bool): True if this is a hyperparameter tuning run, False otherwise.
        """
        self.server_type = name
        self.tuning = tuning
        print(f"Server initialized as type: {self.server_type}, Tuning: {self.tuning}")

    def _create_client(self, clientdata: SiteData, modelstate: ModelState, personal_model: bool) -> Client:
        """
        Factory method to create a client instance.

        Base implementation creates a standard `Client`. Subclasses for specific algorithms
        (e.g., FedProxServer) override this to create their corresponding client types
        (e.g., FedProxClient).

        Args:
            clientdata (SiteData): DataLoaders and metadata for the client.
            modelstate (ModelState): The initial model state to provide to the client
                                     (usually a copy of the server's current global state).
            personal_model (bool): Flag indicating if the client should maintain a
                                   separate personal model state.

        Returns:
            Client: An instance of the appropriate Client class.
        """
        print(f"Creating base Client for site {clientdata.site_id}")
        return Client(
            config=self.config,
            data=clientdata,
            modelstate=modelstate.copy(), # Pass a copy to ensure client has independent state
            metrics_calculator=MetricsCalculator(self.config.dataset_name),
            personal_model=personal_model
        )

    def add_client(self, clientdata: SiteData):
        """
        Creates and adds a new client to the federation managed by the server.

        Uses the `_create_client` factory method. After adding, updates the
        aggregation weights for all clients.

        Args:
            clientdata (SiteData): DataLoaders and metadata for the client to be added.
        """
        print(f"Adding client: {clientdata.site_id}")
        # Use the factory method to create the client instance
        client = self._create_client(
            clientdata=clientdata,
            modelstate=self.serverstate, # Provide initial global model state
            personal_model=self.personal # Use server's flag
        )

        # Store the client
        self.clients[clientdata.site_id] = client
        # Recalculate weights based on dataset sizes
        self._update_client_weights()
        print(f"Client {clientdata.site_id} added. Total clients: {len(self.clients)}")

    def _update_client_weights(self):
        """
        Recalculates the aggregation weight for each client.

        Weights are typically based on the number of training samples relative
        to the total number of samples across all clients. Handles division by zero.
        """
        total_samples = sum(client.data.num_samples for client in self.clients.values())

        if total_samples == 0:
            print("Warning: Total samples across clients is 0. Setting weights equally.")
            num_clients = len(self.clients)
            weight = 1.0 / num_clients if num_clients > 0 else 1.0
            for client in self.clients.values():
                client.data.weight = weight
        else:
            for client in self.clients.values():
                client.data.weight = client.data.num_samples / total_samples
                # print(f"Client {client.data.site_id} weight set to {client.data.weight:.4f} ({client.data.num_samples} samples)")


    def _aggregate_scores(self, score_dict: Dict[str, float], client_metrics: Dict[str, float], weight: float) -> Dict[str, float]:
        """
        Helper function to aggregate weighted metrics from a client into a central dictionary.

        Args:
            score_dict (Dict[str, float]): The dictionary holding the current aggregated scores.
            client_metrics (Dict[str, float]): The metrics dictionary from a single client.
            weight (float): The weight associated with the client's metrics.

        Returns:
            Dict[str, float]: The updated score_dict with the client's weighted metrics added.
        """
        for metric_name, value in client_metrics.items():
            if metric_name not in score_dict:
                score_dict[metric_name] = 0.0
            # Accumulate weighted score
            score_dict[metric_name] += value * weight
        return score_dict

    def train_round(self, round_num: Optional[int] = None) -> Tuple[float, float, Dict[str, float]]:
        """
        Executes one full communication round of federated learning.

        1. Instructs all clients to train locally (`client.train`).
        2. Instructs all clients to validate (`client.validate`).
        3. Aggregates weighted training and validation metrics from clients.
        4. Calls `aggregate_models` (implemented by subclasses) to update the global model.
        5. Calls `distribute_global_model` (implemented by subclasses) to send the updated model back to clients.
        6. Updates the server's record of the best global model based on aggregated validation loss.

        Returns:
            Tuple[float, float, Dict[str, float]]: A tuple containing:
                - Aggregated average training loss for the round.
                - Aggregated average validation loss for the round.
                - Dictionary of aggregated validation scores for the round.
        """
        agg_train_loss = 0.0
        agg_val_loss = 0.0
        agg_val_score = {}

        # --- Client Training and Validation Phase ---
        client_ids = list(self.clients.keys()) # Get IDs to iterate over
        print(f"Starting training and validation for {len(client_ids)} clients...")
        for client_id in client_ids:
             client = self.clients[client_id]
             # 1. Local Training
             # print(f"  Training client {client_id}...")
             client_train_loss = client.train(self.personal) # `personal` flag passed to client
             # 2. Local Validation
             # print(f"  Validating client {client_id}...")
             client_val_loss, client_val_score = client.validate(self.personal) # `personal` flag passed to client

             # 3. Accumulate Weighted Metrics
             weight = client.data.weight
             agg_train_loss += client_train_loss * weight
             agg_val_loss += client_val_loss * weight
             agg_val_score = self._aggregate_scores(agg_val_score, client_val_score, weight)
             # print(f"  Client {client_id} finished. Val Loss: {client_val_loss:.4f}, Val Acc: {client_val_score.get('accuracy', -1):.4f}")

        
        # --- Server Aggregation and Distribution Phase ---
        # 4. Aggregate models from clients (specific logic in subclasses)
        # print("Aggregating client models...")
        self.layer_metrics_hook(round_num) # This is only implemented for the Analytics server class in layer_metrics/
        self.aggregate_models()
        # 5. Distribute the updated global model (specific logic in subclasses)
        # print("Distributing updated global model...")
        self.distribute_global_model()


        # --- Track Server-Side Metrics ---
        self.serverstate.train_losses.append(agg_train_loss)
        self.serverstate.val_losses.append(agg_val_loss)
        self.serverstate.val_scores.append(agg_val_score)

        # --- Update Best Global Model Snapshot ---
        # 6. Check if the aggregated validation loss improved
        if agg_val_loss < self.serverstate.best_loss:
            self.serverstate.best_loss = agg_val_loss
            # Save a copy of the *current* aggregated global model as the best
            self.serverstate.best_model = copy.deepcopy(self.serverstate.model).to(self.device)
            print(f"Server: New best global model saved with validation loss {agg_val_loss:.4f}")

        return agg_train_loss, agg_val_loss, agg_val_score

    def test_global(self) -> Tuple[float, Dict[str, float]]:
        """
        Evaluates the final performance of the best global model across all clients' test sets.

        Instructs each client to evaluate its `best_model` snapshot (obtained during
        validation) on its local test data. Aggregates the weighted test metrics.

        Note: This method relies on clients having stored their best model during `validate`.
        The `personal` flag determines whether clients test their best *personal* or best *global* snapshot.

        Returns:
            Tuple[float, Dict[str, float]]: A tuple containing:
                - Aggregated average test loss.
                - Dictionary of aggregated test scores.
        """
        agg_test_loss = 0.0
        agg_test_score = {}
        print(f"Starting final global testing on {len(self.clients)} clients...")

        for client_id, client in self.clients.items():
            # Instruct client to test using its saved best model
            client_test_loss, client_test_score = client.test(self.personal) # Use `personal` flag

            # Aggregate weighted results
            weight = client.data.weight
            agg_test_loss += client_test_loss * weight
            agg_test_score = self._aggregate_scores(agg_test_score, client_test_score, weight)
            # print(f"  Client {client_id} Test Loss: {client_test_loss:.4f}, Test Acc: {client_test_score.get('accuracy', -1):.4f}")


        # Store aggregated test results in server state (usually happens once)
        self.serverstate.test_losses.append(agg_test_loss)
        self.serverstate.test_scores.append(agg_test_score)

        print(f"Global testing finished. Aggregated Test Loss: {agg_test_loss:.4f}, Aggregated Test Accuracy: {agg_test_score.get('accuracy', -1):.4f}")
        return agg_test_loss, agg_test_score

    # --- Methods to be implemented by subclasses ---

    def layer_metrics_hook(self, round_num: Optional[int] = None):
        """
        Placeholder for layer analysis 

        Subclass Analytics server overrides this method
        """
        return # Base implementation does nothing

    def aggregate_models(self):
        """
        Placeholder for model aggregation logic.

        Subclasses (e.g., `FedAvgServer`, `LayerServer`) must override this method
        to implement their specific strategy for combining client model updates
        into the server's global model.
        """
        print("Warning: Base Server aggregate_models called. No aggregation performed.")
        return # Base implementation does nothing

    def distribute_global_model(self):
        """
        Placeholder for model distribution logic.

        Subclasses (e.g., `FedAvgServer`, `LayerServer`) must override this method
        to implement how the updated global model is sent back to the clients.
        """
        print("Warning: Base Server distribute_global_model called. No distribution performed.")
        return # Base implementation does nothing


# --- Server Implementations for Specific Algorithms ---

class FLServer(Server):
    """
    Federated Learning Server implementing standard FedAvg aggregation and distribution.

    Inherits from `Server` and provides concrete implementations for
    `aggregate_models` and `distribute_global_model` following the FedAvg algorithm.
    """
    def aggregate_models(self):
        """
        Performs Federated Averaging (FedAvg) aggregation.

        Calculates the weighted average of parameters from all client models
        (either global or personal state depending on `self.personal`)
        and updates the server's global model.
        """
        # print("Performing FedAvg aggregation...")
        # Temporarily store the aggregated parameters to avoid modifying in place during loop
        global_params = OrderedDict()
        model_params = self.serverstate.model.named_parameters() # Get iterator once

        # Initialize aggregated params with zeros
        for name, param in model_params:
            global_params[name] = torch.zeros_like(param.data, device=self.device)

        # Weighted summation of client parameters
        total_weight = 0.0 # To verify weights sum to 1 (or handle cases where they don't)
        for client_id, client in self.clients.items():
            # Determine which client model state to use based on the algorithm's requirement
            client_state = client.get_client_state(personal=self.personal)
            client_weight = client.data.weight
            total_weight += client_weight

            # Iterate through parameters and add weighted client param to aggregation buffer
            for name, client_param in client_state.model.named_parameters():
                if name in global_params:
                    # Ensure client param is on the correct device before adding
                    global_params[name].add_(client_param.data.to(self.device) * client_weight)
                else:
                    # This shouldn't happen if all clients have the same model structure
                    print(f"Warning: Parameter '{name}' from client {client_id} not found in global model during aggregation.")

        # Check if weights summed reasonably close to 1
        if not np.isclose(total_weight, 1.0, atol=1e-6):
             print(f"Warning: Client weights sum to {total_weight}, not 1.0 during aggregation.")
             # Option: Renormalize if necessary, or just proceed. Proceeding for now.


        # Load the aggregated parameters into the server's global model
        self.serverstate.model.load_state_dict(global_params)
        # print("FedAvg aggregation complete.")


    def distribute_global_model(self):
        """
        Distributes the server's current global model state to all clients.

        Each client's `set_model_state` method is called with the state dictionary
        of the server's global model.
        """
        # print("Distributing global model to clients...")
        # Get the state dict of the *current* server model
        global_state_dict = self.serverstate.model.state_dict()

        for client_id, client in self.clients.items():
             # Create a copy to ensure each client gets an independent state dict object if needed,
             # although load_state_dict typically copies the data anyway.
             state_dict_copy = {k: v.clone().detach() for k, v in global_state_dict.items()}
             # Client's method handles loading the state
             client.set_model_state(state_dict_copy) # Base client loads the full state
        # print("Global model distribution complete.")


class FedAvgServer(FLServer):
    """
    Server implementation for the standard Federated Averaging (FedAvg) algorithm.

    Uses the aggregation and distribution logic inherited directly from `FLServer`.
    """
    # No specific overrides needed, inherits FedAvg logic from FLServer.
    pass


class FedProxServer(FLServer):
    """
    Server implementation for the FedProx algorithm.

    Uses standard FedAvg aggregation (`FLServer.aggregate_models`) and distribution
    (`FLServer.distribute_global_model`). Its main distinction is that it creates
    `FedProxClient` instances, which implement the proximal term logic locally.
    """
    def _create_client(self, clientdata: SiteData, modelstate: ModelState, personal_model: bool = False) -> FedProxClient:
        """
        Overrides the factory method to create `FedProxClient` instances.

        Args:
            clientdata (SiteData): Client's data.
            modelstate (ModelState): Initial global model state.
            personal_model (bool): Should be False for standard FedProx.

        Returns:
            FedProxClient: An instance of the FedProx client.
        """
        print(f"Creating FedProxClient for site {clientdata.site_id}")
        # Ensure personal_model is False for FedProx
        if personal_model:
            print("Warning: FedProxServer forcing personal_model=False for client creation.")
        return FedProxClient(
            config=self.config,
            data=clientdata,
            modelstate=modelstate.copy(), # Pass a copy
            metrics_calculator=MetricsCalculator(self.config.dataset_name),
            personal_model=False # FedProx operates on the global model state
        )

class PFedMeServer(FLServer):
    """
    Server implementation for the pFedMe algorithm.

    Uses standard FedAvg aggregation (`FLServer.aggregate_models`) for the server's
    auxiliary global model (`w` in the paper, represented by `self.serverstate.model`).
    The personalization happens within the `PFedMeClient` instances, which train
    their personalized models (`theta`).
    """
    def _create_client(self, clientdata: SiteData, modelstate: ModelState, personal_model: bool = True) -> PFedMeClient:
        """
        Overrides the factory method to create `PFedMeClient` instances.

        Ensures that clients are created with `personal_model=True`.

        Args:
            clientdata (SiteData): Client's data.
            modelstate (ModelState): Initial global model state (represents 'w').
            personal_model (bool): Should be True for pFedMe.

        Returns:
            PFedMeClient: An instance of the pFedMe client.
        """
        print(f"Creating pFedMeClient for site {clientdata.site_id}")
        # Ensure personal_model is True for pFedMe
        if not personal_model:
            print("Warning: pFedMeServer forcing personal_model=True for client creation.")
        return PFedMeClient(
            config=self.config,
            data=clientdata,
            modelstate=modelstate.copy(), # Pass a copy (clients create their own personal copy from this)
            metrics_calculator=MetricsCalculator(self.config.dataset_name),
            personal_model=True # pFedMe requires personal models
        )

class DittoServer(FLServer):
    """
    Server implementation for the Ditto algorithm.

    Ditto involves two phases per round managed by the server:
    1. Global Model Update: Clients train their global model copy, server aggregates (FedAvg).
    2. Personal Model Update: Clients train their personalized model using Ditto's regularization.
    The server aggregates the global models but only distributes the global model.
    Personal models remain local to clients.
    """
    def _create_client(self, clientdata: SiteData, modelstate: ModelState, personal_model: bool = True) -> DittoClient:
        """
        Overrides the factory method to create `DittoClient` instances.

        Ensures clients are created with `personal_model=True`.

        Args:
            clientdata (SiteData): Client's data.
            modelstate (ModelState): Initial global model state.
            personal_model (bool): Should be True for Ditto.

        Returns:
            DittoClient: An instance of the Ditto client.
        """
        print(f"Creating DittoClient for site {clientdata.site_id}")
        # Ensure personal_model is True for Ditto
        if not personal_model:
            print("Warning: DittoServer forcing personal_model=True for client creation.")
        return DittoClient(
            config=self.config,
            data=clientdata,
            modelstate=modelstate.copy(), # Pass a copy
            metrics_calculator=MetricsCalculator(self.config.dataset_name),
            personal_model=True # Ditto requires personal models
        )

    def train_round(self, round_num: Optional[int] = None) -> Tuple[float, float, Dict[str, float]]:
        """
        Executes one communication round for Ditto.

        Includes separate steps for global model training/aggregation and
        personal model training. Tracks and returns metrics based on the
        *personal* model's validation performance after both steps.

        Returns:
            Tuple[float, float, Dict[str, float]]: Aggregated personal model train loss,
                                                   validation loss, and validation scores.
        """
        # --- 1. Global Model Update Phase ---
        print("Ditto Round: Starting Global Model Update Phase")
        agg_global_train_loss = 0.0
        agg_global_val_loss = 0.0
        agg_global_val_score = {} # Track global model performance for potential best model update

        for client_id, client in self.clients.items():
            # Train client's global model state (personal=False)
            _ = client.train(personal=False)
            # Validate client's global model state
            client_global_val_loss, client_global_val_score = client.validate(personal=False)

            # Aggregate global model validation metrics (weighted)
            weight = client.data.weight
            # We don't track global train loss explicitly here, focus on val loss for best model
            agg_global_val_loss += client_global_val_loss * weight
            agg_global_val_score = self._aggregate_scores(agg_global_val_score, client_global_val_score, weight)

        # Aggregate client global models into the server's global model (FedAvg)
        self.layer_metrics_hook() # This is only implemented for the Analytics server class in layer_metrics/
        self.aggregate_models() # Uses FLServer's FedAvg aggregation
        # Distribute the updated server global model to clients' global state
        self.distribute_global_model() # Uses FLServer's distribution

        # --- 2. Personal Model Update Phase ---
        print("Ditto Round: Starting Personal Model Update Phase")
        agg_personal_train_loss = 0.0
        agg_personal_val_loss = 0.0
        agg_personal_val_score = {}

        for client_id, client in self.clients.items():
             # Train client's personal model state (personal=True), regularizing towards the *new* global model
             client_personal_train_loss = client.train(personal=True)
             # Validate client's personal model state
             client_personal_val_loss, client_personal_val_score = client.validate(personal=True)

             # Aggregate personal model metrics (weighted)
             weight = client.data.weight
             agg_personal_train_loss += client_personal_train_loss * weight
             agg_personal_val_loss += client_personal_val_loss * weight
             agg_personal_val_score = self._aggregate_scores(agg_personal_val_score, client_personal_val_score, weight)

        # --- Track Server-Side Metrics (based on personal model performance) ---
        self.serverstate.train_losses.append(agg_personal_train_loss)
        self.serverstate.val_losses.append(agg_personal_val_loss)
        self.serverstate.val_scores.append(agg_personal_val_score)

        # --- Update Best Global Model Snapshot (based on global model's validation performance) ---
        if agg_global_val_loss < self.serverstate.best_loss:
            self.serverstate.best_loss = agg_global_val_loss
            self.serverstate.best_model = copy.deepcopy(self.serverstate.model).to(self.device)
            print(f"Server (Ditto): New best global model saved with validation loss {agg_global_val_loss:.4f}")

        # Return aggregated metrics from the personal model validation phase
        return agg_personal_train_loss, agg_personal_val_loss, agg_personal_val_score


class LocalAdaptationServer(FLServer):
    """
    Server implementation for algorithms involving local adaptation after federation.

    Typically behaves like FedAvg during main rounds. In the final round, it can
    trigger a different training behavior on the clients (e.g., fine-tuning).
    """
    def _create_client(self, clientdata: SiteData, modelstate: ModelState, personal_model: bool = False) -> LocalAdaptationClient:
        """
        Overrides the factory method to create `LocalAdaptationClient` instances.

        Args:
            clientdata (SiteData): Client's data.
            modelstate (ModelState): Initial global model state.
            personal_model (bool): Typically False for this approach.

        Returns:
            LocalAdaptationClient: An instance of the LocalAdaptation client.
        """
        print(f"Creating LocalAdaptationClient for site {clientdata.site_id}")
        return LocalAdaptationClient(
            config=self.config,
            data=clientdata,
            modelstate=modelstate.copy(),
            metrics_calculator=MetricsCalculator(self.config.dataset_name),
            personal_model=personal_model
        )

    # Define train_round with optional final_round parameter
    def train_round(self, round_num: Optional[int] = None) -> Tuple[float, float, Dict[str, float]]:
        """
        Executes one communication round, with special handling for the final round.

        If `final_round` is True, it instructs clients to perform their final
        adaptation step (e.g., one epoch of fine-tuning) after the standard
        aggregation and distribution.

        Args:
            final_round (bool): Flag indicating if this is the final round. Defaults to False.

        Returns:
            Tuple[float, float, Dict[str, float]]: Aggregated train loss, validation loss,
                                                   and validation scores for the round. If
                                                   it's the final round, these metrics reflect
                                                   performance *after* the adaptation step.
        """
        # Standard training, aggregation, distribution happens first via super().train_round()
        # Adaptation is *additional* if final_round.

        # Perform the standard FedAvg-like round first
        train_loss, val_loss, val_score = super().train_round() # Aggregation/distribution happens here
        final_round = (round_num == self.config.rounds - 1)
        if final_round:
            print("LocalAdaptationServer: Starting final adaptation phase on clients...")
            agg_adapt_train_loss = 0.0
            agg_adapt_val_loss = 0.0
            agg_adapt_val_score = {}

            for client_id, client in self.clients.items():
                 # Instruct client to perform final adaptation training step
                 # The client's train method handles the final_round logic (e.g., 1 epoch)
                 client_adapt_train_loss = client.train(self.personal, final_round=True)
                 # Re-validate after adaptation
                 client_adapt_val_loss, client_adapt_val_score = client.validate(self.personal)

                 # Aggregate metrics *after* adaptation
                 weight = client.data.weight
                 agg_adapt_train_loss += client_adapt_train_loss * weight
                 agg_adapt_val_loss += client_adapt_val_loss * weight
                 agg_adapt_val_score = self._aggregate_scores(agg_adapt_val_score, client_adapt_val_score, weight)

            # Update server metrics to reflect the post-adaptation state for the final round
            # Overwrite the last entry in the lists
            if self.serverstate.train_losses: self.serverstate.train_losses[-1] = agg_adapt_train_loss
            if self.serverstate.val_losses: self.serverstate.val_losses[-1] = agg_adapt_val_loss
            if self.serverstate.val_scores: self.serverstate.val_scores[-1] = agg_adapt_val_score

            print(f"LocalAdaptationServer: Final adaptation complete. Post-Adapt Val Loss: {agg_adapt_val_loss:.4f}, Post-Adapt Val Acc: {agg_adapt_val_score.get('accuracy', -1):.4f}")
            # Return the post-adaptation metrics for the final round
            return agg_adapt_train_loss, agg_adapt_val_loss, agg_adapt_val_score
        else:
            # Return metrics from the standard round if not final
            return train_loss, val_loss, val_score


class LayerServer(FLServer):
    """
    Server base class for layer-wise federated learning algorithms (LayerPFL, BABU).

    Overrides `aggregate_models` and `distribute_global_model` from `FLServer`
    to perform operations only on a specified subset of model layers, defined
    by `layers_to_include` in the configuration.
    """
    def aggregate_models(self):
        """
        Aggregates parameters only for the layers specified in `layers_to_include`.

        Performs a weighted average (FedAvg) but restricted to the designated layers.
        Parameters of non-federated layers in the server model remain unchanged.
        """
        if 'layers_to_include' not in self.config.algorithm_params:
            raise ValueError("LayerServer requires 'layers_to_include' in algorithm_params.")
        layers_to_include = self.config.algorithm_params['layers_to_include']
        # print(f"LayerServer: Aggregating layers: {layers_to_include}")

        # Use a buffer to store aggregated parameters for federated layers
        aggregated_layer_params = OrderedDict()

        # Initialize buffer with zeros ONLY for federated layers
        for name, param in self.serverstate.model.named_parameters():
            is_federated = any(layer_prefix in name for layer_prefix in layers_to_include)
            if is_federated:
                aggregated_layer_params[name] = torch.zeros_like(param.data, device=self.device)

        # Weighted summation from clients for federated layers
        total_weight = 0.0
        for client_id, client in self.clients.items():
            # Layer-wise methods typically operate on the main model state
            client_state = client.get_client_state(personal=False)
            client_state_dict = client_state.model.state_dict()
            client_weight = client.data.weight
            total_weight += client_weight

            for name, client_param_data in client_state_dict.items():
                # Check if this parameter belongs to a federated layer AND is in our buffer
                if name in aggregated_layer_params:
                    aggregated_layer_params[name].add_(client_param_data.to(self.device) * client_weight)

        # Check weights sum
        if not np.isclose(total_weight, 1.0, atol=1e-6):
             print(f"Warning: Client weights sum to {total_weight}, not 1.0 during layer aggregation.")

        # Update only the federated layers in the server's global model state
        current_global_state = self.serverstate.model.state_dict()
        for name in current_global_state.keys():
            if name in aggregated_layer_params:
                 current_global_state[name].copy_(aggregated_layer_params[name])
        # Load the updated state back into the model
        self.serverstate.model.load_state_dict(current_global_state)
        # print("LayerServer: Aggregation complete.")


    def distribute_global_model(self):
        """
        Distributes only the parameters of the federated layers to clients.

        Clients use their specialized `set_model_state` (from `LayerClient`)
        to update only the corresponding layers, keeping their local layers unchanged.
        """
        if 'layers_to_include' not in self.config.algorithm_params:
            raise ValueError("LayerServer requires 'layers_to_include' in algorithm_params.")
        layers_to_include = self.config.algorithm_params['layers_to_include']
        # print(f"LayerServer: Distributing layers: {layers_to_include}")

        # Get the current state of the server's global model
        global_state_dict = self.serverstate.model.state_dict()

        # Prepare the state dictionary containing only the federated layers
        distribution_state = OrderedDict()
        for name, param in global_state_dict.items():
             is_federated = any(layer_prefix in name for layer_prefix in layers_to_include)
             if is_federated:
                 distribution_state[name] = param.clone().detach()

        if not distribution_state:
             print("Warning: No layers marked for distribution in LayerServer.")
             return

        # Send the partial state dictionary to each client
        for client_id, client in self.clients.items():
             # The client's set_model_state (from LayerClient) handles partial updates
             client.set_model_state(distribution_state, personal=False)
        # print("LayerServer: Distribution complete.")


class LayerPFLServer(LayerServer):
    """
    Server implementation for the LayerPFL algorithm (fixed or random subset).

    Uses the layer-selective aggregation and distribution logic from `LayerServer`.
    The specific layers are determined by `layers_to_include` in the config.
    It creates `LayerPFLClient` instances.
    """
    def _create_client(self, clientdata: SiteData, modelstate: ModelState, personal_model: bool = False) -> LayerPFLClient:
        """
        Overrides the factory method to create `LayerPFLClient` instances.

        Args:
            clientdata (SiteData): Client's data.
            modelstate (ModelState): Initial global model state.
            personal_model (bool): Should be False for LayerPFL.

        Returns:
            LayerPFLClient: An instance of the LayerPFL client.
        """
        print(f"Creating LayerPFLClient for site {clientdata.site_id}")
        return LayerPFLClient(
            config=self.config,
            data=clientdata,
            modelstate=modelstate.copy(),
            metrics_calculator=MetricsCalculator(self.config.dataset_name),
            personal_model=personal_model # LayerPFL operates on main model state
        )


class BABUServer(LayerServer):
    """
    Server implementation for the BABU algorithm.

    Uses layer-selective logic from `LayerServer` for body aggregation/distribution.
    Adds logic to trigger head-only training on clients in the final round.
    Creates `BABUClient` instances.
    """
    def _create_client(self, clientdata: SiteData, modelstate: ModelState, personal_model: bool = False) -> BABUClient:
        """
        Overrides the factory method to create `BABUClient` instances.

        Args:
            clientdata (SiteData): Client's data.
            modelstate (ModelState): Initial global model state.
            personal_model (bool): Should be False for BABU.

        Returns:
            BABUClient: An instance of the BABU client.
        """
        print(f"Creating BABUClient for site {clientdata.site_id}")
        return BABUClient(
            config=self.config,
            data=clientdata,
            modelstate=modelstate.copy(),
            metrics_calculator=MetricsCalculator(self.config.dataset_name),
            personal_model=personal_model # BABU operates on main model state
        )

    # Define train_round with optional final_round parameter
    def train_round(self, round_num: Optional[int] = None) -> Tuple[float, float, Dict[str, float]]:
        """
        Executes one communication round for BABU.

        Performs standard body aggregation/distribution in all rounds.
        If `final_round` is True, it additionally instructs clients to train
        only their head layers after receiving the final aggregated body.

        Args:
            final_round (bool): Flag indicating if this is the final round. Defaults to False.

        Returns:
            Tuple[float, float, Dict[str, float]]: Aggregated train loss, validation loss,
                                                   and validation scores. If final round,
                                                   metrics reflect performance after head tuning.
        """
        # Perform the standard layer-wise round first (aggregates/distributes body)
        train_loss, val_loss, val_score = super().train_round() # Uses LayerServer logic
        final_round = (round_num == self.config.rounds - 1)
        if final_round:
            print("BABUServer: Starting final head tuning phase on clients...")
            agg_head_train_loss = 0.0
            agg_head_val_loss = 0.0
            agg_head_val_score = {}

            for client_id, client in self.clients.items():
                 # Client must be a BABUClient to have train_head() method
                 if isinstance(client, BABUClient):
                     # Instruct client to train its head
                     client_head_train_loss = client.train_head()
                     # Re-validate after head tuning
                     client_head_val_loss, client_head_val_score = client.validate(personal=False)

                     # Aggregate metrics *after* head tuning
                     weight = client.data.weight
                     agg_head_train_loss += client_head_train_loss * weight
                     agg_head_val_loss += client_head_val_loss * weight
                     agg_head_val_score = self._aggregate_scores(agg_head_val_score, client_head_val_score, weight)
                 else:
                     print(f"Warning: Client {client_id} is not a BABUClient. Skipping head tuning.")


            # Update server metrics to reflect the post-head-tuning state for the final round
            if self.serverstate.train_losses: self.serverstate.train_losses[-1] = agg_head_train_loss
            if self.serverstate.val_losses: self.serverstate.val_losses[-1] = agg_head_val_loss
            if self.serverstate.val_scores: self.serverstate.val_scores[-1] = agg_head_val_score

            print(f"BABUServer: Final head tuning complete. Post-Tuning Val Loss: {agg_head_val_loss:.4f}, Post-Tuning Val Acc: {agg_head_val_score.get('accuracy', -1):.4f}")
            # Return the post-head-tuning metrics
            return agg_head_train_loss, agg_head_val_loss, agg_head_val_score
        else:
            # Return metrics from the standard body round
            return train_loss, val_loss, val_score


class FedLPServer(FLServer):
    """
    Server implementation for the FedLP algorithm.

    Overrides `aggregate_models` to perform probabilistic layer-wise aggregation.
    Clients (`FedLPClient`) determine which layers participate locally based on a
    probability (`layer_preserving_rate`) and send only those layers. The server
    aggregates participating layers, normalizing by the total weight of participants
    for each layer. Layers with no participants retain their previous state.
    """
    def _create_client(self, clientdata: SiteData, modelstate: ModelState, personal_model: bool = False) -> FedLPClient:
        """
        Overrides the factory method to create `FedLPClient` instances.

        Args:
            clientdata (SiteData): Client's data.
            modelstate (ModelState): Initial global model state.
            personal_model (bool): Should be False for FedLP.

        Returns:
            FedLPClient: An instance of the FedLP client.
        """
        print(f"Creating FedLPClient for site {clientdata.site_id}")
        return FedLPClient(
            config=self.config,
            data=clientdata,
            modelstate=modelstate.copy(),
            metrics_calculator=MetricsCalculator(self.config.dataset_name),
            personal_model=personal_model # FedLP operates on main model state
        )

    def aggregate_models(self):
        """
        Performs FedLP aggregation based on probabilistic layer participation.

        Collects pruned model states (containing only participating layers) from clients.
        Aggregates each layer by averaging parameters only from clients whose indicator
        for that layer was 1. Normalizes by the sum of weights of participating clients
        for that specific layer. If a layer had no participants, its parameters remain unchanged.
        """
        print("Performing FedLP aggregation...")
        # Store participating client IDs and their weights for each layer
        layer_participants = {} # {layer_name: [(client_id, weight), ...]}
        # Store the actual parameters received from participating clients for each layer
        layer_params_received = {} # {layer_name: {client_id: {param_name: tensor, ...}}}

        # --- Collect pruned states and participation info from clients ---
        for client_id, client in self.clients.items():
             # Client needs to be FedLPClient to have get_pruned_model_state
             if isinstance(client, FedLPClient):
                 # Get state dict with only participating layers and the indicators used
                 pruned_state, indicators = client.get_pruned_model_state()

                 client_weight = client.data.weight

                 # Store received parameters and track participation
                 for layer_name, participated in indicators.items():
                     if participated == 1:
                         # Track participant for this layer
                         if layer_name not in layer_participants:
                             layer_participants[layer_name] = []
                         layer_participants[layer_name].append((client_id, client_weight))

                         # Store parameters belonging to this participating layer
                         if layer_name not in layer_params_received:
                             layer_params_received[layer_name] = {}
                         if client_id not in layer_params_received[layer_name]:
                              layer_params_received[layer_name][client_id] = {}

                         # Find parameters in pruned_state belonging to this layer
                         for param_name, param_tensor in pruned_state.items():
                              if param_name.startswith(layer_name):
                                  layer_params_received[layer_name][client_id][param_name] = param_tensor
             else:
                 print(f"Warning: Client {client_id} is not a FedLPClient. Skipping its contribution.")


        # --- Aggregate layer by layer ---
        new_global_state_dict = self.serverstate.model.state_dict() # Start with current global state

        for layer_name, participants in layer_participants.items():
             if not participants: continue # Skip if somehow no participants were recorded (shouldn't happen with above logic)

             # Calculate total weight of participants *for this layer*
             total_layer_weight = sum(weight for _, weight in participants)

             if total_layer_weight > 0:
                 # Find all parameter names belonging to this layer in the global model
                 params_in_layer = [name for name in new_global_state_dict if name.startswith(layer_name)]

                 for param_name in params_in_layer:
                     # Zero out the global parameter before accumulating
                     new_global_state_dict[param_name].zero_()
                     # Aggregate from participating clients
                     for client_id, client_weight in participants:
                          # Check if client sent this parameter (should have if layer participated)
                          if client_id in layer_params_received.get(layer_name, {}) and \
                             param_name in layer_params_received[layer_name][client_id]:

                              param_data = layer_params_received[layer_name][client_id][param_name]
                              # Normalize weight and add contribution
                              normalized_weight = client_weight / total_layer_weight
                              new_global_state_dict[param_name].add_(param_data.to(self.device) * normalized_weight)
                          # else: # Should not happen if logic is correct
                          #      print(f"Warning: Missing param '{param_name}' from participating client '{client_id}' for layer '{layer_name}'.")

             # else: Layer had participants listed but total weight was 0? Log warning. Keep param as is.
             elif participants: # Log only if participants list wasn't empty
                  print(f"Warning: Layer '{layer_name}' had participants but total weight was 0. Skipping aggregation for this layer.")

        # Load the potentially modified state dict back into the server model
        self.serverstate.model.load_state_dict(new_global_state_dict)
        print("FedLP aggregation complete.")


class FedLAMAServer(FLServer):
    """
    Server implementation for the FedLAMA algorithm.

    FedLAMA adaptively adjusts the aggregation frequency for different layers
    based on estimated parameter divergence. Layers with higher divergence are
    aggregated more frequently.
    """
    def __init__(self, config: TrainerConfig, globalmodelstate: ModelState):
        """
        Initializes the FedLAMAServer.

        Sets up FedLAMA-specific parameters (tau_prime, phi) and round counter.

        Args:
            config (TrainerConfig): Configuration object.
            globalmodelstate (ModelState): Initial global model state.
        """
        super().__init__(config, globalmodelstate)
        # Base aggregation interval (τ')
        self.tau_prime = config.algorithm_params.get('tau_prime', 1)
        # Interval increase factor (φ)
        self.phi = config.algorithm_params.get('phi', 1)
        # Current communication round number
        self.round = 0
        # Dictionary to store aggregation interval for each parameter name {param_name: interval}
        self.aggregation_intervals: Optional[Dict[str, int]] = None
        print(f"FedLAMAServer initialized with tau_prime={self.tau_prime}, phi={self.phi}")


    def _create_client(self, clientdata: SiteData, modelstate: ModelState, personal_model: bool = False) -> FedLAMAClient:
        """
        Overrides the factory method to create `FedLAMAClient` instances.

        Args:
            clientdata (SiteData): Client's data.
            modelstate (ModelState): Initial global model state.
            personal_model (bool): Should be False for FedLAMA.

        Returns:
            FedLAMAClient: An instance of the FedLAMA client.
        """
        print(f"Creating FedLAMAClient for site {clientdata.site_id}")
        return FedLAMAClient(
            config=self.config,
            data=clientdata,
            modelstate=modelstate.copy(),
            metrics_calculator=MetricsCalculator(self.config.dataset_name),
            personal_model=personal_model # FedLAMA operates on main model state
        )

    def calculate_layer_discrepancy(self) -> Tuple[Dict[str, float], Dict[str, int]]:
        """
        Calculates the average L2 norm difference between each global model parameter
        and the corresponding client parameters across all clients.

        Returns:
            Tuple[Dict[str, float], Dict[str, int]]: A tuple containing:
                - discrepancies: Dict mapping parameter name to its average discrepancy.
                - layer_dims: Dict mapping parameter name to its number of elements.
        """
        discrepancies = {name: 0.0 for name, _ in self.serverstate.model.named_parameters()}
        layer_dims = {name: param.numel() for name, param in self.serverstate.model.named_parameters()}
        num_clients = len(self.clients)

        if num_clients == 0:
             return discrepancies, layer_dims # Return zeros if no clients

        global_state_dict = self.serverstate.model.state_dict()

        for client_id, client in self.clients.items():
            client_state = client.get_client_state(personal=False)
            client_state_dict = client_state.model.state_dict()
            for name, global_param_data in global_state_dict.items():
                 if name in client_state_dict:
                     global_param_device = global_param_data.to(self.device)
                     client_param_device = client_state_dict[name].to(self.device)
                     # Calculate L2 norm of the difference
                     diff_norm = torch.norm(global_param_device - client_param_device, p=2).item()
                     discrepancies[name] += diff_norm
                 else:
                      print(f"Warning: Parameter '{name}' missing in client {client_id} during discrepancy calculation.")

        # Average the discrepancy across clients
        avg_discrepancies = {name: diff / num_clients for name, diff in discrepancies.items()}

        return avg_discrepancies, layer_dims

    def find_aggregation_cutoff(self, sorted_discrepancies: List[Tuple[str, float]], layer_dims: Dict[str, int]) -> int:
        """
        Finds the optimal cutoff point `l` based on the FedLAMA paper's heuristic.

        Sorts layers by discrepancy and finds the index `l` that best balances
        cumulative discrepancy (delta_l) and cumulative parameter size (lambda_l),
        aiming for delta_l ≈ 1 - lambda_l.

        Args:
            sorted_discrepancies (List[Tuple[str, float]]): List of (param_name, discrepancy)
                                                             tuples, sorted by discrepancy (ascending).
            layer_dims (Dict[str, int]): Dictionary mapping parameter name to its size.

        Returns:
            int: The index `l` (1-based) representing the cutoff point. Layers with
                 index < `l` have lower discrepancy.
        """
        total_discrepancy = sum(d * layer_dims.get(name, 0) for name, d in sorted_discrepancies)
        total_size = sum(layer_dims.values())

        if total_discrepancy == 0 or total_size == 0:
             print("Warning: Total discrepancy or size is zero in find_aggregation_cutoff. Returning cutoff 0.")
             return 0 # Avoid division by zero, return 0 or handle appropriately

        best_l = 0 # Default cutoff (all layers aggregate frequently)
        min_abs_diff = float('inf')

        cumulative_disc = 0.0
        cumulative_size = 0.0

        # Iterate through sorted parameters to find the best split point
        for i, (param_name, disc) in enumerate(sorted_discrepancies):
            param_size = layer_dims.get(param_name, 0)
            cumulative_disc += disc * param_size
            cumulative_size += param_size

            # Normalized cumulative discrepancy (delta_l in paper)
            delta_l = cumulative_disc / total_discrepancy
            # Normalized cumulative size (lambda_l in paper)
            lambda_l = cumulative_size / total_size

            # Calculate the absolute difference for the heuristic |delta_l - (1 - lambda_l)|
            abs_diff = abs(delta_l - (1.0 - lambda_l))

            # Update best cutoff if this point yields a smaller difference
            if abs_diff < min_abs_diff:
                min_abs_diff = abs_diff
                best_l = i + 1 # Cutoff is *after* this index i

        # print(f"FedLAMA: Best cutoff l = {best_l} found.")
        return best_l

    def adjust_aggregation_intervals(self) -> Dict[str, int]:
        """
        Adjusts the aggregation interval for each parameter based on calculated discrepancies.

        Layers with lower discrepancy (before the cutoff `l`) get a longer interval (phi * tau_prime),
        while layers with higher discrepancy (at or after cutoff `l`) get the base interval (tau_prime).

        Returns:
            Dict[str, int]: The updated dictionary mapping parameter names to their
                            new aggregation intervals.
        """
        print("FedLAMA: Adjusting aggregation intervals...")
        # 1. Calculate discrepancies
        discrepancies, layer_dims = self.calculate_layer_discrepancy()

        # 2. Sort parameters by discrepancy (ascending)
        # Convert dict items to list of tuples for sorting
        sorted_params = sorted(discrepancies.items(), key=lambda item: item[1])

        # 3. Find the optimal cutoff index `l`
        cutoff_l = self.find_aggregation_cutoff(sorted_params, layer_dims)

        # 4. Set new intervals based on the cutoff
        new_intervals = {}
        for i, (param_name, _) in enumerate(sorted_params):
            # If index `i` is before the cutoff `l` (i.e., 0 <= i < cutoff_l), use longer interval
            if i < cutoff_l:
                new_intervals[param_name] = self.phi * self.tau_prime
            else: # Otherwise, use base interval
                new_intervals[param_name] = self.tau_prime

        # print(f"FedLAMA: New intervals calculated: {new_intervals}")
        return new_intervals

    def aggregate_models(self):
        """
        Performs FedLAMA aggregation with adaptive intervals.

        Checks if aggregation intervals need updating based on the current round.
        Then, performs standard FedAvg aggregation but *only* for parameters
        whose aggregation interval condition is met (round % interval == 0).
        Increments the round counter.
        """
        # --- Initialize or Update Aggregation Intervals ---
        # Initialize intervals on the first call (round 0)
        if self.aggregation_intervals is None:
            print("FedLAMA: Initializing aggregation intervals.")
            self.aggregation_intervals = {
                name: self.tau_prime # Start with base interval for all
                for name, _ in self.serverstate.model.named_parameters()
            }

        # Update intervals periodically (e.g., at round 2 and then every phi*tau_prime rounds)
        # Adjust condition based on paper or desired frequency. Using original code's condition.
        # Note: `self.round` is 0-based index. Condition checks for rounds 2, T, 2T, ... where T = phi*tau_prime
        if self.round == 2 or (self.round > 0 and self.round % (self.phi * self.tau_prime) == 0):
            self.aggregation_intervals = self.adjust_aggregation_intervals()

        # --- Perform Selective Aggregation ---
        print(f"FedLAMA (Round {self.round+1}): Performing selective aggregation.")
        # Get current global state to potentially update selectively
        current_global_state = self.serverstate.model.state_dict()
        # Buffer for parameters that *will* be aggregated this round
        aggregation_buffer = OrderedDict()

        # Identify parameters due for aggregation
        params_to_aggregate = set()
        for name, interval in self.aggregation_intervals.items():
             # Aggregate always in early rounds? (round < 2) or based on interval
             if self.round < 2 or (self.round % interval == 0):
                 params_to_aggregate.add(name)
                 aggregation_buffer[name] = torch.zeros_like(current_global_state[name], device=self.device)

        if not params_to_aggregate:
             print("FedLAMA: No parameters due for aggregation this round.")
             self.round += 1 # Still increment round counter
             return

        # print(f"FedLAMA: Aggregating parameters: {list(params_to_aggregate)}")

        # Weighted summation from clients ONLY for parameters due for aggregation
        total_weight = 0.0
        for client_id, client in self.clients.items():
            client_state = client.get_client_state(personal=False)
            client_state_dict = client_state.model.state_dict()
            client_weight = client.data.weight
            total_weight += client_weight

            for name in params_to_aggregate:
                if name in client_state_dict:
                     aggregation_buffer[name].add_(client_state_dict[name].to(self.device) * client_weight)
                # else: # Parameter missing from client? Should not happen.

        # Check weights sum
        if not np.isclose(total_weight, 1.0, atol=1e-6):
             print(f"Warning: Client weights sum to {total_weight}, not 1.0 during FedLAMA aggregation.")

        # Update the global model state only for the aggregated parameters
        for name in params_to_aggregate:
             current_global_state[name].copy_(aggregation_buffer[name])

        # Load the updated state back into the model
        self.serverstate.model.load_state_dict(current_global_state)

        # Increment the round counter AFTER aggregation
        self.round += 1
        print("FedLAMA: Aggregation complete.")


class pFedLAServer(FLServer):
    """
    Server implementation for the pFedLA algorithm.

    Uses a HyperNetwork to generate personalized layer-wise aggregation weights
    for each client. The server manages the HyperNetwork, client embeddings, and
    server-side copies of client model parameters. It orchestrates a round by:
    1. Generating a personalized model for each client using the HyperNetwork.
    2. Distributing the personalized model to the client.
    3. Receiving parameter *updates* (deltas) from clients after local training.
    4. Updating its internal copy of the client's model parameters using the delta.
    5. Updating the HyperNetwork parameters using the client's delta via backpropagation.
    """
    def __init__(self, config: TrainerConfig, globalmodelstate: ModelState):
        """
        Initializes the pFedLAServer.

        Sets up HyperNetwork parameters and placeholders for HN, client embeddings, etc.

        Args:
            config (TrainerConfig): Configuration object.
            globalmodelstate (ModelState): Initial global model state (used as backbone reference).
        """
        super().__init__(config, globalmodelstate)
        # pFedLA specific parameters from config
        algorithm_params = config.algorithm_params
        self.embedding_dim = algorithm_params.get('embedding_dim', 32)
        self.hidden_dim = algorithm_params.get('hidden_dim', 64)
        self.hn_lr = algorithm_params.get('hn_lr', 0.01)

        # Placeholders - initialized once all clients are added
        self.hypernetwork: Optional[HyperNetwork] = None
        # Server's copies of each client's model parameters. List of lists of tensors.
        self.client_models: Optional[List[List[torch.Tensor]]] = None
        self.layer_names: Optional[List[str]] = None       # Names of all parameters
        self.trainable_names: Optional[List[str]] = None # Names of trainable parameters
        print(f"pFedLAServer initialized. HN params: emb={self.embedding_dim}, hidden={self.hidden_dim}, lr={self.hn_lr}")


    def _initialize_hypernetwork(self):
        """
        Initializes the HyperNetwork, client embeddings, and server-side client models.

        Called once after all clients have been added to the server.
        """
        if self.hypernetwork is not None:
             print("Warning: HyperNetwork already initialized.")
             return

        print("pFedLAServer: Initializing HyperNetwork...")
        num_clients = len(self.clients)
        if num_clients == 0:
             raise RuntimeError("Cannot initialize HyperNetwork with zero clients.")

        # Initialize the HyperNetwork module
        self.hypernetwork = HyperNetwork(
            embedding_dim=self.embedding_dim,
            client_num=num_clients,
            hidden_dim=self.hidden_dim,
            backbone=self.serverstate.model, # Pass backbone to get layer structure
        ).to(self.device)

        # Initialize server's copies of client models (start identical to global model)
        # List of parameter lists: [[client0_p1, client0_p2,...], [client1_p1, client1_p2,...]]
        self.client_models = [
            [param.clone().detach().to(self.device) for param in self.serverstate.model.parameters()]
            for _ in range(num_clients)
        ]

        # Store parameter names for reference
        self.layer_names = [name for name, _ in self.serverstate.model.named_parameters()]
        self.trainable_names = [
            name for name, param in self.serverstate.model.named_parameters()
            if param.requires_grad
        ]
        print(f"HyperNetwork initialized for {num_clients} clients.")


    def _create_client(self, clientdata: SiteData, modelstate: ModelState, personal_model: bool = False) -> pFedLAClient:
        """
        Overrides the factory method to create `pFedLAClient` instances.

        Args:
            clientdata (SiteData): Client's data.
            modelstate (ModelState): Initial global model state (not directly used by client).
            personal_model (bool): Should be False for pFedLA.

        Returns:
            pFedLAClient: An instance of the pFedLA client.
        """
        print(f"Creating pFedLAClient for site {clientdata.site_id}")
        # pFedLA client doesn't manage a separate personal state; server sends the full state.
        if personal_model:
            print("Warning: pFedLAServer forcing personal_model=False for client creation.")
        return pFedLAClient(
            config=self.config,
            data=clientdata,
            modelstate=modelstate.copy(), # Client uses this state object, but server overwrites model params
            metrics_calculator=MetricsCalculator(self.config.dataset_name),
            personal_model=False
        )

    def add_client(self, clientdata: SiteData):
        """
        Overrides `add_client` to initialize the HyperNetwork after all expected
        clients have been added.

        Args:
            clientdata (SiteData): Client's data.
        """
        super().add_client(clientdata) # Adds the client using _create_client
        # Check if all clients are now present, then initialize HN
        if len(self.clients) == self.config.num_clients:
            if self.hypernetwork is None:
                 self._initialize_hypernetwork()
            else:
                 print("Warning: All clients added, but HyperNetwork was already initialized.")


    def generate_client_model(self, client_id: str) -> OrderedDict:
        """
        Generates a personalized model state dictionary for a specific client.

        Uses the HyperNetwork to get layer-wise aggregation weights (alpha) and
        combines the server's stored copies of all client models based on these weights.

        Args:
            client_id (str): The ID of the client for whom to generate the model.

        Returns:
            OrderedDict: The state dictionary for the personalized model.

        Raises:
            RuntimeError: If the HyperNetwork is not initialized.
        """
        if self.hypernetwork is None or self.client_models is None or self.layer_names is None:
            raise RuntimeError("HyperNetwork or client models not initialized. Cannot generate client model.")

        client_idx = int(client_id.split('_')[-1]) - 1 # Assumes client_id format 'client_N'
        # 1. Get personalized aggregation weights (alpha) from HyperNetwork
        # alpha = {layer_base_name: tensor_of_weights_for_all_clients}
        alpha = self.hypernetwork(client_idx) # Pass index to HN

        # 2. Prepare server-side client parameters for aggregation
        # layer_params = {param_name: stacked_tensor_for_all_clients}
        layer_params = {}
        # zip(*self.client_models) transposes the list of lists:
        # [[c0p1, c0p2,...], [c1p1, c1p2,...]] -> [(c0p1, c1p1), (c0p2, c1p2), ...]
        for name, params_across_clients in zip(self.layer_names, zip(*self.client_models)):
            # Ensure all tensors are on the correct device before stacking
            params_on_device = [p.to(self.device) for p in params_across_clients]
            try:
                 # Stack tensors along a new dimension (dim=0)
                 layer_params[name] = torch.stack(params_on_device, dim=0)
            except RuntimeError as e:
                 print(f"Error stacking parameters for '{name}': {e}. Shapes: {[p.shape for p in params_on_device]}")
                 raise e


        # 3. Perform weighted aggregation for each parameter
        personalized_params_dict = OrderedDict()
        num_clients = len(self.clients)

        for name in self.layer_names:
            if name in self.trainable_names:
                # Get the base layer name (e.g., 'layer1' from 'layer1.weight')
                base_name = name.split('.')[0]
                # Get the aggregation weights for this layer generated by HN for this client
                if base_name in alpha:
                     weights = alpha[base_name].to(self.device) # Shape: [num_clients]
                else:
                     # Should not happen if HN includes all trainable layers
                     print(f"Warning: Aggregation weights (alpha) missing for trainable layer '{base_name}'. Using uniform weights.")
                     weights = torch.ones(num_clients, device=self.device) / num_clients
            else:
                weights = torch.zeros(num_clients, device=self.device)
                weights[client_idx] = 1.0

            # Normalize weights (softmax in HN should ensure this, but double-check)
            if not torch.isclose(weights.sum(), torch.tensor(1.0, device=self.device)):
                 # print(f"Warning: Weights for layer '{base_name}'/'{name}' do not sum to 1 ({weights.sum()}). Renormalizing.")
                 weights = weights / weights.sum() if weights.sum() != 0 else torch.ones_like(weights) / num_clients


            # Aggregate: Sum(weights[i] * client_model[i][param])
            # Need to reshape weights to multiply with stacked parameter tensor
            # stacked_params = layer_params[name] # Shape: [num_clients, *param_shape]
            # weights_expanded shape depends on param shape to allow broadcasting

            param_tensor = layer_params[name] # Shape [C, ...]
            param_shape = param_tensor.shape[1:] # Shape of the parameter itself
            # Reshape weights to [C, 1, 1, ...] to match param_tensor dimensions
            view_shape = (num_clients,) + (1,) * len(param_shape)
            weights_expanded = weights.view(*view_shape)#.expand_as(param_tensor) # Expand might be redundant with broadcasting

            # Weighted sum along the client dimension (dim=0)
            personalized_param = torch.sum(weights_expanded * param_tensor, dim=0)
            personalized_params_dict[name] = personalized_param

        return personalized_params_dict

    @torch.enable_grad() # Ensure gradients are enabled for HN update
    def update_hypernetwork(self, client_id: str, delta: OrderedDict):
        """
        Updates the HyperNetwork parameters using the parameter updates (delta)
        received from a client.

        Performs backpropagation through the HyperNetwork's generation process
        using the client's delta as the gradient signal.

        Args:
            client_id (str): The ID of the client providing the update.
            delta (OrderedDict): The parameter updates (current_param - initial_param)
                                 from the client.
        """
        if self.hypernetwork is None or self.client_models is None or self.trainable_names is None:
             print("Warning: Cannot update HyperNetwork, not initialized.")
             return

        client_idx = int(client_id.split('_')[-1]) - 1

        # Ensure delta tensors are on the correct device and require grad? No, delta is grad output target.
        delta_on_device = {name: param.to(self.device) for name, param in delta.items()}

        # Prepare lists for torch.autograd.grad
        # Outputs: Server's stored parameters for the specific client *before* update
        #          Filter for only trainable parameters.
        client_params_before_update = [
            p for p, name in zip(self.client_models[client_idx], self.layer_names) if name in self.trainable_names
        ]
        # Inputs: Parameters of the HyperNetwork
        hn_params = self.hypernetwork.get_params() # Gets list of HN parameters (embeddings, MLP weights/biases)
        # Grad Outputs: The deltas received from the client, corresponding to trainable params
        grad_outputs = [
            delta_on_device[name] for name in self.layer_names if name in self.trainable_names
        ]

        if not client_params_before_update or not grad_outputs or not hn_params:
             print(f"Warning: Skipping HN update for {client_id}. Missing parameters or delta.")
             return

        # Calculate gradients of HN parameters w.r.t. the client's delta
        # This requires requires_grad=True on the outputs (client_params) and HN params.
        # We need to regenerate the client model with grad enabled w.r.t HN params.

        # Ensure HN params require grad
        for p in hn_params:
            p.requires_grad_(True)

        for p in client_params_before_update: p.requires_grad_(True)

        try:
            hn_grads = torch.autograd.grad(
                outputs=client_params_before_update, # Parameters that delta corresponds to
                inputs=hn_params,                    # Parameters we want gradients for
                grad_outputs=grad_outputs,           # The 'gradient' signal (delta)
                allow_unused=True                    # Some HN params might not affect all client params
            )
        except Exception as e:
            print(f"Error during torch.autograd.grad for HN update: {e}")
            # Reset requires_grad and return
            for p in hn_params: p.requires_grad_(False)
            for p in client_params_before_update: p.requires_grad_(False)
            return


        # --- Apply Gradients to HyperNetwork Parameters ---
        with torch.no_grad(): # Perform update step without tracking gradients
            for param, grad in zip(hn_params, hn_grads):
                if grad is not None:
                    param.sub_(grad * self.hn_lr) # param = param - hn_lr * grad
                # else: # Parameter was unused in computation for this client/delta
                #     pass

        # Reset requires_grad states
        for p in hn_params: p.requires_grad_(False)
        for p in client_params_before_update: p.requires_grad_(False)

        # print(f"HyperNetwork updated using delta from {client_id}.")


    def update_client_model(self, client_id: str, delta: OrderedDict):
        """
        Updates the server's internal copy of a client's model parameters.

        Adds the received delta to the stored parameters for the specified client.

        Args:
            client_id (str): The ID of the client whose model copy to update.
            delta (OrderedDict): The parameter updates received from the client.
        """
        if self.client_models is None or self.layer_names is None:
             print("Warning: Cannot update client model, server state not initialized.")
             return

        client_idx = int(client_id.split('_')[-1]) - 1

        # Ensure delta is on the correct device
        delta_on_device = {name: param.to(self.device) for name, param in delta.items()}

        updated_params_list = []
        current_client_params = self.client_models[client_idx]

        # Iterate through current params and names simultaneously
        for i, (param_name, current_param) in enumerate(zip(self.layer_names, current_client_params)):
            if param_name in delta_on_device:
                 # Update: current_param = current_param + delta
                 new_param_data = (current_param.data + delta_on_device[param_name].data).clone().detach()
                 updated_params_list.append(new_param_data)
            else:
                 # If delta doesn't contain this param (e.g., non-trainable), keep original
                 updated_params_list.append(current_param.clone().detach()) # Keep a detached copy

        # Replace the old list of parameters with the new list
        self.client_models[client_idx] = updated_params_list
        # print(f"Server-side model copy for {client_id} updated.")


    def train_round(self, round_num: Optional[int] = None) -> Tuple[float, float, Dict[str, float]]:
        """
        Executes one communication round for pFedLA.

        Orchestrates model generation, distribution, client training, delta collection,
        and updates to server-side client models and the HyperNetwork.

        Returns:
            Tuple[float, float, Dict[str, float]]: Aggregated train loss, validation loss,
                                                   and validation scores from clients.
        """
        if self.hypernetwork is None:
             raise RuntimeError("pFedLA server cannot train, HyperNetwork not initialized.")

        agg_train_loss = 0.0
        agg_val_loss = 0.0
        agg_val_score = {}
        start_time = time.time()

        client_ids = list(self.clients.keys())
        print(f"pFedLA Round: Processing {len(client_ids)} clients...")
        for client_id in client_ids:
            client = self.clients[client_id]
            # print(f" Processing client {client_id}...")

            # 1. Generate personalized model using HN
            personalized_state_dict = self.generate_client_model(client_id)

            # 2. Distribute personalized model to client
            # Client's set_model_state also stores this as initial_params
            client.set_model_state(personalized_state_dict)

            # 3. Client trains locally
            client_train_loss = client.train(personal=False) # pFedLA client uses main state slot
            client_val_loss, client_val_score = client.validate(personal=False)

            # 4. Client computes updates (delta)
            delta = client.compute_updates()

            # 5. Server updates its copy of the client model
            self.update_client_model(client_id, delta)

            # 6. Server updates the HyperNetwork using the delta
            self.update_hypernetwork(client_id, delta)

            # 7. Aggregate metrics
            weight = client.data.weight
            agg_train_loss += client_train_loss * weight
            agg_val_loss += client_val_loss * weight
            agg_val_score = self._aggregate_scores(agg_val_score, client_val_score, weight)

        mid_time = time.time()
        print(f"pFedLA Round: Client processing finished. Time: {mid_time - start_time:.2f}s.")

        # Track aggregated metrics
        self.serverstate.train_losses.append(agg_train_loss)
        self.serverstate.val_losses.append(agg_val_loss)
        self.serverstate.val_scores.append(agg_val_score)
        if agg_val_loss < self.serverstate.best_loss:
            self.serverstate.best_loss = agg_val_loss
            self.serverstate.best_model = copy.deepcopy(self.serverstate.model).to(self.device)
            print(f"Server (pFedLA): Best aggregated validation loss updated to {agg_val_loss:.4f}. (Note: 'best_model' snapshot might be reference backbone).")

        return agg_train_loss, agg_val_loss, agg_val_score


    # Override aggregation/distribution methods to do nothing, as pFedLA handles updates differently
    def aggregate_models(self):
        """Override: pFedLA does not use standard FedAvg aggregation."""
        pass

    def distribute_global_model(self):
        """Override: pFedLA distributes personalized models, not a single global one."""
        pass