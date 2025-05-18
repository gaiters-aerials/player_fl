"""
Defines the client-side logic for various federated learning algorithms.

Includes:
- Dataclasses for configuration (`TrainerConfig`), site data (`SiteData`),
  and model state (`ModelState`).
- A `MetricsCalculator` for evaluating model predictions.
- A base `Client` class handling core training, validation, and testing loops.
- Specialized client classes inheriting from `Client` to implement algorithm-specific
  behaviors (e.g., `FedProxClient`, `DittoClient`, `LayerClient`, `pFedLAClient`).
"""
from helper import * # Import helper functions (move_to_device, cleanup_gpu, etc.)
from configs import * # Import configurations (DEVICE, etc.)


@dataclass
class TrainerConfig:
    """
    Configuration dataclass for client training parameters.

    Attributes:
        dataset_name (str): Name of the dataset being used.
        device (str): The compute device ('cuda' or 'cpu').
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training and evaluation loaders.
        epochs (int): Number of local training epochs per communication round.
        rounds (int): Total number of communication rounds (used for context, not directly by client).
        num_clients (int): Total number of clients (used for context).
        requires_personal_model (bool): Flag indicating if the algorithm requires
                                        a separate personalized model state (e.g., Ditto, pFedMe).
        algorithm_params (Optional[Dict]): Dictionary holding algorithm-specific parameters
                                           (e.g., regularization strength, layers to federate).
    """
    dataset_name: str
    device: str
    learning_rate: float
    batch_size: int
    epochs: int = 5         # Default local epochs
    rounds: int = 20        # Default total rounds
    num_clients: int = 5    # Default number of clients
    requires_personal_model: bool = False
    algorithm_params: Optional[Dict] = None


@dataclass
class SiteData:
    """
    Holds the DataLoaders and metadata for a single client site.

    Attributes:
        site_id (str): A unique identifier for the client site.
        train_loader (DataLoader): DataLoader for the client's training data.
        val_loader (DataLoader): DataLoader for the client's validation data.
        test_loader (DataLoader): DataLoader for the client's test data.
        weight (float): The weight of this client in federated aggregation,
                        typically proportional to its number of training samples.
                        Defaults to 1.0, usually updated by the server.
        num_samples (int): The number of samples in the training dataset for this client.
                           Automatically calculated from train_loader.
    """
    site_id: str
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    weight: float = 1.0 # Initial weight, server usually recalculates
    num_samples: int = field(init=False) # Calculated after initialization

    def __post_init__(self):
        """Calculates num_samples after the object is initialized."""
        if self.train_loader is not None and self.train_loader.dataset is not None:
            self.num_samples = len(self.train_loader.dataset)
        else:
            self.num_samples = 0
            print(f"Warning: Could not determine num_samples for site {self.site_id}. Train loader or dataset is None.")


@dataclass
class ModelState:
    """
    Holds the state associated with a single model (global or personalized).

    Includes the model itself, optimizer, loss criterion, tracking lists for
    metrics, and references to the best performing model state found so far.

    Attributes:
        model (nn.Module): The PyTorch neural network model.
        optimizer (torch.optim.Optimizer): The optimizer instance for the model.
        criterion (nn.Module): The loss function.
        best_loss (float): The best validation loss achieved so far. Initialized to infinity.
        best_model (Optional[nn.Module]): A deep copy of the model state that achieved `best_loss`.
        train_losses (List[float]): List tracking average training loss per epoch/round.
        val_losses (List[float]): List tracking average validation loss per epoch/round.
        val_scores (List[Dict]): List tracking validation metrics (dict) per epoch/round.
        test_losses (List[float]): List tracking average test loss (usually only one final value).
        test_scores (List[Dict]): List tracking test metrics (usually only one final value).
        best_metrics (Dict): Dictionary storing the best test metrics achieved by `best_model`.
    """
    model: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    best_loss: float = float('inf')
    best_model: Optional[nn.Module] = None
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_scores: List[Dict] = field(default_factory=list) # Store dicts of scores
    test_losses: List[float] = field(default_factory=list)
    test_scores: List[Dict] = field(default_factory=list) # Store dicts of scores

    # Stores the test metrics associated with the best_model snapshot
    best_metrics: Dict = field(default_factory=lambda: {
        'loss': float('inf'),
        'accuracy': 0.0,
        'balanced_accuracy': 0.0,
        'f1_macro': 0.0,
        'f1_weighted': 0.0,
        'mcc': 0.0 # Matthews Correlation Coefficient
    })

    def __post_init__(self):
        """Initializes best_model with a copy of the initial model if not provided."""
        if self.best_model is None and self.model is not None:
            # Ensure the best_model is on the same device as the original model
            device = next(self.model.parameters()).device
            self.best_model = copy.deepcopy(self.model).to(device)

    def copy(self):
        """
        Creates a deep copy of this ModelState.

        This is crucial for algorithms that maintain separate global and
        personalized model states (like Ditto, pFedMe). It ensures the copied
        model and optimizer are independent.

        Returns:
            ModelState: A new ModelState instance with deep copies of the model,
                        optimizer state, and other attributes reset or copied as appropriate.
        """
        # Create a new instance of the model architecture
        device = next(self.model.parameters()).device
        new_model = copy.deepcopy(self.model).to(device)

        # Create a new optimizer instance for the new model
        # Copy optimizer state if needed (important for adaptive optimizers like Adam)
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        new_optimizer = type(self.optimizer)(new_model.parameters(), **self.optimizer.defaults)
        try:
            new_optimizer.load_state_dict(optimizer_state)
        except ValueError as e:
            # This can happen if the model structure changes unexpectedly or if the state is incompatible.
            print(f"Warning: Could not load optimizer state during ModelState copy. Optimizer state reset. Error: {e}")
            # Re-initialize optimizer without loading state if loading fails
            new_optimizer = type(self.optimizer)(new_model.parameters(), **self.optimizer.defaults)


        # Create the new ModelState instance. We reset metric lists and best loss/model.
        return ModelState(
            model=new_model,
            optimizer=new_optimizer,
            criterion=self.criterion # Loss function can usually be shared (it's stateless)
            # best_loss, best_model, train_losses, etc., will use their defaults (reset state)
        )

class MetricsCalculator:
    """
    Utility class for calculating classification performance metrics.
    """
    def __init__(self, dataset_name: str):
        """
        Initializes the MetricsCalculator.

        Args:
            dataset_name (str): The name of the dataset (currently unused,
                                but could be used for dataset-specific logic later).
        """
        self.dataset_name = dataset_name

    def process_predictions(self, labels: Tensor, predictions: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes raw model output (logits) and labels into NumPy arrays for metric calculation.

        Args:
            labels (Tensor): Ground truth labels (typically 1D tensor).
            predictions (Tensor): Raw model outputs (logits, typically 2D tensor [batch_size, num_classes]).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - labels_np: NumPy array of ground truth labels.
                - predictions_np: NumPy array of predicted class indices (after argmax).
        """
        # Move tensors to CPU and convert to NumPy arrays
        predictions_np = predictions.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        # Convert logits to predicted class indices
        if predictions_np.ndim > 1 and predictions_np.shape[1] > 1:
             # Assumes classification task with multiple output neurons (logits)
            predictions_np = predictions_np.argmax(axis=1)
        elif predictions_np.ndim == 1 or predictions_np.shape[1] == 1:
             # Handle binary classification or single output neuron cases if needed
             # Example for binary with threshold 0.5:
             # predictions_np = (predictions_np > 0.5).astype(int).flatten()
             pass # Assuming argmax handles the current cases correctly


        return labels_np, predictions_np

    def calculate_metrics(self, labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """
        Calculates various classification metrics.

        Args:
            labels (np.ndarray): Ground truth labels.
            predictions (np.ndarray): Predicted class indices.

        Returns:
            Dict[str, float]: A dictionary containing calculated metrics:
                              'accuracy', 'balanced_accuracy', 'f1_macro',
                              'f1_weighted', 'mcc'. Returns zeros if calculation fails.
        """
        try:
            # Ensure inputs are not empty
            if labels.size == 0 or predictions.size == 0:
                 print("Warning: Empty labels or predictions array passed to calculate_metrics.")
                 return {
                     'accuracy': 0.0, 'balanced_accuracy': 0.0, 'f1_macro': 0.0,
                     'f1_weighted': 0.0, 'mcc': 0.0
                 }

            # Calculate metrics
            accuracy = (predictions == labels).mean()
            # Use zero_division=0 to handle cases where a class might have no predictions/labels in a batch/subset
            balanced_accuracy = balanced_accuracy_score(labels, predictions)
            f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
            f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(labels, predictions)

            return {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'mcc': mcc
            }
        except ValueError as e:
            # Catch potential errors
            print(f"Warning: Could not calculate some metrics. Error: {e}")


class Client:
    """
    Base Client class for federated learning.

    Manages local model state(s), data loaders, and handles the core logic for
    training, validation, and testing on the client's local data. Can manage
    both a global model state and an optional personalized model state.
    """
    def __init__(self,
                 config: TrainerConfig,
                 data: SiteData,
                 modelstate: ModelState,
                 metrics_calculator: MetricsCalculator,
                 personal_model: bool = False):
        """
        Initializes the Client.

        Args:
            config (TrainerConfig): Configuration for training parameters.
            data (SiteData): DataLoaders and metadata for this client.
            modelstate (ModelState): The initial model state (usually the global model state
                                     provided by the server). A copy is made for the client.
            metrics_calculator (MetricsCalculator): An instance for calculating metrics.
            personal_model (bool): If True, creates and manages a separate personal model
                                   state, initialized as a copy of the global state.
                                   Used by algorithms like Ditto, pFedMe.
        """
        self.config = config
        self.data = data
        self.device = config.device
        self.metrics_calculator = metrics_calculator

        # Initialize global model state for this client (a copy of the server's state)
        self.global_state = modelstate.copy() # Makes an independent copy for the client

        # Create a separate personal model state if required by the algorithm
        self.personal_state = self.global_state.copy() if personal_model else None

        # Ensure models are on the correct device initially
        self.global_state.model.to(self.device)
        if self.personal_state:
            self.personal_state.model.to(self.device)

    def get_client_state(self, personal: bool) -> ModelState:
        """
        Retrieves the appropriate model state (global or personal).

        Args:
            personal (bool): If True, returns the personal model state.
                             Otherwise, returns the global model state.

        Returns:
            ModelState: The requested model state object.

        Raises:
            ValueError: If `personal` is True but no personal state exists.
        """
        if personal:
            if self.personal_state is None:
                raise ValueError("Requested personal model state, but it was not initialized.")
            return self.personal_state
        else:
            return self.global_state

    def set_model_state(self, state_dict: OrderedDict):
        """
        Loads a state dictionary into the client's global model.

        Typically used by the server to distribute the updated global model.

        Args:
            state_dict (OrderedDict): The state dictionary to load.
        """
        state = self.get_client_state(personal=False)
        try:
            state.model.load_state_dict(state_dict)
            state.model.to(self.device) # Ensure model stays on device after loading state
        except RuntimeError as e:
            print(f"Error loading state dict for client {self.data.site_id}: {e}")

    def update_best_model(self, loss: float, personal: bool):
        """
        Updates the best model state if the current validation loss is lower.

        Stores a deep copy of the model that achieved the best validation loss so far.

        Args:
            loss (float): The current validation loss.
            personal (bool): If True, updates the best personal model; otherwise,
                             updates the best global model snapshot.

        Returns:
            bool: True if the best model was updated, False otherwise.
        """
        state = self.get_client_state(personal)
        if loss < state.best_loss:
            state.best_loss = loss
            # Store a copy of the *current* model state as the new best
            state.best_model = copy.deepcopy(state.model).to(self.device)
            return True
        return False

    def train_epoch(self, personal: bool) -> float:
        """
        Performs one epoch of training on the local training data.

        Args:
            personal (bool): If True, trains the personal model; otherwise,
                             trains the global model.

        Returns:
            float: The average training loss for this epoch.
        """
        state = self.get_client_state(personal)
        model = state.model.to(self.device) # Ensure model is on device
        model.train() # Set model to training mode
        total_loss = 0.0
        num_batches = 0

        try:
            for batch_x, batch_y in self.data.train_loader:
                # Move data to the appropriate device
                batch_x = move_to_device(batch_x, self.device)
                batch_y = move_to_device(batch_y, self.device)

                # Standard training step
                state.optimizer.zero_grad()
                outputs = model(batch_x)
                loss = state.criterion(outputs, batch_y)
                loss.backward()
                state.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            state.train_losses.append(avg_loss) # Track epoch loss
            return avg_loss

        except Exception as e:
             print(f"Error during training epoch for client {self.data.site_id}: {e}")
             return float('inf') # Return high loss on error
        finally:
            # Clean up tensors to potentially free GPU memory, though Python's GC handles most cases.
            # Explicit del might help in very memory-constrained scenarios.
            del batch_x, batch_y, outputs, loss
            # Optionally move model back to CPU if memory is extremely tight between clients/rounds
            # model.to('cpu')
            # cleanup_gpu() # Calling cleanup_gpu frequently can slow things down

    def train(self, personal: bool) -> float:
        """
        Runs the training process for the configured number of local epochs.

        Args:
            personal (bool): If True, trains the personal model; otherwise,
                             trains the global model.

        Returns:
            float: The average training loss from the final local epoch.
        """
        final_loss = 0.0
        for epoch in range(self.config.epochs):
            epoch_loss = self.train_epoch(personal)
            if epoch == self.config.epochs - 1:
                final_loss = epoch_loss
        return final_loss

    def evaluate(self, loader: DataLoader, personal: bool, validate: bool) -> Tuple[float, Dict[str, float]]:
        """
        Evaluates the model on a given DataLoader (validation or test).

        Args:
            loader (DataLoader): The DataLoader to evaluate on.
            personal (bool): If True, evaluates the personal model; otherwise,
                             evaluates the global model.
            validate (bool): If True, evaluates the *current* model state.
                             If False (testing), evaluates the *best* model state saved during validation.

        Returns:
            Tuple[float, Dict[str, float]]: A tuple containing:
                - Average loss on the dataloader.
                - Dictionary of calculated performance metrics.
        """
        state = self.get_client_state(personal)
        # Choose the model to evaluate: current for validation, best for testing
        model_to_eval = state.model if validate else state.best_model
        if model_to_eval is None:
             print(f"Warning: Model for evaluation ({'validation' if validate else 'testing'}, {'personal' if personal else 'global'}) is None for client {self.data.site_id}. Returning high loss and zero metrics.")
             return float('inf'), { 'accuracy': 0.0, 'balanced_accuracy': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0, 'mcc': 0.0 }

        model = model_to_eval.to(self.device) # Ensure model is on device
        model.eval() # Set model to evaluation mode

        total_loss = 0.0
        num_batches = 0
        all_predictions_np = []
        all_labels_np = []

        try:
            with torch.no_grad(): # Disable gradient calculations for evaluation
                for batch_x, batch_y in loader:
                    batch_x = move_to_device(batch_x, self.device)
                    batch_y = move_to_device(batch_y, self.device)

                    outputs = model(batch_x)
                    loss = state.criterion(outputs, batch_y)
                    total_loss += loss.item()
                    num_batches += 1

                    # Process predictions and labels for metric calculation
                    labels, predictions = self.metrics_calculator.process_predictions(
                        batch_y, outputs
                    )
                    all_predictions_np.extend(predictions)
                    all_labels_np.extend(labels)

            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            # Calculate metrics using aggregated predictions and labels
            metrics = self.metrics_calculator.calculate_metrics(
                np.array(all_labels_np),
                np.array(all_predictions_np)
            )
            return avg_loss, metrics

        except Exception as e:
             print(f"Error during evaluation for client {self.data.site_id}: {e}")
             # raise e
             return float('inf'), { 'accuracy': 0.0, 'balanced_accuracy': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0, 'mcc': 0.0 }
        finally:
            del batch_x, batch_y, outputs, loss
            # Optionally move model back to CPU
            # model.to('cpu')
            # cleanup_gpu()


    def validate(self, personal: bool) -> Tuple[float, Dict[str, float]]:
        """
        Validates the current model state on the validation DataLoader.

        Updates the best model state if performance improves.

        Args:
            personal (bool): If True, validates the personal model; otherwise,
                             validates the global model.

        Returns:
            Tuple[float, Dict[str, float]]: Validation loss and metrics.
        """
        state = self.get_client_state(personal)
        val_loss, val_metrics = self.evaluate(
            self.data.val_loader,
            personal,
            validate=True # Evaluate the current model
        )

        # Track validation performance
        state.val_losses.append(val_loss)
        state.val_scores.append(val_metrics)

        # Update the best model based on this validation loss
        self.update_best_model(val_loss, personal)
        return val_loss, val_metrics

    def test(self, personal: bool) -> Tuple[float, Dict[str, float]]:
        """
        Tests the best performing model state (found during validation) on the test DataLoader.

        Args:
            personal (bool): If True, tests the best personal model; otherwise,
                             tests the best global model.

        Returns:
            Tuple[float, Dict[str, float]]: Test loss and metrics.
        """
        state = self.get_client_state(personal)
        test_loss, test_metrics = self.evaluate(
            self.data.test_loader,
            personal,
            validate=False # Evaluate the best model saved previously
        )

        # Track test performance (usually called once at the end)
        state.test_losses.append(test_loss)
        state.test_scores.append(test_metrics)

        # Store these metrics also in the best_metrics dictionary
        state.best_metrics['loss'] = test_loss
        state.best_metrics.update(test_metrics)


        return test_loss, test_metrics


# --- Algorithm-Specific Client Implementations ---

class FedProxClient(Client):
    """
    Client implementation for the FedProx algorithm.

    Inherits from the base `Client` and overrides the `train_epoch` method
    to include the proximal regularization term in the loss calculation.
    The proximal term penalizes deviation from the global model received
    from the server.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the FedProxClient.

        Retrieves the regularization parameter (mu or reg_param) from the config.
        """
        super().__init__(*args, **kwargs)
        if 'reg_param' not in self.config.algorithm_params:
             raise ValueError("FedProx requires 'reg_param' (mu) in algorithm_params.")
        self.reg_param = self.config.algorithm_params['reg_param'] # mu parameter in FedProx paper

    def train_epoch(self, personal: bool = False) -> float:
        """
        Performs one epoch of FedProx training.

        Adds the proximal term to the standard loss before backpropagation.
        Note: FedProx typically operates on the global model (personal=False).

        Args:
            personal (bool): Should generally be False for standard FedProx.
                             Included for consistency with the base class signature.

        Returns:
            float: The average *original* training loss (excluding proximal term) for the epoch.
        """
        if personal:
             print("Warning: FedProxClient training called with personal=True. FedProx typically updates the global model.")

        state = self.get_client_state(personal=False) # FedProx modifies the main model state
        model = state.model.to(self.device)
        model.train()
        # Get a reference to the global model parameters *before* training starts
        # The server must ensure global_state.model holds the model from the start of the round.
        global_model_params = [p.detach().clone() for p in self.global_state.model.parameters()] # Crucial: Use the initial global model

        total_original_loss = 0.0
        num_batches = 0

        try:
            for batch_x, batch_y in self.data.train_loader:
                batch_x = move_to_device(batch_x, self.device)
                batch_y = move_to_device(batch_y, self.device)

                state.optimizer.zero_grad()
                outputs = model(batch_x)
                # Calculate the primary task loss (e.g., CrossEntropy)
                loss = state.criterion(outputs, batch_y)

                # Calculate the FedProx proximal term
                proximal_term = self.compute_proximal_term(
                    model.parameters(),
                    global_model_params, # Compare against the initial global model of the round
                )

                # Combine the task loss and the proximal term
                total_loss_batch = loss + proximal_term
                total_loss_batch.backward()

                state.optimizer.step()
                total_original_loss += loss.item() # Track the original loss, not including prox term
                num_batches += 1

            avg_loss = total_original_loss / num_batches if num_batches > 0 else 0.0
            state.train_losses.append(avg_loss)
            return avg_loss

        except Exception as e:
             print(f"Error during FedProx training epoch for client {self.data.site_id}: {e}")
             return float('inf')
        finally:
            del batch_x, batch_y, outputs, loss, proximal_term, total_loss_batch, global_model_params
            # cleanup_gpu()

    def compute_proximal_term(self, model_params: Iterator[nn.Parameter], reference_params: List[Tensor]) -> Tensor:
        """
        Calculates the proximal term for FedProx.

        Term = (mu / 2) * || w - w_global ||^2

        Args:
            model_params (Iterator[nn.Parameter]): Parameters of the current local model.
            reference_params (List[Tensor]): Parameters of the reference model (global model at round start).

        Returns:
            Tensor: The calculated proximal term (scalar).
        """
        proximal_term = torch.tensor(0.0, device=self.device)
        # Use zip ensuring parameters are aligned correctly
        for param, ref_param in zip(model_params, reference_params):
             if param.requires_grad: # Only include trainable parameters
                 # Ensure ref_param is on the same device
                 ref_param_device = ref_param.to(self.device)
                 proximal_term += torch.norm(param - ref_param_device, p=2)**2
        return (self.reg_param / 2) * proximal_term


class PFedMeClient(Client):
    """
    Client implementation for the pFedMe algorithm.

    pFedMe aims for personalization by optimizing a personalized model `theta`
    while using a proximal term to keep it close to an updated version of the
    global model `w`. The client trains its *personal* model state.

    Note: The original pFedMe involves multiple steps (approximating `w`, then
    updating `theta`). This implementation simplifies by directly regularizing
    the personal model towards the current global model state during personal training.
    A more faithful implementation might require a custom optimizer or more complex training loop.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the pFedMeClient. Requires a personal model state.

        Retrieves the regularization parameter (lambda in pFedMe paper) from config.
        """
        super().__init__(*args, **kwargs)
        if 'reg_param' not in self.config.algorithm_params:
             raise ValueError("pFedMe requires 'reg_param' (lambda) in algorithm_params.")
        # lambda in pFedMe paper, controls penalty for deviating from global model
        self.reg_param = self.config.algorithm_params['reg_param']

    def train_epoch(self, personal: bool = True) -> float:
        """
        Performs one epoch of pFedMe training on the *personal* model.

        Adds a proximal term regularizing the personal model towards the
        current global model state.

        Args:
            personal (bool): Should always be True for pFedMe client training.

        Returns:
            float: The average *original* training loss (excluding proximal term) for the epoch.
        """
        if not personal:
             print("Warning: pFedMeClient train_epoch called with personal=False. pFedMe trains the personal model.")
             # Decide whether to proceed or raise error. Let's proceed but log warning.

        state = self.get_client_state(personal=True) # pFedMe trains the personal model
        model = state.model.to(self.device)
        model.train()
        # Global model state is used as the reference for regularization
        global_model = self.global_state.model.to(self.device)
        global_model_params = [p.detach().clone() for p in global_model.parameters()] # Reference params

        total_original_loss = 0.0
        num_batches = 0

        try:
            for batch_x, batch_y in self.data.train_loader:
                batch_x = move_to_device(batch_x, self.device)
                batch_y = move_to_device(batch_y, self.device)

                state.optimizer.zero_grad()
                outputs = model(batch_x) # Forward pass using the personal model
                loss = state.criterion(outputs, batch_y)

                # Calculate the proximal term (like FedProx, but personal vs global)
                proximal_term = self.compute_proximal_term(
                    model.parameters(),
                    global_model_params, # Regularize towards the *current* global model
                )

                total_batch_loss = loss + proximal_term
                total_batch_loss.backward()

                state.optimizer.step()
                total_original_loss += loss.item()
                num_batches += 1

            avg_loss = total_original_loss / num_batches if num_batches > 0 else 0.0
            state.train_losses.append(avg_loss)
            return avg_loss

        except Exception as e:
             print(f"Error during pFedMe training epoch for client {self.data.site_id}: {e}")
             return float('inf')
        finally:
            del batch_x, batch_y, outputs, loss, proximal_term, total_batch_loss, global_model, global_model_params
            # cleanup_gpu()

    def train(self, personal: bool = True) -> float:
        """
        Runs the pFedMe training process for the configured number of local epochs.

        Defaults to training the personal model (`personal=True`).

        Args:
            personal (bool): Should be True for standard pFedMe client training.

        Returns:
            float: The average training loss from the final local epoch.
        """
        return super().train(personal) # Always train personal model

    def compute_proximal_term(self, model_params: Iterator[nn.Parameter], reference_params: List[Tensor]) -> Tensor:
        """
        Calculates the proximal term for pFedMe.

        Term = (lambda / 2) * || theta - w ||^2
        where theta are personal parameters, w are global parameters.

        Args:
            model_params (Iterator[nn.Parameter]): Parameters of the current personal model.
            reference_params (List[Tensor]): Parameters of the reference model (current global model).

        Returns:
            Tensor: The calculated proximal term (scalar).
        """
        proximal_term = torch.tensor(0.0, device=self.device)
        for param, ref_param in zip(model_params, reference_params):
            if param.requires_grad:
                 ref_param_device = ref_param.to(self.device)
                 proximal_term += torch.norm(param - ref_param_device, p=2)**2
        return (self.reg_param / 2) * proximal_term


class DittoClient(Client):
    """
    Client implementation for the Ditto algorithm.

    Ditto maintains both a global model (trained similarly to FedAvg) and a
    separate personalized model. The personalized model is trained locally
    with a regularization term that penalizes its deviation from the global model,
    applied directly to the gradients during the personal training phase.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the DittoClient. Requires a personal model state.

        Retrieves the regularization parameter (lambda) from config.
        """
        super().__init__(*args, **kwargs)
        if 'reg_param' not in self.config.algorithm_params:
             raise ValueError("Ditto requires 'reg_param' (lambda) in algorithm_params.")
        self.reg_param = self.config.algorithm_params['reg_param'] # lambda in Ditto paper


    def train_epoch(self, personal: bool) -> float:
        """
        Performs one epoch of training, either for the global or personal model.

        - If `personal` is False: Performs a standard training epoch on the global model.
        - If `personal` is True: Performs a training epoch on the personal model,
          adding gradient regularization based on the difference between personal
          and global models.

        Args:
            personal (bool): Determines whether to train the global (False) or
                             personal (True) model.

        Returns:
            float: The average training loss for the epoch.
        """
        if not personal:
            # --- Global Model Training Step (like FedAvg) ---
            return super().train_epoch(personal=False)
        else:
            # --- Personal Model Training Step (with Ditto regularization) ---
            state = self.get_client_state(personal=True)
            model = state.model.to(self.device)
            model.train()
            # Global model state is needed for regularization
            global_model = self.global_state.model.to(self.device)
            # Detach global parameters to avoid computing gradients for them here
            global_model_params = [p.detach().clone() for p in global_model.parameters()]

            total_loss = 0.0
            num_batches = 0

            try:
                for batch_x, batch_y in self.data.train_loader:
                    batch_x = move_to_device(batch_x, self.device)
                    batch_y = move_to_device(batch_y, self.device)

                    state.optimizer.zero_grad()
                    outputs = model(batch_x) # Forward pass using personal model
                    loss = state.criterion(outputs, batch_y)
                    loss.backward() # Compute initial gradients based on task loss

                    # Apply Ditto's gradient regularization
                    self.add_gradient_regularization(
                        model.parameters(),
                        global_model_params # Compare against the global model state
                    )

                    state.optimizer.step() # Update personal model parameters
                    total_loss += loss.item()
                    num_batches += 1

                avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
                state.train_losses.append(avg_loss)
                return avg_loss

            except Exception as e:
                 print(f"Error during Ditto personal training epoch for client {self.data.site_id}: {e}")
                 return float('inf')
            finally:
                del batch_x, batch_y, outputs, loss, global_model, global_model_params
                # cleanup_gpu()

    def add_gradient_regularization(self, model_params: Iterator[nn.Parameter], reference_params: List[Tensor]):
        """
        Adds the Ditto regularization term directly to the gradients.

        Regularization: gradient = gradient + lambda * (personal_param - global_param)

        Args:
            model_params (Iterator[nn.Parameter]): Parameters of the personal model (with gradients).
            reference_params (List[Tensor]): Parameters of the reference model (global model state).
        """
        for param, ref_param in zip(model_params, reference_params):
            if param.requires_grad and param.grad is not None:
                # Ensure ref_param is on the correct device
                ref_param_device = ref_param.to(param.device)
                # Calculate the regularization term: lambda * (w_personal - w_global)
                reg_term = self.reg_param * (param.detach() - ref_param_device) # Use detached param to avoid loop in grad calc
                # Add the regularization term to the existing gradient
                param.grad.add_(reg_term)


class LocalAdaptationClient(Client):
    """
    Client implementation for algorithms involving local adaptation after federation.

    Standard training rounds behave like the base `Client`. A special `train`
    method allows for different behavior (e.g., fewer epochs) in a final
    adaptation phase, controlled by the `final_round` flag passed by the server.
    This is used by algorithms like FedAvg + Finetuning.
    """
    def __init__(self, *args, **kwargs):
        """Initializes the LocalAdaptationClient."""
        super().__init__(*args, **kwargs)

    def train(self, personal: bool, final_round: bool = False) -> float:
        """
        Runs the training process, potentially with modified behavior in the final round.

        Args:
            personal (bool): If True, trains the personal model; otherwise, the global model.
                             (Typically False for this strategy).
            final_round (bool): If True, triggers the final adaptation phase (e.g., might
                                run only `train_epoch` once instead of multiple epochs).
                                If False, performs standard multi-epoch training.

        Returns:
            float: The average training loss from the final local epoch performed.
        """
        if not final_round:
            # Standard training phase (multiple epochs)
            return super().train(personal)
        else:
            # Final adaptation phase (e.g., only one epoch)
            print(f"Client {self.data.site_id}: Performing final local adaptation epoch.")
            return self.train_epoch(personal) # Run just one epoch


class LayerClient(Client):
    """
    Base client for layer-wise federated learning algorithms (LayerPFL, BABU).

    Overrides `set_model_state` to selectively update only the layers
    specified for federation, preserving the state of local-only layers.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the LayerClient.

        Retrieves the list of layer name prefixes to be federated from config.
        """
        # The server needs to place 'layers_to_include' into algorithm_params
        if 'layers_to_include' not in kwargs['config'].algorithm_params:
             raise ValueError("LayerClient requires 'layers_to_include' in algorithm_params.")
        self.layers_to_include = kwargs['config'].algorithm_params['layers_to_include']
        super().__init__(*args, **kwargs)

    def set_model_state(self, state_dict: OrderedDict, personal: bool = False):
        """
        Selectively loads the received state dictionary.

        Only parameters whose names contain prefixes listed in `layers_to_include`
        are updated from the `state_dict`. Other parameters retain their current values.

        Args:
            state_dict (OrderedDict): The state dictionary received (usually from the server,
                                      containing updates for federated layers).
            personal (bool): If True, updates the personal model; otherwise, the global model.
                             (Typically False for layer-wise methods acting on the main model).
        """
        state = self.get_client_state(personal)
        current_state = state.model.state_dict()

        # Iterate through the parameters in the received state_dict
        for name, param_update in state_dict.items():
            # Check if this parameter belongs to a layer designated for federation
            if any(layer_prefix in name for layer_prefix in self.layers_to_include):
                 # Check if the parameter exists in the current model's state
                 if name in current_state:
                     # Copy the updated parameter value from the received state_dict
                     current_state[name].copy_(param_update)
                 else:
                     print(f"Warning: Parameter '{name}' found in received state_dict but not in client model. Skipping.")

        # Load the modified state dictionary back into the model
        state.model.load_state_dict(current_state)
        state.model.to(self.device) # Ensure model stays on device

class LayerPFLClient(LayerClient):
    """
    Client implementation for LayerPFL (fixed or random subset).

    Inherits the selective state update mechanism from `LayerClient`.
    The specific layers included are determined by the `layers_to_include`
    parameter passed during initialization (set by the server based on config).
    """
    def __init__(self, *args, **kwargs):
        """Initializes the LayerPFLClient."""
        super().__init__(*args, **kwargs)


class BABUClient(LayerClient):
    """
    Client implementation for the BABU (Body-And-Body Update) algorithm.

    Inherits selective state update from `LayerClient`. Adds functionality to
    switch between training only the "body" layers (federated layers) and
    training only the "head" layer (local classification layer) during a
    final fine-tuning phase.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the BABUClient.

        Sets the model to train only the body layers by default.
        """
        super().__init__(*args, **kwargs)
        # Initialize the model to train only the body layers initially.
        self.set_head_body_training(train_head=False)

    def set_head_body_training(self, train_head: bool):
        """
        Configures the model to train either the head or the body layers.

        Freezes gradients for the layers not being trained. Resets the
        optimizer state when switching modes, as the set of trainable parameters changes.

        Args:
            train_head (bool): If True, sets the head layer(s) to be trainable
                               and freezes the body. If False, sets the body layer(s)
                               to be trainable and freezes the head.
        """
        state = self.get_client_state(personal=False) # BABU operates on the main model
        model = state.model

        print(f"Client {self.data.site_id}: Setting training mode to {'Head' if train_head else 'Body'}")

        # Identify parameters belonging to head vs body
        head_param_names = []
        body_param_names = []
        trainable_params = []

        for name, param in model.named_parameters():
            # Determine if the parameter is part of the head or body
            # A parameter is part of the head if its name does NOT contain any of the 'body' prefixes.
            is_body = any(layer_prefix in name for layer_prefix in self.layers_to_include)
            is_head = not is_body

            # Set requires_grad based on the desired training mode
            if train_head:
                param.requires_grad = is_head
            else: # Train body
                param.requires_grad = is_body

            # Collect names for logging and parameters for the new optimizer
            if param.requires_grad:
                trainable_params.append(param)
                if is_head:
                    head_param_names.append(name)
                else:
                    body_param_names.append(name)
            # else: param remains frozen

        # Reset the optimizer to only consider the currently trainable parameters
        # This discards any momentum or state from the previous optimizer mode.
        current_optimizer_defaults = state.optimizer.defaults
        state.optimizer = type(state.optimizer)(
            trainable_params, # Pass only the parameters that require gradients
            **current_optimizer_defaults
        )


    def train_head(self) -> float:
        """
        Trains only the head layer(s) of the model for the configured number of epochs.

        Typically called during the final round of BABU.

        Returns:
            float: The average training loss from the final head-tuning epoch.
        """
        print(f"Client {self.data.site_id}: Starting head training phase.")
        # Configure model and optimizer for head training
        self.set_head_body_training(train_head=True)
        # Run the standard training loop (which now only affects the head)
        final_loss = self.train(personal=False) # Use standard train method
        return final_loss


class FedLPClient(Client):
    """
    Client implementation for the FedLP (Federated Learning with Layer-wise
    Probabilistic model aggregation) algorithm.

    Implements logic to probabilistically decide which layers participate in
    aggregation for each round and provides the state of participating layers
    to the server.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the FedLPClient.

        Retrieves the layer preservation rate (probability `p`) from config.
        """
        super().__init__(*args, **kwargs)
        if 'layer_preserving_rate' not in self.config.algorithm_params:
             raise ValueError("FedLP requires 'layer_preserving_rate' (p) in algorithm_params.")
        # Probability 'p' that a layer's update is preserved/aggregated
        self.layer_preserving_rate = self.config.algorithm_params['layer_preserving_rate']

    def generate_layer_indicators(self) -> Dict[str, int]:
        """
        Generates binary indicators (0 or 1) for each logical layer in the model.

        A '1' indicates the layer participates in aggregation for this round,
        determined by a Bernoulli trial with probability `layer_preserving_rate`.
        A '0' indicates the layer does not participate.

        Returns:
            Dict[str, int]: A dictionary mapping base layer names (e.g., 'conv1', 'fc2')
                            to their participation indicator (0 or 1).
        """
        indicators = {}
        state = self.get_client_state(personal=False) # FedLP operates on the main model state

        # Get unique base layer names (e.g., 'layer1', 'fc1' from 'layer1.weight', 'fc1.bias')
        # Assumes layer names are typically separated by '.'
        layer_names = set(name.split('.')[0] for name, _ in state.model.named_parameters())

        for layer_name in layer_names:
            # Perform a Bernoulli trial for each unique layer
            participates = random.random() < self.layer_preserving_rate
            indicators[layer_name] = 1 if participates else 0

        return indicators

    def get_pruned_model_state(self) -> Tuple[OrderedDict, Dict[str, int]]:
        """
        Generates layer participation indicators and returns the model state dictionary
        containing only the parameters from participating layers.

        Returns:
            Tuple[OrderedDict, Dict[str, int]]: A tuple containing:
                - pruned_state: State dictionary with parameters only from layers with indicator=1.
                - indicators: The dictionary of layer participation indicators used.
        """
        # Generate indicators for this round
        indicators = self.generate_layer_indicators()
        state = self.get_client_state(personal=False)
        full_state_dict = state.model.state_dict()
        pruned_state = OrderedDict()

        # Iterate through the full state dictionary
        for name, param in full_state_dict.items():
            # Determine the base layer name for this parameter
            layer_name = name.split('.')[0]
            # Include the parameter if its layer is participating
            if layer_name in indicators and indicators[layer_name] == 1:
                pruned_state[name] = param.clone().detach() # Send a detached copy

        return pruned_state, indicators


class FedLAMAClient(Client):
    """
    Client implementation for the FedLAMA algorithm.

    FedLAMA's core logic (adaptive aggregation frequency) resides on the server side.
    The client behaves like a standard FedAvg client during its local training phase.
    It doesn't require special handling of personal models or layer selection here.
    """
    def __init__(self, *args, **kwargs):
        """Initializes the FedLAMAClient."""
        super().__init__(*args, **kwargs)
        # No special initialization needed on the client side for FedLAMA


class pFedLAClient(Client):
    """
    Client implementation for the pFedLA (Personalized Federated Learning via
    Layer-wise Aggregation weights) algorithm.

    pFedLA uses a hypernetwork on the server to generate personalized aggregation
    weights. The client receives a fully personalized model state from the server,
    trains it locally, and computes the parameter *updates* (delta) compared to
    the model state it started the round with. These updates are sent back to
    the server to update the hypernetwork and the server's copies of client models.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the pFedLAClient.

        Stores the initial model parameters received from the server to compute
        updates later. pFedLA typically operates on the 'global_state' slot,
        as the personalization happens via the server generating the full model state.
        """
        super().__init__(*args, **kwargs)

        # Store the initial parameters received from the server at the start of the round.
        self.initial_params = OrderedDict()
        self._store_initial_params()

    def _store_initial_params(self):
        """Stores a deep copy of the current model's parameters."""
        state = self.get_client_state(personal=False) # pFedLA uses the main state slot
        self.initial_params = OrderedDict() # Reset before storing
        for name, param in state.model.state_dict().items():
            self.initial_params[name] = param.clone().detach()

    def set_model_state(self, state_dict: OrderedDict):
        """
        Overrides the base method to load the personalized model state received
        from the server and immediately store it as the initial parameters for
        the upcoming training round.

        Args:
            state_dict (OrderedDict): The personalized model state generated by the server's hypernetwork.
        """
        # Load the state received from the server
        super().set_model_state(state_dict) # Loads into self.global_state.model
        # Store these newly received parameters as the baseline for calculating updates
        self._store_initial_params()

    def train(self, personal: bool = False) -> float:
        """
        Runs the local training process.

        After training, the `initial_params` (stored at the start of the call
        to `set_model_state` or `__init__`) remain unchanged, allowing `compute_updates`
        to calculate the difference correctly.

        Args:
            personal (bool): Should always be False for pFedLA client logic, as the
                             personalization is handled by the model received.

        Returns:
            float: The average training loss from the final local epoch.
        """
        if personal:
             print("Warning: pFedLAClient train called with personal=True. Using personal=False.")

        # Train the model received from the server
        final_loss = super().train(personal=False)

        # Note: We do NOT update initial_params here. They represent the state *before* this training.
        # compute_updates will compare the *current* state.model against self.initial_params.

        return final_loss

    def compute_updates(self) -> OrderedDict:
        """
        Computes the difference (delta) between the current model parameters
        (after local training) and the initial parameters stored at the beginning
        of the round.

        Returns:
            OrderedDict: A state dictionary containing the parameter updates (deltas).
        """
        updates = OrderedDict()
        state = self.get_client_state(personal=False)
        current_params = state.model.state_dict()

        if not self.initial_params:
             print(f"Error: Client {self.data.site_id} cannot compute updates, initial_params not stored.")
             # Return empty updates or raise error
             return updates

        for name, current_param in current_params.items():
            if name in self.initial_params:
                # Ensure both tensors are on the same device before subtraction
                initial_param_device = self.initial_params[name].to(current_param.device)
                # Calculate the update delta
                updates[name] = (current_param - initial_param_device).clone().detach()
            else:
                # This case should ideally not happen if model structure is consistent
                print(f"Warning: Parameter '{name}' found in current model but not in initial parameters. Skipping update calculation for this parameter.")

        return updates