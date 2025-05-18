ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
import sys
sys.path.append(f'{ROOT_DIR}/code')
from helper import move_to_device, cleanup_gpu 
from configs import  *
from clients import Client
from layer_analytics import  *


@dataclass
class TrainerConfig:
    """Configuration for training parameters."""
    dataset_name: str
    device: str
    learning_rate: float
    batch_size: int
    epochs: int = 5
    rounds: int = 20
    num_clients: int = 5
    requires_personal_model: bool = False
    algorithm_params: Optional[Dict] = None
    num_cpus: int = 4 # Added for analysis

@dataclass
class SiteData:
    """Holds DataLoader and metadata for a site."""
    site_id: str
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    weight: float = 1.0
    
    def __post_init__(self):
        if self.train_loader is not None:
            self.num_samples = len(self.train_loader.dataset)

@dataclass
class ModelState:
    """Holds state for a single model (global or personalized)."""
    model: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    best_loss: float = float('inf')
    best_model: Optional[nn.Module] = None
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_scores: List[float] = field(default_factory=list)
    test_losses: List[float] = field(default_factory=list)
    test_scores: List[float] = field(default_factory=list)

    def __post_init__(self):
        if self.best_model is None and self.model is not None:
            self.best_model = copy.deepcopy(self.model).to(next(self.model.parameters()).device)
    
    def copy(self):
        """Create a new ModelState with copied model and optimizer."""
        # Create new model instance
        new_model = copy.deepcopy(self.model).to(next(self.model.parameters()).device)
        
        # Setup optimizer
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        new_optimizer = type(self.optimizer)(new_model.parameters(), **self.optimizer.defaults)
        new_optimizer.load_state_dict(optimizer_state)
        
        # Create new model state
        return ModelState(
            model=new_model,
            optimizer=new_optimizer,
            criterion= self.criterion 
        )
    
class MetricsCalculator:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def process_predictions(self, labels, predictions):
        """Process model predictions"""
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        predictions = predictions.argmax(axis=1)
            
        return labels, predictions

    def calculate_metrics(self, labels, predictions):
        """Calculate multiple classification metrics."""
        return {
            'accuracy': (predictions == labels).mean(),
            'balanced_accuracy': balanced_accuracy_score(labels, predictions),
            'f1_macro': f1_score(labels, predictions, average='macro'),
            'f1_weighted': f1_score(labels, predictions, average='weighted'),
            'mcc': matthews_corrcoef(labels, predictions)
        }


# --- AnalyticsClient ---
class AnalyticsClient(Client):
    """
    Extends the base Client to add methods for local model analysis and
    activation extraction. FIX: Uses RandomSampler for metrics batch.
    """
    def __init__(self,
                 config: TrainerConfig,
                 data: SiteData,
                 modelstate: ModelState,
                 metrics_calculator: MetricsCalculator,
                 personal_model: bool = False): # personal_model likely not needed if using get_client_state
        # Pass personal_model to super if it uses it, otherwise can remove
        super().__init__(config, data, modelstate, metrics_calculator, personal_model=personal_model)
        print(f"AnalyticsClient created for site {self.data.site_id}")
        self.requires_personal_model = config.requires_personal_model # Store this if needed


    def _get_random_data_sample(self, seed: int) -> Optional[Tuple[Union[Tensor, Tuple], Tensor]]:
        """
        Uses the provided seed for reproducibility of the sample.
        """
        if self.data.train_loader is None or self.data.train_loader.dataset is None or len(self.data.train_loader.dataset) == 0:
            print(f"Warning: Client {self.data.site_id} train_loader or dataset is empty/None.")
            return None
        try:
            # Create a generator with the specified seed
            g = torch.Generator()
            g.manual_seed(seed)
            
            # Create a new RandomSampler with the generator
            sampler = RandomSampler(
                self.data.train_loader.dataset, 
                replacement=True, 
                num_samples=self.data.train_loader.batch_size,
                generator=g
            )
            
            # Temporarily create a new DataLoader with this sampler
            random_loader = DataLoader(
                dataset=self.data.train_loader.dataset,
                batch_size=self.data.train_loader.batch_size,
                sampler=sampler,
                collate_fn=self.data.train_loader.collate_fn,
                num_workers=0,  # Use 0 workers for seeded sampling consistency
                pin_memory=self.data.train_loader.pin_memory,
                generator=g  # Also set the generator for the DataLoader
            )
            
            data_batch = next(iter(random_loader))
            del random_loader # Clean up temporary loader
            del sampler
            return data_batch
        except StopIteration:
            print(f"Warning: Client {self.data.site_id} random_loader failed (StopIteration).")
            return None
        except Exception as e:
             print(f"Error getting random data batch for client {self.data.site_id}: {e}")
             traceback.print_exc()
             return None


    def run_local_metrics_calculation(self, seed: int) -> pd.DataFrame:
        """
        Performs local model analysis (weights, gradients, Hessian).
        """
        print(f"Client {self.data.site_id}: Running local metrics calculation (seed={seed})...")
        state = self.get_client_state(personal=self.requires_personal_model) # Use the correct model state
        if state is None or state.model is None or state.criterion is None:
             print(f"Error: Client {self.data.site_id} state, model, or criterion is None.")
             return pd.DataFrame()

        model_to_analyze = state.model
        criterion = state.criterion

        # --- Sample data using the seeded random sampler ---
        data_batch = self._get_random_data_sample(seed)
        if data_batch is None:
            print(f"Client {self.data.site_id}: Failed to get data batch. Cannot calculate metrics.")
            return pd.DataFrame()

        # --- Perform forward/backward pass HERE to get grads for analysis ---
        metrics_df = pd.DataFrame() # Initialize empty DataFrame
        attention_data = None # For embedding layers
        try:
             # Ensure model is on the correct device and in train mode
            with ModelDeviceManager(model_to_analyze, self.device, eval_mode=False) as model_on_device:
                model_on_device.train() # Ensure train mode for gradients

                features, labels = data_batch
                # Check for tuple features (e.g., for attention masks)
                if isinstance(features, (list, tuple)):
                     # Assume (input_ids, attention_mask) format
                     if len(features) >= 2 and isinstance(features[0], torch.Tensor) and isinstance(features[1], torch.Tensor):
                         attention_data = (features[0], features[1]) # Pass tokens and mask
                     else:
                          print(f"Warning: Client {self.data.site_id} received tuple features but format unexpected. Proceeding without attention data for Hessian.")
                     # Ensure features are moved to device correctly
                     features = move_to_device(features, self.device)
                else:
                    # Standard tensor features
                    features = move_to_device(features, self.device)

                labels = move_to_device(labels, self.device)

                model_on_device.zero_grad()
                outputs = model_on_device(features)
                loss = criterion(outputs, labels)

                # *** CRITICAL: Calculate gradients with create_graph=True ***
                loss.backward(create_graph=True, retain_graph=True)

                # --- Now call the analysis function with the gradients computed ---
                metrics_df = calculate_local_layer_metrics(
                    model=model_on_device, # Pass the model with grads
                    device=self.device,    # Pass device
                    hvp_seed=seed,         # Pass the seed specifically for HVP random vector 'v'
                    attention_data=attention_data # Pass token/mask data if available
                )

                # --- Clean up gradients after analysis ---
                model_on_device.zero_grad(set_to_none=True)
                del loss, outputs, features, labels # Manual cleanup

        except Exception as e:
             print(f"Error during Client {self.data.site_id} local metrics calculation (incl. grad computation): {e}")
             traceback.print_exc()
             # Ensure gradients are cleared even if error occurs mid-process
             try:
                 model_to_analyze.zero_grad(set_to_none=True)
             except Exception:
                 pass # Ignore errors during cleanup
             metrics_df = pd.DataFrame() # Return empty on error
        finally:
             cleanup_gpu() # Clear GPU cache

        print(f"Client {self.data.site_id}: Local metrics calculation finished.")
        return metrics_df


    def run_activation_extraction(self, probe_data_batch: Union[Tensor, Tuple]) -> List[Tuple[str, np.ndarray]]:
        """
        Extracts model activations using the server-provided probe data batch.
        (No change needed here based on the fixes, relies on server providing correct batch)
        """
        print(f"Client {self.data.site_id}: Running activation extraction...")
        # Determine which model state to use based on config/context
        # Usually for activations, you compare the *final* models (local best or global)
        state = self.get_client_state(personal=self.requires_personal_model)
        if state is None or state.model is None:
             print(f"Error: Client {self.data.site_id} state or model is None for activation extraction.")
             return []

        model_to_analyze = state.model # Or state.best_model if comparing best models

        activations = get_model_activations(
            model=model_to_analyze,
            probe_data_batch=probe_data_batch,
            device=self.device,
            site_id=self.data.site_id # Pass site_id for storage in hook
        )
        print(f"Client {self.data.site_id}: Activation extraction finished.")
        return activations

