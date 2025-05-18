"""
Module for calculating detailed model analysis metrics including layer weights,
gradient importance, Hessian properties, and activation similarity.
"""
ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/layer_pfl'
import sys
sys.path.append(f'{ROOT_DIR}/code')
from helper import move_to_device, cleanup_gpu 
from configs import *

cpus = int(os.getenv('SLURM_CPUS_PER_TASK', 6))

class ModelDeviceManager:
    """Context manager to temporarily move a model to a device."""
    def __init__(self, model: nn.Module, device: str, eval_mode: bool = False):
        self.model = model
        self.device = device
        try:
            self.original_device = next(model.parameters()).device
        except StopIteration:
            self.original_device = 'cpu' # Fallback if no parameters
        except AttributeError: # Handle models with no parameters initially
            self.original_device = 'cpu'


        self.original_mode = model.training
        self.eval_mode = eval_mode

    def __enter__(self):
        try:
            # Only move if the model has parameters
            if len(list(self.model.parameters())) > 0:
                 self.model.to(self.device)
            else:
                 print("Warning: Model has no parameters, skipping move to device.")
        except Exception as e:
            print(f"Warning: Failed to move model to device {self.device}. Error: {e}")
            pass
        if self.eval_mode:
            self.model.eval()
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            # Only move back if the model has parameters
             if len(list(self.model.parameters())) > 0:
                self.model.to(self.original_device)
             else:
                print("Warning: Model has no parameters, skipping move back to original device.")
        except Exception as e:
             print(f"Warning: Failed to move model back to device {self.original_device}. Error: {e}")
        if self.eval_mode and self.original_mode:
            self.model.train()
        cleanup_gpu()

# --- Weight/Importance Metrics (Keep as is) ---
def _model_weight_data(param: Tensor) -> Dict[str, float]:
    param_cpu = param.detach().cpu()
    data = {
        'Weight Mean': abs(param_cpu).mean().item(),
        'Weight Variance': param_cpu.var().item(),
    }
    for threshold in [0.01, 0.05, 0.1]:
        small_weights = (param_cpu.abs() < threshold).sum().item()
        data[f'% Weights < {threshold}'] = 100 * small_weights / param_cpu.numel() if param_cpu.numel() > 0 else 0
    return data

def _model_weight_importance(param: Tensor, gradient: Tensor, data: Optional[Tuple[Tensor, Tensor]] = None, emb_table: bool = False) -> Dict[str, float]:
    # FIX: Added data and emb_table handling like old code
    param_cpu = param.detach().cpu()
    grad_cpu = gradient.detach().cpu()

    if emb_table:
        token_indices, mask = data
        token_indices_cpu = token_indices.cpu()
        # Check if indices are within bounds
        if token_indices_cpu.max() < param_cpu.shape[0]:
             param_cpu = param_cpu[token_indices_cpu].mean(dim=0)
             grad_cpu = grad_cpu[token_indices_cpu].mean(dim=0)
        else:
             print(f"Warning: token_indices max {token_indices_cpu.max()} out of bounds for param shape {param_cpu.shape}. Skipping embedding averaging.")
             # Fallback or handle error - here we just use the original tensors which might be wrong
             pass # Use full tensors if indices are bad


    importance = (param_cpu * grad_cpu).pow(2).sum().item()
    importance_per = importance / param_cpu.numel() if param_cpu.numel() > 0 else 0
    grad_var = grad_cpu.var().item()
    return {
        'Gradient Importance': importance,
        'Gradient Importance per': importance_per,
        'Gradient Variance': grad_var
    }

# --- Eigenvalue Helper (Keep as is for now, only computes SVD) ---
def _compute_eigenvalues(matrix: Tensor) -> Tuple[np.ndarray, float]:
    try:
        # Ensure matrix is float and on CPU for stability if needed, though svdvals handles devices
        matrix_float = matrix.float().cpu()
        eigenvalues = torch.linalg.svdvals(matrix_float) # SVD values
        sum_eigenvalues = torch.sum(eigenvalues).item()
        return eigenvalues.cpu().numpy(), sum_eigenvalues
    except torch.linalg.LinAlgError:
        print(f"Warning: SVD computation failed for matrix of shape {matrix.shape}. Returning zeros.")
        num_eigenvalues = min(matrix.shape)
        return np.zeros(num_eigenvalues), 0.0
    except Exception as e:
        print(f"Warning: Error during SVD computation for matrix of shape {matrix.shape}: {e}. Returning zeros.")
        num_eigenvalues = min(matrix.shape)
        return np.zeros(num_eigenvalues), 0.0

# --- MODIFIED Hessian Metrics ---
def _hessian_metrics(param: Tensor, hvp_seed: int, data: Optional[Tuple[Tensor, Tensor]] = None, emb_table: bool = False) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate Hessian-related metrics using the existing gradient on the parameter.
    Assumes the gradient was computed with create_graph=True.
    """
    metrics = {
        'SVD Sum EV': np.nan, 'EV Skewness': np.nan, '% EV small': np.nan,
        '% EV neg': np.nan, 'Gradient Importance 2': np.nan,
        'Gradient Importance per 2': np.nan, 'Hessian Variance': np.nan,
        'Condition Number': np.nan, 'Operator norm': np.nan,
        'SVD Eigenvalues': np.array([]),
    }

    if not param.requires_grad or param.grad is None or not param.grad.requires_grad:
        return metrics

    first_grads = param.grad # This holds the gradient dL/dw

    try:
        # --- Generate random vector 'v' using Generator ---
        generator = torch.Generator(device=param.device)
        generator.manual_seed(hvp_seed)
        v = torch.randn(param.size(), generator=generator, device=param.device, dtype=param.dtype)
        v_norm = torch.norm(v)
        if v_norm > 1e-8: v = v / v_norm
        else: return metrics

        # Calculate Hessian-vector product (HVP): H*v = d/dw(dL/dw) * v
        Hv = torch.autograd.grad(
            outputs=first_grads,
            inputs=param,
            grad_outputs=v,
            retain_graph=True,
            allow_unused=True
        )[0]

        if Hv is None:
            print(f"Warning: Hessian-vector product (Hv) is None for parameter shape {param.shape}.")
            return metrics

        # --- Prepare Processed Versions for Importance Metric ---
        Hv_for_importance = Hv.detach().clone()
        param_for_importance = param.detach().clone()
        is_conv = param_for_importance.dim() == 4
        # Define is_emb clearly for both importance and SVD logic sections
        is_emb = emb_table and data is not None and param_for_importance.dim() == 2

        if is_emb:
            token_indices, mask = data
            token_indices_flat = token_indices.reshape(-1)
            token_indices_cpu = token_indices_flat.cpu()
            # Check bounds before averaging for importance
            if token_indices_cpu.numel() > 0 and token_indices_cpu.max() < param_for_importance.shape[0]:
                param_for_importance = param_for_importance[token_indices_cpu].mean(dim=0)
                Hv_for_importance = Hv_for_importance[token_indices_cpu].mean(dim=0)
            else:
                 print(f"Warning: Invalid token indices for embedding importance averaging. Using full tensors for importance.")
                 pass # Importance calculation will use non-averaged tensors

        # --- Second-order Importance (Uses potentially averaged param/Hv) ---
        param_cpu = param_for_importance.cpu()
        Hv_cpu = Hv_for_importance.cpu()
        importance2 = (param_cpu * Hv_cpu).pow(2).sum().item()
        importance_per2 = importance2 / param_cpu.numel() if param_cpu.numel() > 0 else 0
        metrics['Gradient Importance 2'] = importance2
        metrics['Gradient Importance per 2'] = importance_per2
        metrics['Hessian Variance'] = Hv_cpu.var().item() # Variance of tensor used for importance

        # --- SVD Calculation (on original Hv or slices) ---
        svd_e = np.array([]) # Initialize result array
        svd_sum_e = 0.0      # Initialize result sum

        original_hv_detached = Hv.detach() # Use original HVP for SVD logic

        if is_conv:
            # --- Convolutional Logic  ---
            c_out, c_in, k, k = original_hv_detached.shape
            num_channels_processed = 0
            temp_svd_sum_e = 0.0
            svd_e_list = []
            for channel in range(c_in):
                hv_channel = original_hv_detached[:, channel, :, :].reshape(c_out, -1)
                if hv_channel.numel() > 0 and min(hv_channel.shape) > 0:
                     svd_e_c, svd_sum_e_c = _compute_eigenvalues(hv_channel)
                     svd_e_list.append(torch.from_numpy(svd_e_c))
                     temp_svd_sum_e += svd_sum_e_c
                     num_channels_processed += 1
            if num_channels_processed > 0:
                 svd_sum_e = temp_svd_sum_e / num_channels_processed
                 svd_e_all = torch.cat(svd_e_list) if svd_e_list else torch.tensor([])
                 if svd_e_all.numel() > 0:
                     svd_e = (svd_e_all / num_channels_processed).numpy()

        elif is_emb:
            # original_hv_detached has shape (VocabSize, EmbDim)
            print(f"Calculating SVD for Embedding layer HVP matrix (shape: {original_hv_detached.shape})")
            if original_hv_detached.numel() > 0 and min(original_hv_detached.shape) > 0:
                 # Treat the HVP as a 2D matrix and compute its SVD values
                 svd_e, svd_sum_e = _compute_eigenvalues(original_hv_detached.float())
            else:
                 print(f"Warning: Skipping SVD for empty/invalid embedding HVP matrix shape: {original_hv_detached.shape}")
                 # svd_e/svd_sum_e remain empty/zero

        elif original_hv_detached.dim() >= 1 and original_hv_detached.numel() > 0: # FC layers or others
            # --- FC Layer Logic (Remains the same) ---
            matrix_for_svd = original_hv_detached.reshape(original_hv_detached.shape[0], -1).float()
            if matrix_for_svd.numel() > 0 and min(matrix_for_svd.shape) > 0:
                 svd_e, svd_sum_e = _compute_eigenvalues(matrix_for_svd)
            # else svd_e/svd_sum_e remain empty/zero
        else:
             print(f"Skipping SVD for HVP tensor with invalid shape: {original_hv_detached.shape}")
             # svd_e/svd_sum_e remain empty/zero


        # --- Calculate Metrics from SVD Eigenvalues (if available) ---
        # This part uses the computed svd_e (numpy array) and svd_sum_e (float)
        # which are now calculated consistently for FC and Embedding layers
        if len(svd_e) > 0:
            svd_e = svd_e[np.isfinite(svd_e)] # Clean up non-finite values
            if len(svd_e) > 0 and svd_e.max() > 1e-12:
                metrics['SVD Eigenvalues'] = svd_e
                metrics['SVD Sum EV'] = svd_sum_e
                metrics['Operator norm'] = svd_e.max().item()
                min_positive_ev = svd_e[svd_e > 1e-8].min() if np.any(svd_e > 1e-8) else 1e-8
                # Avoid division by zero or near-zero
                if min_positive_ev > 1e-12:
                    metrics['Condition Number'] = metrics['Operator norm'] / min_positive_ev
                else:
                     metrics['Condition Number'] = np.inf

                try:
                    metrics['EV Skewness'] = scipy.stats.skew(svd_e)
                except ValueError:
                    metrics['EV Skewness'] = np.nan
                threshold = 0.01 * metrics['Operator norm']
                metrics['% EV small'] = 100 * np.sum(svd_e < threshold) / len(svd_e)
                metrics['% EV neg'] = 100 * np.sum(svd_e < -1e-8) / len(svd_e)
            else:
                 print(f"Warning: All SVD eigenvalues are zero or non-finite after cleaning for param shape {param.shape}. Setting SVD metrics to NaN.")

    except Exception as e:
         print(f"Error during Hessian metrics calculation for param shape {param.shape}: {e}")
         traceback.print_exc()
         metrics = {k: np.nan if isinstance(v, (float, np.floating)) else (np.array([]) if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}

    return metrics


# --- MODIFIED Analyse Layer ---
def _analyse_layer(param: Tensor, hvp_seed: int, data: Optional[Tuple[Tensor, Tensor]] = None, emb_table: bool = False) -> Optional[Dict[str, Union[float, np.ndarray]]]:
    """
    Analyse a single layer's weights and gradients, including Hessian metrics.
    """
    if param.grad is None:
        return None

    layer_data = _model_weight_data(param)
    # Pass data/emb_table flag for correct importance calculation if needed
    layer_data.update(_model_weight_importance(param, param.grad, data=data, emb_table=emb_table))

    # Pass hvp_seed, data, emb_table for consistent HVP and correct Hessian processing
    hessian_data = _hessian_metrics(param, hvp_seed=hvp_seed, data=data, emb_table=emb_table)
    layer_data.update(hessian_data)

    return layer_data

# --- MODIFIED Calculate Local Layer Metrics ---
def calculate_local_layer_metrics(model: nn.Module,
                                  device: str,
                                  hvp_seed: int, # Pass the specific seed for HVP calculation
                                  attention_data: Optional[Tuple[Tensor, Tensor]] = None # Explicitly pass token_indices/mask if needed
                                  ) -> pd.DataFrame:
    """
    Calculates weight stats, gradient importance, and Hessian metrics
    for each layer using a single data batch.
    """
    layer_data_dict = {}

    try:
        # --- Forward/Backward Pass (ASSUMED TO BE DONE *BEFORE* CALLING THIS FUNCTION) ---
        # This function now *analyzes* the gradients already present on the model parameters.
        # The gradients must have been computed with create_graph=True externally.
        with ModelDeviceManager(model, device, eval_mode=False) as model_on_device: # Keep in train mode for grads
            named_params = dict(model_on_device.named_parameters())

            for name, param in named_params.items():
                if not param.requires_grad or param.grad is None or "bias" in name:
                    continue

                # --- Determine Layer Type and Prepare Data for Analysis ---
                emb_table = False
                layer_data_for_hessian = None # Specific data needed for Hessian/Importance

                # Simple check for embedding table names
                if 'token_embedding' in name.lower():
                     emb_table = True
                     layer_data_for_hessian = attention_data # Pass tokens/mask
                     # print(f"Analyzing Embedding Layer: {name}")

                # Keep simple name extraction
                layer_name_parts = name.split('.')[:-1] # e.g., ['encoder', 'layers', '0', 'self_attn', 'out_proj']
                simple_name = '.'.join(layer_name_parts) if layer_name_parts else name # e.g., encoder.layers.0.self_attn.out_proj

                # --- Call Analyse Layer ---
                # Pass the full parameter always. _analyse_layer and _hessian_metrics handle dims.
                layer_metrics = _analyse_layer(
                    param=param,
                    hvp_seed=hvp_seed,
                    data=layer_data_for_hessian,
                    emb_table=emb_table
                )

                if layer_metrics:
                    # Store results using the simplified layer name
                    layer_data_dict[simple_name] = layer_metrics


    except Exception as e:
        print(f"Error during local layer metrics calculation: {e}")
        traceback.print_exc()
        return pd.DataFrame()
    finally:
        cleanup_gpu() # Still good practice

    # --- Format Output (Keep as is) ---
    if not layer_data_dict:
        print("Warning: No layer metrics were calculated.")
        return pd.DataFrame()

    valid_layer_data = {
        name: metrics for name, metrics in layer_data_dict.items()
        if metrics and not all(np.isnan(v) if isinstance(v, (float, np.floating)) else False for v in metrics.values())
    }
    if not valid_layer_data:
        print("Warning: All calculated layer metrics resulted in NaN values or analysis failed.")
        return pd.DataFrame()

    # Remove large eigenvalue arrays before creating DataFrame
    for layer, metrics in valid_layer_data.items():
        if 'SVD Eigenvalues' in metrics:
            del metrics['SVD Eigenvalues']

    return pd.DataFrame.from_dict(valid_layer_data, orient='index')


# --- Activation Similarity Logic (Hooks remain the same) ---

_activation_hooks = []
_activation_dict: Dict[str, List[Tuple[str, np.ndarray]]] = {}

def _hook_fn(layer_name: str, site_id: str):
    """Hook function to capture activations."""
    def hook(module, input, output):
        global _activation_dict
        # Detach, move to CPU, convert to NumPy
        # Handle potential tuple outputs (e.g., some transformers) - take the first tensor
        if isinstance(output, tuple):
             activation_tensor = output[0]
        else:
             activation_tensor = output
        # Ensure it's a tensor before detaching
        if isinstance(activation_tensor, torch.Tensor):
            activation = activation_tensor.detach().cpu().numpy()
            if site_id not in _activation_dict:
                _activation_dict[site_id] = []
            _activation_dict[site_id].append((layer_name, activation))
        # Else: Skip non-tensor outputs silently or add a warning
    return hook

def _register_hooks(model: nn.Module, site_id: str):
    """Registers forward hooks on relevant modules."""
    global _activation_hooks
    for handle in _activation_hooks:
        handle.remove()
    _activation_hooks = []

    for name, module in model.named_modules():
        # More inclusive hook registration for various layer types
        is_leaf = len(list(module.children())) == 0

        # Hook if it's a relevant compute layer or the root module for simple models
        if is_leaf:
            hook_name = name if name else type(module).__name__
            handle = module.register_forward_hook(_hook_fn(hook_name, site_id))
            _activation_hooks.append(handle)
            # print(f"Registered hook for {hook_name}")  # Uncomment for debugging

def _remove_hooks():
    """Removes all registered hooks."""
    global _activation_hooks
    for handle in _activation_hooks:
        handle.remove()
    _activation_hooks = []


def get_model_activations(model: nn.Module,
                          probe_data_batch: Union[Tensor, Tuple],
                          device: str,
                          site_id: str) -> List[Tuple[str, np.ndarray]]:
    """
    Runs a probe data batch through the model and captures activations
    from registered hooks.
    """
    global _activation_dict
    _activation_dict[site_id] = []  # Reset activations for this site_id

    try:
        # Use eval_mode=True for activations - typically done without dropout etc.
        with ModelDeviceManager(model, device, eval_mode=True) as model_on_device:
            _register_hooks(model_on_device, site_id)

            # Handle different probe_data_batch formats
            if isinstance(probe_data_batch, tuple) and len(probe_data_batch) >= 1:
                features = probe_data_batch[0]
            else:
                features = probe_data_batch
                print(f"Warning: Unexpected probe_data_batch format: {type(probe_data_batch)}")
            
            features = move_to_device(features, device)

            # Run forward pass to collect activations
            with torch.no_grad():
                try:
                    _ = model_on_device(features)
                except Exception as fwd_err:
                    print(f"Error in model forward pass: {fwd_err}")
                    traceback.print_exc()

            _remove_hooks()

    except Exception as e:
        print(f"Error during activation extraction for site {site_id}: {e}")
        traceback.print_exc()
        _remove_hooks()  # Ensure hooks are removed on error
        return []
    finally:
        cleanup_gpu()

    activations = _activation_dict.get(site_id, [])
    if not activations:
        print(f"Warning: No activations collected for site {site_id}")
    return activations


# --- MODIFIED Activation Similarity Calculation ---
def calculate_activation_similarity(activations_dict: Dict[str, List[Tuple[str, np.ndarray]]],
                                   probe_data_batch: Optional[Union[Tensor, Tuple]] = None,
                                   cpus: int = 4) -> Dict[str, pd.DataFrame]:
    """
    Calculates pairwise similarity between site activations layer by layer using NetRep.
    Fixed version with proper layer skipping, robust mask handling, and error recovery.
    Correctly handles embedding layers based on their dimensionality.
    """
    if not activations_dict:
        return {}

    comparison_data = {}
    site_ids = list(activations_dict.keys())
    if len(site_ids) < 2:
        print("Warning: Need activations from at least 2 sites for similarity comparison.")
        return {}

    # Assume all sites have the same layers in the same order
    first_site_id = site_ids[0]
    if not activations_dict[first_site_id]:
        print(f"Warning: No activations found for the first site {first_site_id}.")
        return {}

    # Get layer names from the first site, PROPERLY skipping the last layer like old code
    layer_names_with_acts = activations_dict[first_site_id]
    if len(layer_names_with_acts) <= 1:
        print("Warning: Not enough layers with activations to compare (only 1 or 0).")
        return {}
    
    # Skip the last layer which is just the prediction (no longer a represenation)
    layer_names = [name for name, _ in layer_names_with_acts]
    
    # More robust mask extraction with multiple fallbacks
    mask = None
    is_attention_model = False
    
    # Try to extract mask from probe_data_batch using various methods
    if probe_data_batch is not None:
    # Extract from tuple where first element is (data, mask)
        if isinstance(probe_data_batch[0], tuple) and len(probe_data_batch[0]) >= 2:
            potential_mask = probe_data_batch[0][1]  # Second element of features tuple
            if isinstance(potential_mask, torch.Tensor):
                mask = potential_mask.cpu().numpy()
                is_attention_model = True
                print("Detected attention mask")
        
    # Convert mask to boolean if needed
    if mask is not None and mask.dtype != bool:
        mask = mask.astype(bool)
    
    print(f"Mask detected: {mask is not None}, Attention model: {is_attention_model}")
    print(f"Processing {len(layer_names)} layers: {layer_names}")

    for layer_idx, layer_name in enumerate(layer_names):
        print(f"Processing layer {layer_idx+1}/{len(layer_names)}: {layer_name}")
        
        # Force attention model detection for embedding or 3D activation layers with 'attention' in name
        layer_is_attention = is_attention_model
        
        # Gather activations for this layer from all sites
        layer_activations = []
        valid_site_ids_for_layer = []
        
        for site_id in site_ids:
            # Check if site has this layer
            if layer_idx < len(activations_dict[site_id]):
                site_layer_name, activation_data = activations_dict[site_id][layer_idx]
                
                # Verify layer name matches 
                if site_layer_name == layer_name and isinstance(activation_data, np.ndarray):
                    # Special detection for transformer layers by shape or name
                    if len(activation_data.shape) == 3 and ('attention' in layer_name.lower()):
                        layer_is_attention = True
                    
                    layer_activations.append(activation_data)
                    valid_site_ids_for_layer.append(site_id)
                else:
                    print(f"Warning: Layer mismatch for site {site_id}. Expected '{layer_name}', got '{site_layer_name}'")
            else:
                print(f"Warning: Site {site_id} has fewer layers than expected")

        if len(layer_activations) < 2:
            print(f"Skipping layer {layer_name}: Need at least two sites with valid activations.")
            continue

        # Check shapes consistency 
        first_shape = layer_activations[0].shape
        if not all(act.shape == first_shape for act in layer_activations):
            print(f"Warning: Activation shapes inconsistent for layer {layer_name}.")
            # Log the differing shapes
            for i, site_id in enumerate(valid_site_ids_for_layer):
                print(f"  Site {site_id}: {layer_activations[i].shape}")
            print("Skipping layer due to shape inconsistency.")
            continue

        current_num_sites = len(valid_site_ids_for_layer)
        act_shape = first_shape
        
        # More nuanced layer type detection
        is_conv_layer = len(act_shape) == 4  # NCHW format from PyTorch hooks
        
        # Check for embedding in name
        is_embedding_layer = 'embedding' in layer_name.lower()
        
        # Check for transformer/attention characteristics - needs 3D shape for mask operations
        is_transformer_layer = (
            (layer_is_attention and len(act_shape) == 3) or  # Only if 3D
            (len(act_shape) == 3 and ('attention' in layer_name.lower())) # 3D attention layers
        )
        
        # For reporting only
        layer_type = "Conv" if is_conv_layer else (
            "Transformer3D" if is_transformer_layer else (
                "Embedding2D" if is_embedding_layer and len(act_shape) == 2 else (
                    "Embedding3D" if is_embedding_layer and len(act_shape) == 3 else "Linear/Other"
                )
            )
        )
        
        print(f"Layer {layer_name}: Type={layer_type}, Shape={act_shape}")

        # Use LinearMetric with consistent settings
        metric = LinearMetric(alpha=1.0, center_columns=True, score_method="angular")
        
        # Initialize with zeros like old code (not NaN)
        result_matrix = np.zeros((current_num_sites, current_num_sites))
        adjustment = 0.0  # Default

        try:
            # Different processing based on layer type
            if is_conv_layer:
                # Transpose to NHWC format for convolve_metric (as in old code)
                acts_nhwc = [np.transpose(act, (0, 2, 3, 1)) for act in layer_activations]
                
                # Process pairwise combinations
                for i, j in combinations(range(current_num_sites), 2):
                    try:
                        # Try with float64 casting first
                        dist = convolve_metric(metric, acts_nhwc[i].astype(np.float64), 
                                              acts_nhwc[j].astype(np.float64), num_processes = cpus)
                        min_dist = dist.min()  # Get minimum distance as per old code
                        result_matrix[i, j] = min_dist
                        result_matrix[j, i] = min_dist
                    except Exception as conv_err:
                        print(f"Error in convolve_metric for sites {i},{j}: {conv_err}")
                        # Try fallback without explicit casting
                        try:
                            dist = convolve_metric(metric, acts_nhwc[i], acts_nhwc[j], num_processes = cpus)
                            min_dist = dist.min()
                            result_matrix[i, j] = min_dist
                            result_matrix[j, i] = min_dist
                            print("Fallback succeeded")
                        except Exception as fallback_err:
                            print(f"Fallback also failed: {fallback_err}")
                print(result_matrix, flush = True)
                # Set adjustment factor for conv layers (from old code)
                adjustment = 0.0  # Keep as-is for conv layers
                
            elif is_transformer_layer and mask is not None:
                # For attention/transformer layers with mask
                try:
                    # Ensure mask shape compatibility
                    n_batch, seq_len = act_shape[0], act_shape[1]
                    
                    if mask.shape[0] < n_batch or mask.shape[1] < seq_len:
                        print(f"Resizing mask from {mask.shape} to match activations {(n_batch, seq_len)}")
                        # Broadcast/tile the mask to match activation dimensions
                        repeated_mask = np.repeat(mask[:1], n_batch, axis=0)
                        mask_aligned = repeated_mask[:, :seq_len]
                    else:
                        mask_aligned = mask[:n_batch, :seq_len]
                    
                    # Apply mask and pool
                    mask_expanded = np.expand_dims(mask_aligned, axis=-1)
                    masked_acts = [act * mask_expanded for act in layer_activations]
                    
                    # Sum over sequence dimension as in old code
                    pooled_acts = [np.sum(masked_act, axis=1) for masked_act in masked_acts]
                    
                    # Calculate distances
                    _, result_matrix = metric.pairwise_distances(pooled_acts, pooled_acts, processes=cpus)
                    
                    # No adjustment needed
                    adjustment = 0.0
                    
                except Exception as attn_err:
                    print(f"Error processing transformer layer {layer_name}: {attn_err}")
                    traceback.print_exc()
                    
                    # Fallback: try processing as a normal layer
                    print("Falling back to normal layer processing")
                    acts_flat = [act.reshape(act.shape[0], -1) for act in layer_activations]
                    _, result_matrix = metric.pairwise_distances(acts_flat, acts_flat, processes=cpus)
                    adjustment = 0.0  # Use the adjustment from FC layers
            
            elif is_embedding_layer and len(act_shape) == 3 and mask is not None:
                # Handle 3D embedding with mask - like transformer layers
                try:
                    # Ensure mask shape compatibility
                    n_batch, seq_len = act_shape[0], act_shape[1]
                    
                    if mask.shape[0] < n_batch or mask.shape[1] < seq_len:
                        print(f"Resizing mask from {mask.shape} to match 3D embedding {(n_batch, seq_len)}")
                        repeated_mask = np.repeat(mask[:1], n_batch, axis=0)
                        mask_aligned = repeated_mask[:, :seq_len]
                    else:
                        mask_aligned = mask[:n_batch, :seq_len]
                    
                    # Apply mask and pool
                    mask_expanded = np.expand_dims(mask_aligned, axis=-1)
                    masked_acts = [act * mask_expanded for act in layer_activations]
                    
                    # Sum over sequence dimension as in old code
                    pooled_acts = [np.sum(masked_act, axis=1) for masked_act in masked_acts]
                    
                    # Calculate distances
                    _, result_matrix = metric.pairwise_distances(pooled_acts, pooled_acts, processes=cpus)
                    
                    # No adjustment needed
                    adjustment = 0.0
                    print(f"Processed {layer_name} as 3D embedding with masking")
                    
                except Exception as emb_err:
                    print(f"Error processing 3D embedding layer {layer_name}: {emb_err}")
                    traceback.print_exc()
                    
                    # Fallback: process as 2D by flattening
                    print("Falling back to flattened processing for 3D embedding")
                    acts_flat = [act.reshape(act.shape[0], -1) for act in layer_activations]
                    _, result_matrix = metric.pairwise_distances(acts_flat, acts_flat, processes=cpus)
                    adjustment = 0.0  # Use the adjustment from FC layers
                
            elif is_embedding_layer and len(act_shape) == 2:
                # Handle 2D embedding without mask - process like any other 2D activation
                print(f"Processing {layer_name} as 2D embedding layer")
                # No need to reshape if already 2D (samples, features)
                _, result_matrix = metric.pairwise_distances(layer_activations, layer_activations, processes=cpus)
                adjustment = 0.0  # Same as FC layers
            
            else:  # Default for FC layers or others
                # Reshape to (samples, features)
                acts_flat = [act.reshape(act.shape[0], -1) for act in layer_activations]
                
                # Calculate distances
                _, result_matrix = metric.pairwise_distances(acts_flat, acts_flat, processes=cpus)
                
                # Add adjustment as in old code
                adjustment = 0.0

            # Apply adjustment factor if non-zero
            if adjustment != 0.0:
                result_matrix += adjustment
                print(f"Applied adjustment factor: {adjustment}")

            # Create DataFrame with proper site_id indexing
            result_df = pd.DataFrame(result_matrix, 
                                    index=valid_site_ids_for_layer, 
                                    columns=valid_site_ids_for_layer)
            
            comparison_data[layer_name] = result_df
            print(f"Successfully processed layer {layer_name}")

        except Exception as e:
            print(f"Error calculating similarity for layer {layer_name}: {e}")
            traceback.print_exc()
            # Create a DataFrame of zeros instead of NaNs for errors
            result_df = pd.DataFrame(np.zeros((current_num_sites, current_num_sites)),
                                    index=valid_site_ids_for_layer, 
                                    columns=valid_site_ids_for_layer)
            comparison_data[layer_name] = result_df
            print(f"Created fallback zero matrix for layer {layer_name}")

        finally:
            # Clean up intermediate variables
            if 'acts_nhwc' in locals(): del acts_nhwc
            if 'masked_acts' in locals(): del masked_acts
            if 'pooled_acts' in locals(): del pooled_acts
            if 'acts_flat' in locals(): del acts_flat
            del layer_activations
            gc.collect()

    # Final reporting
    processed_layers = list(comparison_data.keys())
    print(f"Completed processing {len(processed_layers)} of {len(layer_names)} layers")
    
    return comparison_data