#!/usr/bin/env python
# coding: utf-8

"""
Script to load, process, and visualize results from the Federated Learning Analytics Pipeline.
Loads data saved by analytics_pipeline.py, averages metrics across runs,
groups/renames layers, and generates plots for gradient/Hessian metrics and
activation similarity.
"""

import pandas as pd
import numpy as np
import pickle
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
import sys
from pandas.api.types import is_numeric_dtype # Import for checking numeric types



warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
plt.style.use('seaborn-v0_8-whitegrid') # Example style
stages = ['first', 'final']
# --- Data Loading ---

def load_analytics_data(dataset: str, results_dir: str) -> dict:
    """Loads the raw analytics results dictionary from the pickle file."""
    analytics_dir = os.path.join(results_dir, 'analytics')
    results_path = os.path.join(analytics_dir, f'{dataset}_analytics_results.pkl')

    if not os.path.exists(results_path):
        print(f"Error: Analytics results file not found at {results_path}")
        return {}

    try:
        with open(results_path, 'rb') as f:
            raw_results = pickle.load(f)
        print(f"Loaded analytics results from {results_path}")
        return raw_results if isinstance(raw_results, dict) else {}
    except (EOFError, pickle.UnpicklingError):
        print(f"Warning: Analytics results file {results_path} is empty or corrupted.")
        return {}
    except Exception as e:
        print(f"Error loading analytics results from {results_path}: {e}")
        return {}

# --- Data Processing ---

def average_results_across_runs(raw_results: dict, server_type: str) -> dict:
    """
    Processes raw analytics results for a specific server_type,
    averages across runs, and restructures for plotting functions.
    """
    if not raw_results: return {}

    collected_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    client_ids_found = set()
    num_valid_runs = 0

    for run_name, run_data in raw_results.items():
        if not run_name.startswith('run_'): continue
        if server_type not in run_data or 'status' in run_data[server_type] and run_data[server_type]['status'] == 'failed':
            print(f"Skipping run '{run_name}' for server '{server_type}' due to missing data or previous failure.")
            continue

        server_data = run_data[server_type]
        run_has_data = False # Flag to check if this run contributed any data

        for stage in ['first', 'final']:
            # Process Grad/Hess metrics
            if 'grad_hess' in server_data and stage in server_data['grad_hess']:
                for client_id, df in server_data['grad_hess'][stage].items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        collected_data[stage]['grad_hess'][client_id].append(df)
                        client_ids_found.add(client_id)
                        run_has_data = True

            # Process Similarity metrics
            if 'similarity' in server_data and stage in server_data['similarity']:
                 for layer_name, df in server_data['similarity'][stage].items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                         collected_data[stage]['similarity'][layer_name].append(df)
                         run_has_data = True # Similarity data also counts

        if run_has_data:
             num_valid_runs += 1 # Only count runs that provided some processable data

    if num_valid_runs == 0:
         print(f"No valid run data found for server type '{server_type}' after checking all runs.")
         return {}
    print(f"Processing data for server '{server_type}' across {num_valid_runs} valid runs.")

    sorted_client_ids = sorted(list(client_ids_found))
    client_to_site_map = {client_id: i for i, client_id in enumerate(sorted_client_ids)}
    if client_ids_found: # Print only if clients were found
         print(f"Client ID to Site Integer Mapping: {client_to_site_map}")

    averaged_results = defaultdict(lambda: defaultdict(dict)) # Use defaultdict for stages and metric types

    for stage, stage_data in collected_data.items():
        for metric_type, key_data in stage_data.items():
            for key, df_list in key_data.items():
                if not df_list: continue

                try:
                    # Ensure all dataframes in list are valid before concat
                    valid_dfs = [df for df in df_list if isinstance(df, pd.DataFrame) and not df.empty]
                    if not valid_dfs: continue

                    mean_df = pd.concat(valid_dfs).groupby(level=0).mean()
                    mean_df.index = [str(i).replace('.weight', '') for i in mean_df.index]

                    output_key = client_to_site_map.get(key) if metric_type == 'grad_hess' else key
                    if output_key is not None:
                         averaged_results[stage][metric_type][output_key] = mean_df

                except ValueError as ve:
                    print(f"ValueError during averaging for {stage}/{metric_type}/{key}: {ve}. Check DataFrame compatibility. Skipping.")
                except Exception as e:
                    print(f"Error averaging data for {stage}/{metric_type}/{key}: {e}. Skipping.")

    return json_compatible_defaultdict(averaged_results) # Convert back to regular dicts

def json_compatible_defaultdict(d):
    """Recursively converts defaultdict to dict."""
    if isinstance(d, defaultdict):
        d = {k: json_compatible_defaultdict(v) for k, v in d.items()}
    return d

# --- Layer Processing Helpers ---

def extract_numerical_index(layer_name: str) -> int:
    """Extracts the first sequence of digits found in a layer name."""
    match = re.search(r'\d+', layer_name)
    return int(match.group()) if match else -1

def get_max_layer_number(index: pd.Index) -> int:
    """Determines the maximum numerical index found across various layer naming patterns."""
    max_num = -1
    for layer_name in index:
        num = extract_numerical_index(str(layer_name))
        if num > max_num:
            max_num = num
    return max_num

# --- Helper function to extract max number (similar to old code's logic) ---
def extract_max_layer_num(index):
    """
    Extracts the maximum numerical index found in layer names based on common patterns.
    More robustly handles various potential layer naming schemes.
    """
    max_num = -1
    patterns = [
        r'layer(\d+)',            # Matches layerX
        r'attention(\d+)',        # Matches attentionX
        r'attn(\d+)',             # Matches attnX
        r'embedding(\d+)',        # Matches embeddingX (if numbered)
        r'token_embedding_table(\d+)', # Specific embedding tables
        r'position_embedding_table(\d+)',# Specific embedding tables
        r'proj(\d+)',             # Matches projX
        r'resid(\d+)',            # Matches residX
        r'fc(\d+)',               # Matches fcX
        r'linear(\d+)',           # Matches linearX
        r'conv(\d+)',             # Matches convX
    ]

    for layer_name in index:
        layer_name_str = str(layer_name)
        current_max_in_name = -1
        for pattern in patterns:
            matches = re.findall(pattern, layer_name_str)
            if matches:
                # Find the maximum number found by this pattern in this layer name
                try:
                    # Use max on the extracted numbers converted to int
                    local_max = max(int(m) for m in matches)
                    current_max_in_name = max(current_max_in_name, local_max)
                except ValueError:
                    pass # Ignore if conversion fails
        max_num = max(max_num, current_max_in_name) # Update overall max

    return max_num if max_num != -1 else 0 # Return 0 if no numbers found, avoids range issues

# --- MODIFIED group_and_rename_metrics_layers ---
def group_and_rename_metrics_layers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups layers based on common prefixes and types using explicit mapping rules
    similar to the old code, aggregates metrics (mean/sum), and renames/reorders
    them for plotting consistency.
    """
    if df.empty:
        print("Warning: Input DataFrame is empty in group_and_rename_metrics_layers.")
        return df

    original_indices = df.index.astype(str)
    original_indices_series = pd.Series(original_indices, index=original_indices) # Create Series for alignment

    # 1. Determine max_num based on layer indices
    max_num = extract_max_layer_num(original_indices)
    # print(f"Determined max_num for mapping: {max_num}") # Reduce verbosity

    # 2. Build the explicit mapping dictionary (combining old code logic)
    mapping = {}
    # ... (mapping dictionary creation code remains the same) ...
    if max_num > 0:
        for i in range(1, max_num + 1):
            # Embeddings
            mapping[f'token_embedding_table{i}'] = f'embedding{i}'
            mapping[f'position_embedding_table{i}'] = f'embedding{i}'
            # Attention related (map proj to attention)
            mapping[f'proj{i}.0'] = f'attention{i}'
            mapping.update({f'attention{i}.{sub}': f'attention{i}' for sub in ['query', 'key', 'value', 'out_proj', 'wq', 'wk', 'wv', 'wo']}) # More robust attn sublayers
            mapping.update({f'attn{i}.{sub}': f'attention{i}' for sub in ['query', 'key', 'value', 'out_proj', 'wq', 'wk', 'wv', 'wo']}) # Handle 'attn' prefix too
            # FC related (map resid to fc)
            mapping[f'resid{i}.0'] = f'fc{i}.0'
            # Handle potential numbered FC layers directly
            mapping[f'fc{i}'] = f'fc{i}.0'
            mapping[f'linear{i}'] = f'fc{i}.0' # Map numbered linear to fc

    conv_pattern = re.compile(r'(?:layer|conv)(\d+)[._]c[._](\d+)') # Match layerX_c_Y or convX_c_Y or layerX.c.Y etc.
    for layer_name in original_indices:
        match = conv_pattern.match(layer_name)
        if match:
            layer_num = match.group(1)
            mapping[layer_name] = f'conv{layer_num}'
        if 'token_embedding' in layer_name and layer_name not in mapping: mapping[layer_name] = 'embedding0' # Fallback name if not numbered
        if 'position_embedding' in layer_name and layer_name not in mapping: mapping[layer_name] = 'embedding0'
        if ('attention' in layer_name or 'attn' in layer_name) and layer_name not in mapping: mapping[layer_name] = 'attention0'
     
    # ... (mapping dictionary creation code continues) ...


    # 3. Apply the mapping to get target group names for each original index
    # FIX: Use .where() to fill NaNs with original values
    mapped_series = original_indices_series.map(mapping) # Map original indices (as Series)
    target_group_names = mapped_series.where(pd.notna(mapped_series), original_indices_series)
    # target_group_names is now a Series where index=original_name, value=group_name

    # 4. Define aggregation logic
    # ... (agg_funcs creation code remains the same) ...
    agg_funcs = {}
    metrics_to_sum = ['Gradient Importance', 'Gradient Importance 2']
    for col in df.columns:
        if col in metrics_to_sum and is_numeric_dtype(df[col]):
            agg_funcs[col] = 'sum'
        elif is_numeric_dtype(df[col]):
            agg_funcs[col] = 'mean'
    if not agg_funcs:
         print("Warning: No numeric columns found to aggregate in group_and_rename_metrics_layers.")
         # Use the unique group names as the index for the empty DataFrame
         return pd.DataFrame(index=target_group_names.unique())

    # ...

    # 5. Group by the target names and aggregate
    try:
        # Assign target names temporarily for groupby
        df_temp = df.copy()
        # Use the pandas Series target_group_names directly for grouping
        # This aligns based on the index (which matches df_temp's index)
        df_grouped = df_temp.groupby(target_group_names).agg(agg_funcs)
    except Exception as e:
         print(f"Error during groupby/aggregation: {e}. Returning partially processed data if possible.")
         return pd.DataFrame()

    # 6. Reorder the final grouped DataFrame
    # ... (Reordering logic remains the same) ...
    layer_order_prefixes = ['embedding', 'conv', 'layer', 'attention', 'attn',
                           'proj', 'resid', 'fc' , 'fc_final']
    ordered_index = []
    processed = set()
    for prefix in layer_order_prefixes:
        matching_layers = []
        for layer_name in df_grouped.index:
            if str(layer_name).startswith(prefix) and layer_name not in processed:
                 matching_layers.append(layer_name)
                 processed.add(layer_name)
        def sort_key(layer_name):
            match = re.search(r'\d+$', str(layer_name)) # Look for digits at the end
            return int(match.group()) if match else float('inf')
        try:
            matching_layers_sorted = sorted(matching_layers, key=sort_key)
        except Exception:
             matching_layers_sorted = sorted(matching_layers)
        ordered_index.extend(matching_layers_sorted)
    remaining_layers = sorted([layer for layer in df_grouped.index if layer not in processed])
    ordered_index.extend(remaining_layers)
    try:
        if set(ordered_index) == set(df_grouped.index):
            df_ordered = df_grouped.reindex(ordered_index)
        else:
            print(f"Warning: Layer reordering mismatch. Index lengths: {len(ordered_index)} vs {len(df_grouped.index)}. Sets equal: {set(ordered_index) == set(df_grouped.index)}. Returning original group order.")
            df_ordered = df_grouped
    except Exception as e:
        print(f"Error during final reindexing: {e}. Returning original group order.")
        df_ordered = df_grouped
    # ...

    return df_ordered



# --- process_metrics_for_plotting using the modified function ---
def process_metrics_for_plotting(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """ Processes metrics DataFrame: groups layers using the new logic."""
    # max_layer_num calculation is now inside group_and_rename...
    # We can still call get_max_layer_number for info if needed, but it's not required by the grouping anymore
    max_layer_num_info = get_max_layer_number(df.index) # Still useful info maybe?
    df_processed = group_and_rename_metrics_layers(df)
    return df_processed, max_layer_num_info # Return the processed df and max_num info


# --- Plotting Functions ---

def plot_model_metrics(dataset, averaged_results, server_type, results_dir):
    """Plot the model metrics using averaged results."""
    metrics_graph = ['Gradient Importance per', 'Gradient Variance', 'SVD Sum EV']
    

    if not averaged_results:
        print(f"No averaged results available for {dataset}, server {server_type}.")
        return

    # Determine number of rows/cols for subplots
    n_rows = len(metrics_graph)
    n_cols = len(stages)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 7), sharex='col') # Adjusted size, sharex
    # Ensure axes is always 2D array even if n_rows or n_cols is 1
    if n_rows == 1 and n_cols == 1: axes = np.array([[axes]])
    elif n_rows == 1: axes = np.array([axes])
    elif n_cols == 1: axes = axes.reshape(-1, 1)

    handles_dict = {}

    for m_idx, metric in enumerate(metrics_graph):
        for stage_idx, stage in enumerate(stages):
            ax = axes[m_idx][stage_idx]
            metrics_data_for_stage = averaged_results.get(stage, {}).get('grad_hess', {})

            # --- Plot and save individual graph FIRST ---
            try:
                individual_fig = plot_graph(metric, stage, metrics_data_for_stage, dataset)
                save_plot(individual_fig, dataset, stage, metric, server_type, results_dir)
            except Exception as e:
                 print(f"Error generating/saving individual plot for {metric}, {stage}: {e}")

            # --- Add to composite plot ---
            if not metrics_data_for_stage:
                 ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
                 ax.set_title(f'{metric} - {stage.capitalize()}', fontsize=16, fontweight='bold')
                 continue

            has_site_data = False
            for site, df_site_stage in metrics_data_for_stage.items():
                if not isinstance(df_site_stage, pd.DataFrame) or df_site_stage.empty: continue
                if dataset == 'Sentiment' and site in [0, 3]: continue
                if metric not in df_site_stage.columns: continue

                df_processed, _ = process_metrics_for_plotting(df_site_stage)
                if df_processed.empty or metric not in df_processed.columns: continue

                metric_series = df_processed[metric].copy()
                metric_series = pd.to_numeric(metric_series, errors='coerce').dropna()
                if metric_series.empty: continue

                norm_factor = metric_series.iloc[0] if len(metric_series) > 0 and metric_series.iloc[0] != 0 else 1.0
                normalized_metric = metric_series / norm_factor
                if metric == 'Gradient Importance per':
                    normalized_metric = normalized_metric.cumsum()

                line, = ax.plot(normalized_metric.index, normalized_metric.values, label=f'Site {site}', linewidth=3, marker='o', markersize=6)
                has_site_data = True
                if f'Site {site}' not in handles_dict:
                     handles_dict[f'Site {site}'] = line

            if has_site_data:
                ax.set_yscale('log')
                ax.tick_params(axis='x', rotation=45, labelsize=12)
                ax.tick_params(axis='y', labelsize=12)
                ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
                ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgrey')

            else:
                ax.text(0.5, 0.5, 'No valid site data', ha='center', va='center', fontsize=12)

            ax.set_title(f'{metric} - {stage.capitalize()}', fontsize=16, fontweight='bold')
            if m_idx == n_rows - 1:
                ax.set_xlabel('Layer Group / Name', fontsize=14)
            if stage_idx == 0:
                 naming_dict = {'Gradient Importance per': 'Cumul. Importance',
                                'Gradient Variance': 'Grad Variance',
                                'SVD Sum EV': 'Sum SVD(H*v)',
                                }
                 ax.set_ylabel(naming_dict.get(metric, metric), fontsize=14)

    if handles_dict:
         fig.legend(handles_dict.values(), handles_dict.keys(), loc='center right', fontsize=14, title="Sites", bbox_to_anchor=(1.03, 0.5)) # Adjust anchor

    fig.suptitle(f'{dataset} - {server_type.upper()} - Grad/Hess Metrics', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 0.96]) # Adjust rect to make space
    plt.show() # Show composite plot
    # --- Save Composite Plot ---
    composite_plot_dir = os.path.join(results_dir, 'plots', dataset, server_type)
    os.makedirs(composite_plot_dir, exist_ok=True)
    composite_filename = os.path.join(composite_plot_dir, f"{dataset}_{server_type}_grad_hess_composite.pdf")
    try:
        fig.savefig(composite_filename, bbox_inches='tight', dpi=300)
        print(f"Saved composite grad/hess plot: {composite_filename}")
    except Exception as e:
        print(f"Error saving composite grad/hess plot {composite_filename}: {e}")
    plt.close(fig) # Close the figure after showing/saving


def save_plot(fig, dataset, stage, metric, server_type, results_dir):
    """ Save plot in results individually, adding server_type."""
    naming_dict = {'Gradient Importance per':'Layer_importance',
                   'Gradient Variance':'Gradient_Variance',
                   'SVD Sum EV':'Hessian_EV_sum',
                   }
    m_title = naming_dict.get(metric, metric.replace(' ', '_').replace('/', '_')) # Sanitize name
    plot_dir = os.path.join(results_dir, 'plots', dataset, server_type) # Organize plots
    os.makedirs(plot_dir, exist_ok=True)
    filename = os.path.join(plot_dir, f"{dataset}_{server_type}_{m_title}_{stage}.pdf")
    try:
        fig.savefig(filename, bbox_inches='tight', dpi=300) # Add dpi
        print(f"Saved plot: {filename}")
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")
    plt.close(fig)


def process_model_similarity_data(similarity_data_dict):
    """
    Processes the model similarity data (averaged across runs) for plotting.
    Input dict: {layer_name: mean_df_across_runs}
    Output: DataFrame where index=Layer, columns=Site, value=Avg Dissimilarity for that site.
    
    This version properly averages sublayers within the same layer rather than summing them.
    - Improved to handle layer naming patterns like 'layer3.1', 'layer3.2', etc.
    - Maps 'flatten' layers to 'fc1'
    - Generally groups by name before any dot
    """
    if not similarity_data_dict: 
        return pd.DataFrame()
    
    combined_data = pd.DataFrame()
    
    # Group layers by their type and layer number
    layer_groups = defaultdict(list)
    
    # Extract max layer number to help with grouping
    max_num = extract_max_layer_num(list(similarity_data_dict.keys()))
    
    # Define mapping for various layer types
    layer_mapping = {}
    
    # Create mappings for embedding layers and others based on max_num
    for i in range(1, max_num+1):
        layer_mapping[f'token_embedding_table{i}'] = f'embedding{i}'
        layer_mapping[f'position_embedding_table{i}'] = f'embedding{i}'
        layer_mapping[f'proj{i}'] = f'attention{i}'
        layer_mapping[f'resid{i}'] = f'fc{i}'
        
        # Add attention sublayers
        for sub_layer in ['query', 'key', 'value']:
            layer_mapping[f'attention{i}.{sub_layer}'] = f'attention{i}'
    
    # Special mapping for flatten layers (to fc1)
    for layer_name in similarity_data_dict.keys():
        if 'flatten' in layer_name.lower():
            pass #layer_mapping[layer_name] = 'fc1'
    
    # Add convolutional sublayers and handle dot-notation layers
    for layer_name in similarity_data_dict.keys():
        # First check if it's already in mapping
        if layer_name in layer_mapping:
            continue
            
        # Pattern 1: layer + number + _c_ + number (conv layers)
        conv_pattern = re.compile(r'layer(\d+)[._]c[._](\d+)')
        match = conv_pattern.match(layer_name)
        if match:
            layer_num = match.group(1)
            layer_mapping[layer_name] = f'conv_layer{layer_num}'
            continue
        
        # Split on dot and use the first part for grouping
        if '.' in layer_name:
            base_name = layer_name.split('.')[0]
            # Special case for fc1/fc2 with numbers
            if base_name.startswith('fc'):
                layer_mapping[layer_name] = base_name
            # For ResNet-style layer3.1, etc.
            elif re.match(r'layer\d+', base_name) or re.match(r'conv\d+', base_name):
                layer_mapping[layer_name] = base_name
            continue
    
    # Print mappings for debugging (limited to avoid excessive output)
    mapping_examples = list(layer_mapping.items())[:5]
    print(f"Created {len(layer_mapping)} layer mappings. Examples: {mapping_examples}...")
    
    # Apply the mapping and do base name grouping for unmapped layers
    for layer_name, similarity_df in similarity_data_dict.items():
        # Skip if not a valid DataFrame
        if not isinstance(similarity_df, pd.DataFrame) or similarity_df.empty:
            print(f"Warning: Skipping invalid similarity matrix for layer '{layer_name}'")
            continue
        
        # Determine the group name
        if layer_name in layer_mapping:
            # Use predefined mapping if available
            group_name = layer_mapping[layer_name]
        elif '.' in layer_name:
            # If no mapping but has a dot, use part before dot
            group_name = layer_name.split('.')[0]
        else:
            # Use the whole name as is
            group_name = layer_name
        
        # Add to appropriate group
        layer_groups[group_name].append((layer_name, similarity_df))
    
    # Process each group - AVERAGE the similarity matrices (not sum)
    processed_similarity = {}
    for group_name, layer_dfs in layer_groups.items():
        if not layer_dfs:
            continue
            
        # Check if all DFs have same shape
        shapes = set(df.shape for _, df in layer_dfs)
        if len(shapes) > 1:
            print(f"Warning: Inconsistent shapes in group '{group_name}'. Using first compatible set.")
            # Get most common shape
            common_shape = max(shapes, key=lambda s: sum(1 for _, df in layer_dfs if df.shape == s))
            # Filter DFs with that shape
            layer_dfs = [(name, df) for name, df in layer_dfs if df.shape == common_shape]
            
        if not layer_dfs:
            continue
            
        # Average the similarity matrices in this group
        dfs_to_average = [df for _, df in layer_dfs]
        
        # Convert all indices and columns to strings for consistent joining
        for i, df in enumerate(dfs_to_average):
            dfs_to_average[i] = df.copy()
            dfs_to_average[i].index = dfs_to_average[i].index.astype(str)
            dfs_to_average[i].columns = dfs_to_average[i].columns.astype(str)
        
        try:
            # Stack all DFs and take mean by group
            stacked_df = pd.concat(dfs_to_average)
            # Group by index/columns and average
            avg_df = stacked_df.groupby(level=0).mean()
            # Convert indices back to original type if possible
            try:
                avg_df.index = avg_df.index.astype(int)
                avg_df.columns = avg_df.columns.astype(int)
            except ValueError:
                # Keep as strings if conversion fails
                pass
                
            processed_similarity[group_name] = avg_df
        except Exception as e:
            print(f"Error averaging similarity for group '{group_name}': {e}")
            # Fallback to first DF
            processed_similarity[group_name] = layer_dfs[0][1]
    
    # Calculate average dissimilarity per site for each grouped layer
    for layer, df in processed_similarity.items():
        avg_dissimilarity = {}
        
        for col in df.columns:
            # Calculate mean dissimilarity to other sites (exclude self-comparison)
            other_sites_dissimilarity = df.loc[df.index != col, col]
            
            if not other_sites_dissimilarity.empty:
                avg_dissimilarity[col] = other_sites_dissimilarity.mean()
            else:
                avg_dissimilarity[col] = 0.0
                
        combined_data[layer] = pd.Series(avg_dissimilarity)
    
    # Order layers in a logical sequence
    if not combined_data.empty:
        try:
            # Define layer order prefixes with a specific order
            layer_order_prefixes = ['embedding', 'conv', 'layer', 'attention', 'attn', 
                                    'flatten', 'proj', 'resid', 'fc', 'linear', 'output', 'classifier']
            
            # Categorize layers by prefix
            layer_dict = {prefix: [] for prefix in layer_order_prefixes}
            other_layers = []
            
            # Sort layers into their categories
            for layer in combined_data.columns:
                found = False
                for prefix in layer_order_prefixes:
                    if str(layer).startswith(prefix):
                        layer_dict[prefix].append(layer)
                        found = True
                        break
                        
                if not found:
                    other_layers.append(layer)
            
            # Sort layers within each category - try to extract numbers for proper ordering
            for prefix in layer_dict:
                try:
                    # Sort numerically if possible (extract digits after the prefix)
                    layer_dict[prefix] = sorted(
                        layer_dict[prefix], 
                        key=lambda x: int(re.search(r'\d+', str(x)).group()) if re.search(r'\d+', str(x)) else float('inf')
                    )
                except Exception as e:
                    print(f"Error sorting layers for prefix '{prefix}': {e}")
                    # Fall back to lexicographical sort
                    layer_dict[prefix] = sorted(layer_dict[prefix])
            
            # Build the final ordered list
            custom_order = []
            for prefix in layer_order_prefixes:
                custom_order.extend(layer_dict[prefix])
            custom_order.extend(sorted(other_layers))
            
            # Reorder the columns if all layers are accounted for
            if set(custom_order) == set(combined_data.columns):
                combined_data = combined_data[custom_order]
            else:
                print("Warning: Custom similarity layer order mismatch.")
                
        except Exception as e:
            print(f"Warning: Error during similarity layer reordering: {e}")
    
    return combined_data


# --- Plotting Functions ---
# ... (plot_graph, save_plot, plot_model_metrics remain the same) ...
# ... (process_model_similarity_data remains the same) ...


# MODIFIED plot_similarity_graph to optionally plot on existing axes
def plot_similarity_graph(df, stage, dataset, server_type, ax=None):
    """
    Plots a similarity graph for a given stage, dataset, and server_type.
    If ax is provided, plots on that Axes object, otherwise creates a new figure.
    """
    if ax is None:
        # Create new figure ONLY if no axes are provided (for individual plots)
        fig, ax = plt.subplots(figsize=(10, 8))
        standalone_fig = True
    else:
        # Use the provided axes (for composite plot)
        fig = ax.get_figure() # Get the figure the axes belong to
        standalone_fig = False

    handles = []
    labels = []
    plot_title = f"{stage.capitalize()} Similarity" # Default title for subplot

    if not isinstance(df, pd.DataFrame) or df.empty:
        print(f"Warning: Empty similarity DataFrame for stage '{stage}'. Cannot plot.")
        ax.text(0.5, 0.5, 'No similarity data', ha='center', va='center', fontsize=16)
        ax.set_title(plot_title, fontsize=24, fontweight='bold') # Set title even if empty
    else:
        # Transpose so layers are x-axis, sites are lines
        df_plot = df.T
        num_sites = df_plot.shape[1]

        for i in range(num_sites):
            site_label = df_plot.columns[i] # Should be 0, 1, 2...
            # OLD STYLE: Line width, marker size
            sns.lineplot(x=df_plot.index, y=df_plot.iloc[:, i], ax=ax, linewidth=4, label=f'Site {site_label}', marker='o', markersize=10, dashes=False)
            # Append the last line added to the specific axes
            handles.append(ax.lines[-1])
            labels.append(f'Site {site_label}')

        # OLD STYLE: Labels, fonts, ticks
        ax.set_ylabel("Dissimilarity", fontsize=24)
        ax.set_xlabel("Layer", fontsize=24)
        ax.set_xticklabels([]) # Hide x-tick labels
        ax.tick_params(axis='y', labelsize=22)

        # Legend is now handled by the calling function for composite plot
        if standalone_fig:
            if dataset == 'Sentiment':
                ax.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=22)
            else:
                ax.legend(handles=handles, labels=labels, fontsize=22)

        # Set title for the subplot
        ax.set_title(plot_title, fontsize=24, fontweight='bold')

    # Only apply layout adjustments if it's a standalone figure
    if standalone_fig:
        # fig.suptitle(f"{dataset} - {server_type.upper()} Similarity", fontsize=24) # Optional main title
        plt.tight_layout()
        if dataset == 'Sentiment':
            plt.subplots_adjust(right=0.75)

    # Return the main figure object (needed by the composite function)
    # and handles/labels (useful if composite needs a shared legend)
    return fig, handles, labels


def save_plot_sim(fig, dataset, stage, server_type, results_dir):
    """ Save similarity plot in results individually. """
    plot_dir = os.path.join(results_dir, 'plots', dataset, server_type)
    os.makedirs(plot_dir, exist_ok=True)
    filename = os.path.join(plot_dir, f"{dataset}_{server_type}_similarity_{stage}.pdf")
    try:
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Saved similarity plot: {filename}")
    except Exception as e:
        print(f"Error saving similarity plot {filename}: {e}")



def plot_model_similarity(dataset, averaged_results, server_type, results_dir):
    """
    Plot and save model similarity metrics individually AND as a composite plot.
    """
    if not averaged_results:
         print(f"No averaged results for {dataset}, server {server_type}.")
         return

    # --- Create Figure for Composite Plot ---
    n_cols = len(stages)
    composite_fig, composite_axes = plt.subplots(1, n_cols, figsize=(n_cols * 10 + 2, 8), sharey=True)
    if n_cols == 1: composite_axes = np.array([composite_axes])

    all_handles = []
    all_labels = []

    for stage_idx, stage in enumerate(stages):
        ax = composite_axes[stage_idx]
        similarity_data_for_stage = averaged_results.get(stage, {}).get('similarity', {})

        if not similarity_data_for_stage:
             print(f"No similarity data for stage '{stage}', server '{server_type}'.")
             ax.text(0.5, 0.5, 'No similarity data', ha='center', va='center', fontsize=16)
             ax.set_title(f"{stage.capitalize()} Similarity", fontsize=24, fontweight='bold')
             continue

        try:
            processed_sim_df = process_model_similarity_data(similarity_data_for_stage)
            if processed_sim_df.empty:
                 print(f"Similarity data processing yielded empty result for stage '{stage}'.")
                 ax.text(0.5, 0.5, 'Processing failed', ha='center', va='center', fontsize=16)
                 ax.set_title(f"{stage.capitalize()} Similarity", fontsize=24, fontweight='bold')
                 continue

            # --- Plot and Save Individual Figure ---
            individual_fig, _, _ = plot_similarity_graph(processed_sim_df, stage, dataset, server_type, ax=None)
            save_plot_sim(individual_fig, dataset, stage, server_type, results_dir)
            plt.close(individual_fig)

            # --- Plot on Composite Figure's Axes ---
            _, handles, labels = plot_similarity_graph(processed_sim_df, stage, dataset, server_type, ax=ax)

            # Collect unique handles/labels
            for handle, label in zip(handles, labels):
                if label not in all_labels:
                    all_labels.append(label)
                    all_handles.append(handle)

        except Exception as e:
            print(f"Error processing/plotting similarity for stage '{stage}', server '{server_type}': {e}")
            ax.text(0.5, 0.5, 'Plotting Error', ha='center', va='center', fontsize=16)
            ax.set_title(f"{stage.capitalize()} Similarity", fontsize=24, fontweight='bold')


    # --- Finalize Composite Plot ---
    if all_handles:
         # --- FIX: Robust sorting key ---
         def get_site_num_from_label(label):
             try:
                 # Assumes label is 'Site client_ID' or 'Site INDEX'
                 site_part = label.split(' ')[-1]
                 # Try splitting by '_' if it's like 'client_1', otherwise assume it's the number itself
                 num_str = site_part.split('_')[-1]
                 return int(num_str)
             except (IndexError, ValueError):
                 # Fallback if format is unexpected, sort alphabetically
                 print(f"Warning: Could not parse site number from label '{label}'. Sorting alphabetically.")
                 return label # Sort by the original label string as fallback

         label_handle_pairs = sorted(zip(all_labels, all_handles), key=lambda pair: get_site_num_from_label(pair[0]))
         # --- End Fix ---

         sorted_labels = [pair[0] for pair in label_handle_pairs]
         sorted_handles = [pair[1] for pair in label_handle_pairs]
         composite_fig.legend(sorted_handles, sorted_labels, loc='center right', fontsize=22, title="Sites")

    composite_fig.suptitle(f'{dataset} - {server_type.upper()} - Model Activation Similarity', fontsize=26, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    # --- Save Composite Plot ---
    composite_plot_dir = os.path.join(results_dir, 'plots', dataset, server_type)
    os.makedirs(composite_plot_dir, exist_ok=True)
    composite_filename = os.path.join(composite_plot_dir, f"{dataset}_{server_type}_similarity_composite.pdf")
    try:
        composite_fig.savefig(composite_filename, bbox_inches='tight', dpi=300)
        print(f"Saved composite similarity plot: {composite_filename}")
    except Exception as e:
        print(f"Error saving composite similarity plot {composite_filename}: {e}")



def plot_graph(metric, stage, metrics_data_for_stage, dataset):
    """
    Plots a particular grad/hess graph for all sites for a given stage.
    MODIFIED: Adjusted figure size, fonts, lines, markers, removed grid, hidden x-labels.
    """
    # OLD STYLE: Figure Size
    fig, ax = plt.subplots(figsize=(10, 8))
    has_data = False
    handles = []
    labels = []

    if not metrics_data_for_stage:
        print(f"Warning: No metrics data for stage '{stage}' to plot '{metric}'.")
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=16) # Keep informative text
    else:
        for site, df_site_stage in metrics_data_for_stage.items():
            if not isinstance(df_site_stage, pd.DataFrame) or df_site_stage.empty:
                # print(f"Warning: Invalid/empty DataFrame for site {site}, stage {stage}. Skipping.") # Reduce verbosity
                continue
            # Skip specific sites if needed (Keep this logic)
            if dataset == 'Sentiment' and site in [0, 3]: continue

            if metric not in df_site_stage.columns:
                 # print(f"Warning: Metric '{metric}' not found for site {site}, stage {stage}. Skipping.")
                 continue

            df_processed, _ = process_metrics_for_plotting(df_site_stage)

            if df_processed.empty or metric not in df_processed.columns:
                 # print(f"Warning: Metric '{metric}' not found after processing for site {site}, stage {stage}. Skipping.")
                 continue

            metric_series = df_processed[metric].copy()
            metric_series = pd.to_numeric(metric_series, errors='coerce').dropna()
            if metric_series.empty: continue

            # OLD STYLE: Normalize same way
            norm_factor = metric_series.iloc[0] if len(metric_series) > 0 and metric_series.iloc[0] != 0 else 1.0
            normalized_metric = metric_series / norm_factor

            if metric == 'Gradient Importance per':
                normalized_metric = normalized_metric.cumsum()

            # OLD STYLE: Line width and marker size
            line, = ax.plot(normalized_metric.index, normalized_metric.values, label=f'Site {site}', linewidth=4, marker='o', markersize=10)
            handles.append(line)
            labels.append(f'Site {site}')
            has_data = True

    if has_data:
        ax.set_yscale('log') # Keep log scale
        # OLD STYLE: Simpler naming, larger fonts
        naming_dict = {'Gradient Importance per': 'Layer Importance',
                       'Gradient Variance': 'Gradient Variance',
                       'SVD Sum EV': 'Hessian EV sum'} # Use old naming
        m_title = naming_dict.get(metric, metric)
        # ax.set_title(f'{dataset}, {m_title} - {stage.capitalize()}', fontsize=24, fontweight='bold') # Old title style (optional)
        ax.set_xlabel('Layer', fontsize=24) # Old label
        ax.set_ylabel(m_title, fontsize=24) # Old label
        # OLD STYLE: Hide x-tick labels and use larger font for y-ticks
        ax.set_xticklabels([])
        ax.tick_params(axis='y', labelsize=22)
        ax.legend(handles, labels, fontsize=22) # Old font size
        # OLD STYLE: Remove explicit grid (rely on base style)
        # ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
        # ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgrey')
    else:
         # Keep informative placeholder if no data
         # ax.set_title(f'{dataset}, {metric} - {stage.capitalize()}', fontsize=24, fontweight='bold')
         ax.set_xlabel('Layer', fontsize=24)
         ax.set_ylabel(metric, fontsize=24)
         ax.text(0.5, 0.5, 'No valid data to plot', ha='center', va='center', fontsize=16)


    plt.tight_layout()
    return fig

def process_results(dataset, results_dir, server_type):
    # Load the raw analytics results
    raw_analytics_results = load_analytics_data(dataset, results_dir)

    if not raw_analytics_results:
        print("Exiting: No analytics data loaded.")
        sys.exit(1)

    # Process and average results for the specified server type
    averaged_data = average_results_across_runs(raw_analytics_results, server_type)

    if not averaged_data:
        print(f"Exiting: No processed data available for server type '{server_type}'.")
        sys.exit(1)

    # Plot Gradient/Hessian Metrics
    print(f"\n--- Plotting Grad/Hess Metrics for {server_type.upper()} ---")
    plot_model_metrics(dataset, averaged_data, server_type, results_dir)

    # Plot Similarity Metrics (if applicable and data exists)
    if server_type == 'local' and averaged_data.get('first', {}).get('similarity') or averaged_data.get('final', {}).get('similarity'):
        print(f"\n--- Plotting Similarity Metrics for {server_type.upper()} ---")
        plot_model_similarity(dataset, averaged_data, server_type, results_dir)
    else:
        print(f"\n--- Skipping Similarity Plots (Not applicable for '{server_type}' or no data) ---")

    print(f"\n--- Analysis script finished for {dataset} / {server_type} ---")

