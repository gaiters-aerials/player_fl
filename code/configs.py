"""
Central configuration file for the Layer-PFL project.

Defines directory paths, constants, default hyperparameters,
algorithm-specific settings (like layers to federate, regularization parameters),
and imports common libraries used throughout the project.
"""
import warnings
# --- Suppress Warnings ---
# Ignore specific warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import os
import gc
import sys
import copy
import time
import json
import random
import logging
import pickle
import traceback
import argparse
from typing import List, Dict, Optional, Tuple, Union, Iterator, Iterable
from datetime import datetime
from functools import wraps
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from multiprocessing import Pool
import numpy as np
import pandas as pd
import scipy.stats
import scipy.stats as stats
from tqdm import tqdm  # Progress bars
from tqdm.contrib.concurrent import process_map  # Progress bars for parallel processing
from PIL import Image
from sklearn import metrics  # General metrics utilities
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score
from netrep.metrics import LinearMetric  # Representation similarity metrics
from netrep.conv_layers import convolve_metric  # Representation similarity metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.linalg import svdvals
from torch.utils.data import DataLoader, Dataset, Subset, RandomSampler
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp
from torchvision.transforms import transforms
from torchvision.datasets import FashionMNIST, EMNIST, CIFAR10
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
import albumentations  # For image augmentations

# --- Core Directories ---
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.dirname(_CURRENT_DIR)
ROOT_DIR = _PROJECT_ROOT
DATA_DIR = f'{ROOT_DIR}/data_2'       # Directory containing datasets
DATA_PROCESSING_DIR = f'{ROOT_DIR}/code/datasets' # Directory for data preproccessing
EVAL_DIR = f'{ROOT_DIR}/code/evaluation' # Directory for evaluation scripts/results
METRIC_DIR = f'{ROOT_DIR}/code/layer_metrics' # Directory for layer-specific metric code
RESULTS_DIR = f'{ROOT_DIR}/results_3' # Directory to save experiment results

# --- Add project directories to Python path ---
# Allows importing modules from these directories
sys.path.append(f'{ROOT_DIR}/code')
sys.path.append(f'{DATA_PROCESSING_DIR}')
sys.path.append(f'{EVAL_DIR}')
sys.path.append(f'{METRIC_DIR}')

# --- Global Settings ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # Set computation device
N_WORKERS = 4 # Default number of workers for DataLoader

# --- Supported Algorithms ---
ALGORITHMS = [
    'local',            # Local training only
    'fedavg',           # Federated Averaging
    'fedprox',          # FedProx (adds proximal term to local loss)
    'pfedme',           # pFedMe (personalized FedAvg with proximal term)
    'ditto',            # Ditto (dual model training: global and personal)
    'localadaptation',  # FedAvg followed by local fine-tuning
    'babu',             # FedAvg for body, local fine-tuning for head
    'fedlp',            # FedLP (layer-wise probabilistic aggregation)
    'fedlama',          # FedLAMA (layer-wise adaptive aggregation frequency)
    'pfedla',           # pFedLA (layer-wise personalized aggregation via hypernetwork)
    'playerfl',         # PLayerFL (FedAvg on a fixed subset of layers)
    'playerfl_random'   # PLayerFL (FedAvg on a randomly chosen prefix subset of layers)
]

# --- Supported Datasets ---
DATASETS = [
    'FMNIST',    # FashionMNIST
    'EMNIST',    # EMNIST
    'CIFAR',     # CIFAR-10
    'Sentiment', # Custom Sentiment Analysis dataset
    'ISIC',      # ISIC Skin Lesion Classification dataset
    'mimic',     # Custom MIMIC-III dataset (NLP)
    'Heart'      # Custom Heart Disease dataset (tabular)
]

# --- Data Heterogeneity Simulation ---
# Dirichlet distribution alpha parameter for simulating label distribution skew
# Lower alpha = higher heterogeneity
DATASET_ALPHA = {
    'EMNIST': 0.5,
    'CIFAR': 0.5,
    "FMNIST": 0.5
}

# --- Default Hyperparameters per Dataset ---
# These serve as starting points and ranges for tuning.
DEFAULT_PARAMS = {
    'FMNIST': {
        'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4], # LRs to try during tuning
        'learning_rate': 1e-3,                      # Default LR for layer metrics analysis
        'num_clients': 5,                           # Number of clients in federation
        'sizes_per_client': 2000,                   # Number of samples per client (if simulating)
        'classes': 10,                              # Number of output classes
        'batch_size': 128,                          # Training batch size
        'epochs_per_round': 1,                      # Local epochs per communication round
        'rounds': 100,                              # Total communication rounds
        'runs': 20,                                 # Number of independent runs for final evaluation
        'runs_lr': 3,                               # Number of independent runs for LR tuning
        'runs_layers':3                             # Number of independent runs for layer analytics
    },
    'EMNIST': {
        'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4],
        'learning_rate': 1e-3,
        'num_clients': 5,
        'sizes_per_client': 3000,
        'classes': 62,
        'batch_size': 128,
        'epochs_per_round': 1,
        'rounds': 75,
        'runs': 10,
        'runs_lr': 3,
        'runs_layers':3   
    },
    'CIFAR': {
        'learning_rates_try': [5e-3, 1e-3, 5e-4],
        'learning_rate': 1e-3,
        'num_clients': 5,
        'sizes_per_client': 10000,
        'classes': 10,
        'batch_size': 128,
        'epochs_per_round': 1,
        'rounds': 100,
        'runs': 10,
        'runs_lr': 3,
        'runs_layers':1   
    },
    'ISIC': {
        'learning_rates_try': [5e-3, 1e-3, 5e-4, 1e-4],
        'learning_rate': 1e-3,
        'num_clients': 4,
        'sizes_per_client': None, # Use actual client sizes
        'classes': 4,
        'batch_size': 128,
        'epochs_per_round': 1,
        'rounds': 60,
        'runs': 5,
        'runs_lr': 1,
        'runs_layers':1   
    },
    'Sentiment': {
        'learning_rates_try': [1e-3, 5e-4, 1e-4, 8e-5],
        'learning_rate': 1e-3,
        'num_clients': 15,
        'sizes_per_client': None, # Use actual client sizes
        'classes': 2,
        'batch_size': 64,
        'epochs_per_round': 1,
        'rounds': 50,
        'runs': 10,
        'runs_lr': 3,
        'runs_layers':3   
    },
    'Heart': {
        'learning_rates_try': [5e-1, 1e-1, 5e-2, 1e-2],
        'learning_rate': 5e-2,
        'num_clients': 4,
        'sizes_per_client': None, # Use actual client sizes
        'classes': 5,
        'batch_size': 32,
        'epochs_per_round': 1,
        'rounds': 20,
        'runs': 50,
        'runs_lr': 10,
        'runs_layers':5   
    },
    'mimic': {
        'learning_rates_try': [1e-3, 5e-4, 1e-4, 8e-5],
        'learning_rate': 1e-4,
        'num_clients': 4,
        'sizes_per_client': None, # Use actual client sizes
        'classes': 2,
        'batch_size': 64,
        'epochs_per_round': 1,
        'rounds': 25,
        'runs': 10,
        'runs_lr': 3,
        'runs_layers':3   
    }
}

# --- Layer Selection for Layer-wise FL Methods ---
# Defines which layers (by name prefix) are considered for federation
# in different layer-wise algorithms.
LAYERS_TO_FEDERATE_DICT = {
    # LayerPFL: Federate a fixed subset based on layer metrics
    "playerfl": {
        'EMNIST': ['layer1.', 'layer2.', 'layer3.', 'fc1'],
        'CIFAR': ['layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.'],
        "FMNIST": ['layer1.', 'layer2.', 'layer3.'],
        'ISIC': ['layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.', 'fc1'],
        "Sentiment": ['token_embedding_table1', 'position_embedding_table1', 'attention1', 'proj1', 'fc1', 'resid1'],
        "Heart": ['fc1'],
        "mimic": ['token_embedding_table1', 'position_embedding_table1', 'attention1', 'proj1']
    },
    # BABU: Federate all layers *except* the head
    "babu": {
        'EMNIST': ['layer1.', 'layer2.', 'layer3.'], 
        'CIFAR': ['layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.'], 
        "FMNIST": ['layer1.', 'layer2.', 'layer3.'], 
        'ISIC': ['layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.'],
        "Sentiment": ['token_embedding_table1', 'position_embedding_table1', 'attention1', 'proj1', 'fc1',  'resid1'], 
        "Heart": ['fc1', 'fc2', 'fc3'],
        "mimic": ['token_embedding_table1', 'position_embedding_table1', 'attention1', 'proj1', 'fc1',  'resid1']
    },
    # LayerPFL_random: Provides the *pool* of all possible layers to federate.
    # A random prefix subset of these layers will be chosen during runtime.
    "playerfl_random":{
            'EMNIST':['layer1.', 'layer2.', 'layer3.', 'fc1', 'fc2'],
            'CIFAR':['layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.', 'fc1', 'fc2'],
            "FMNIST":['layer1.', 'layer2.', 'layer3.', 'fc1', 'fc2'],
            'ISIC':['layer1.', 'layer2.', 'layer3.', 'layer4.', 'layer5.', 'fc1', 'fc2'],
            "Sentiment":['token_embedding_table1','position_embedding_table1', 'attention1', 'proj1', 'fc1', 'resid1', 'fc2'],
            "Heart": ['fc1', 'fc2', 'fc3', 'fc4'],
            "mimic":['token_embedding_table1','position_embedding_table1', 'attention1', 'proj1', 'fc1', 'resid1', 'fc2']
            },
            
}

# --- Regularization Parameters (mu or lambda) ---
# Used by FedProx, pFedMe, Ditto, potentially LayerPFL if modified
REG_PARAMS = {
    'fedprox': {
        'EMNIST': 0.1,
        'CIFAR': 0.1,
        "FMNIST": 0.1,
        'ISIC': 0.1,
        "Sentiment": 0.1,
        "Heart": 0.1,
        "mimic": 0.1
    },
    'pfedme': { # Often denoted as lambda in pFedMe paper
        'EMNIST': 0.1,
        'CIFAR': 0.1,
        "FMNIST": 0.1,
        'ISIC': 0.1,
        "Sentiment": 0.1,
        "Heart": 0.1,
        "mimic": 0.1
    },
    'ditto': { # Often denoted as lambda in Ditto paper
        'EMNIST': 0.1,
        'CIFAR': 0.1,
        "FMNIST": 0.1,
        'ISIC': 0.1,
        "Sentiment": 0.1,
        "Heart": 0.1,
        "mimic": 0.1
    },
    'layerpfl': { # Optional regularization for LayerPFL not used, but included option
        'EMNIST': 0.1,
        'CIFAR': 0.1,
        "FMNIST": 0.1,
        'ISIC': 0.1,
        "Sentiment": 0.1,
        "Heart": 0.1,
        "mimic": 0.1
    },
}

# --- FedLP Specific Parameter ---
# Probability 'p' that a layer's update is preserved/aggregated
LAYER_PRESERVATION_RATES = {
    'EMNIST': 0.7,
    'CIFAR': 0.7,
    "FMNIST": 0.7,
    'ISIC': 0.7,
    "Sentiment": 0.7,
    "Heart": 0.7,
    "mimic": 0.7
}

# --- FedLAMA Specific Parameters ---
# Base aggregation interval and Interval increase factor, defaults from paper taken
LAMA_RATES = {
    'tau_prime': 2,
    'phi': 2

}

# --- pFedLA HyperNetwork Parameters ---
# Configuration for the hypernetwork used in pFedLA
HYPERNETWORK_PARAMS = {
    'embedding_dim': { # Dimension of the client embedding vector
        'EMNIST': 32,
        'CIFAR': 64,
        "FMNIST": 32,
        'ISIC': 64,
        "Sentiment": 64,
        "Heart": 16,
        "mimic": 64
    },
    'hidden_dim': { # Dimension of the hidden layer in the per-layer MLPs
        'EMNIST': 64,
        'CIFAR': 128,
        "FMNIST": 64,
        'ISIC': 128,
        "Sentiment": 128,
        "Heart": 16,
        "mimic": 128
    },
    'hn_lr': { # Learning rate for updating the hypernetwork parameters
        'EMNIST': 0.01,
        'CIFAR': 0.01,
        "FMNIST": 0.01,
        'ISIC': 0.01,
        "Sentiment": 0.01,
        "Heart": 0.01,
        "mimic": 0.01
    }
}

# --- Datasets using Attention Models ---
# List of datasets that use transformer/attention-based models
# This might influence model loading or specific processing steps.
ATTENTION_MODELS = ['Sentiment', 'mimic']