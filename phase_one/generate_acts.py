# CREDIT: Marks & Tegemark (2023)
# The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets

# Imports

import torch as t  # PyTorch for tensor operations and saving .pt files

# Transformers imports (not all used directly, but available for model loading)
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer, 
    AutoTokenizer, 
    OPTForCausalLM, 
    GPTNeoXForCausalLM, 
    AutoModelForCausalLM
)

import argparse      # For parsing command-line arguments
import pandas as pd  # For reading CSV dataset files
from tqdm import tqdm  # Progress bar for loops
import os            # For file/directory operations
import sys
import configparser  # For reading config.ini file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# nnsight: Library for neural network interpretability
# Allows us to "hook" into model layers and extract activations
# Also provides remote execution capability through NDIF
from nnsight import LanguageModel, CONFIG
# If the key isn't set, remote execution will fail with an auth / model access error.
ndif_api_key = os.environ.get("NDIF_API_KEY") or os.environ.get("NNSIGHT_API_KEY")
if ndif_api_key:
    CONFIG.set_default_api_key(ndif_api_key)

# Config

# Debug mode: when True, nnsight will validate the tracing operations
# Set to False for production to avoid overhead
DEBUG = False
if DEBUG:
    # scan=True: checks that interventions are valid
    # validate=True: validates tensor shapes
    tracer_kwargs = {'scan': True, 'validate': True}
else:
    # Disable validation for faster execution
    tracer_kwargs = {'scan': False, 'validate': False}

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read(os.path.join(REPO_ROOT, 'config.ini'))

# Functions

def load_model(model_name, device='remote'):
    """
    Load a language model using nnsight's LanguageModel wrapper.
    
    Args:
        model_name: Key in config.ini (e.g., "llama-2-7b")
        device: Where to run the model
                - "remote": Use NDIF's remote GPUs (no local GPU needed)
                - "cuda:0": Use local NVIDIA GPU
                - "cpu": Use local CPU (slow)
    
    Returns:
        nnsight LanguageModel object that wraps the HuggingFace model
    """
    print(f"Loading model {model_name}...")
    
    # Look up the model path/name from config.ini
    weights_directory = config[model_name]['weights_directory']
    
    if device == 'remote':
        # Remote mode: Don't load weights locally
        # nnsight will send our tracing code to NDIF servers where the model is already loaded
        model = LanguageModel(weights_directory)
    else:
        # Local mode: Actually download and load the model weights
        model = LanguageModel(weights_directory, torch_dtype=t.bfloat16, device_map="auto")
    
    return model


def load_statements(dataset_name):
    """
    Load statements from a CSV dataset file.
    
    Args:
        dataset_name: Name of dataset without .csv extension (e.g., "cities")
    
    Returns:
        List of statement strings from the 'statement' column
    
    Expected CSV format:
        statement,label
        "The city of Paris is in France.",1
        "The city of Paris is in Germany.",0
    """
    # Read CSV from datasets/ directory
    dataset = pd.read_csv(os.path.join(REPO_ROOT, "datasets", f"{dataset_name}.csv"))
    
    # Extract just the statement text column as a list
    statements = dataset['statement'].tolist()
    
    return statements


def get_acts(statements, model, layers, remote=True):
    """
    Extract hidden state activations from specified layers for a batch of statements.
    
    This is the core function that uses nnsight to "intercept" the model's
    internal computations and save the hidden states.
    
    Args:
        statements: List of text strings to process
        model: nnsight LanguageModel object
        layers: List of layer indices to extract (e.g., [13] or [0,1,2,...,31])
        remote: If True, run on NDIF remote servers; if False, run locally
    
    Returns:
        Dictionary mapping layer_index -> activation tensor
        Each tensor has shape (batch_size, hidden_dim)
        For LLaMA-7B, hidden_dim = 4096
    """
    # Dictionary to store activations for each layer
    acts = {}
    
    
    for layer in layers:
        # Create a variable to capture this layer's output
        saved_activation = None
        
        with model.trace(statements, remote=remote, **tracer_kwargs):
            saved_activation = model.model.layers[layer].output[:, -1, :].save()

        acts[layer] = saved_activation
    
    return acts


# Main Execution

if __name__ == "__main__":
    """
    Main script execution:
    1. Parse command-line arguments
    2. Load the model
    3. For each dataset:
       a. Load statements from CSV
       b. Process in batches of 25
       c. Save activations to .pt files
    """
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate activations for statements in a dataset"
    )
    
    parser.add_argument(
        "--model", 
        default="llama-13b",
        help="Model name as defined in config.ini (e.g., llama-2-7b)"
    )
    
    parser.add_argument(
        "--layers", 
        nargs='+',  # Accept one or more values
        type=int,
        help="Layer indices to extract. Use -1 to save ALL layers."
    )
    
    parser.add_argument(
        "--datasets", 
        nargs='+',  # Accept one or more values
        help="Dataset names without .csv extension (e.g., cities neg_cities)"
    )
    
    parser.add_argument(
        "--output_dir", 
        default=os.path.join(REPO_ROOT, "acts"),
        help="Directory to save activations to (default: <repo_root>/acts/)"
    )
    
    parser.add_argument(
        "--noperiod", 
        action="store_true",  # Flag: if present, set to True
        default=False,
        help="Remove trailing period from statements before processing"
    )
    
    parser.add_argument(
        "--device", 
        default="remote",
        help="Device to run on: 'remote', 'cuda:0', or 'cpu'"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=25,
        help="Number of statements per batch (default: 25, use 3-5 for large models like 70B)"
    )
    
    args = parser.parse_args()

    # Setup
    
    # Disable gradient computation
    t.set_grad_enabled(False)
    
    # Load the model
    model = load_model(args.model, args.device)
    
    # Process each dataset
    for dataset in args.datasets:
        # Load statements from the CSV file
        statements = load_statements(dataset)
        
        # Remove trailing periods from statements
        if args.noperiod:
            statements = [statement[:-1] for statement in statements]
        
        # Handle the special case of layers=[-1] meaning "all layers"
        layers = args.layers
        if layers == [-1]:
            # Get total number of layers from the model
            # For LLaMA-7B this is 32, for LLaMA-13B this is 40
            layers = list(range(len(model.model.layers)))
        
        # Create output directory structure
        
        # Base directory: acts/{model_name}/
        save_dir = os.path.join(f"{args.output_dir}", args.model)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # If --noperiod flag is set, add "noperiod" subdirectory
        if args.noperiod:
            save_dir = os.path.join(save_dir, "noperiod")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        
        # Add dataset name subdirectory: acts/{model_name}/{dataset}/
        save_dir = os.path.join(save_dir, dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Process statements in batches
        batch_size = args.batch_size
        
        for idx in tqdm(range(0, len(statements), batch_size)):
            
            # Get activations for this batch
            acts = get_acts(
                statements[idx:idx + batch_size],  # Batch of statements
                model,                              # The model
                layers,                             # Which layers to extract
                args.device == 'remote'             # True if using NDIF remote
            )
            
            # Save each layer's activations to a separate file
            for layer, act in acts.items():
                # Filename format: layer_{layer_num}_{batch_start_idx}.pt
                # e.g., layer_13_0.pt, layer_13_25.pt, layer_13_50.pt, ...
                filename = f"{save_dir}/layer_{layer}_{idx}.pt"
                
                # Save as PyTorch tensor file
                # These can be loaded later with torch.load(filename)
                t.save(act, filename)