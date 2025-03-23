import numpy as np
import pandas as pd
import pickle
import os
import re
import torch
from tqdm import tqdm

def process_sample_token_violations(token_data, token_violations, sample_idx, edge_idx):
    """Process token-level violations for a specific sample and edge."""
    try:
        if token_data is None or sample_idx >= len(token_data):
            return None, None
            
        sample_tokens = token_data[sample_idx]
        if sample_tokens is None:
            return None, None
            
        if token_violations is None or sample_idx >= len(token_violations):
            return None, None
            
        sample_violations = token_violations[sample_idx]
        if sample_violations is None:
            return None, None
            
        if isinstance(sample_violations, torch.Tensor):
            if edge_idx < sample_violations.shape[1]:
                edge_violations = sample_violations[:, edge_idx].numpy()
            else:
                return sample_tokens, None
        elif isinstance(sample_violations, np.ndarray):
            if edge_idx < sample_violations.shape[1]:
                edge_violations = sample_violations[:, edge_idx]
            else:
                return sample_tokens, None
        else:
            return sample_tokens, None
            
        if len(sample_tokens) != len(edge_violations):
            min_len = min(len(sample_tokens), len(edge_violations))
            sample_tokens = sample_tokens[:min_len]
            edge_violations = edge_violations[:min_len]
            
        return sample_tokens, edge_violations
        
    except Exception as e:
        print(f"Error processing sample token violations: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def process_sample_token_activations(token_data, token_activations, sample_idx, latent_idx):
    """Process token-level activations for a specific sample and latent dimension."""
    try:
        if token_data is None or sample_idx >= len(token_data):
            return None, None
            
        sample_tokens = token_data[sample_idx]
        if sample_tokens is None:
            return None, None
            
        if token_activations is None or sample_idx >= len(token_activations):
            return None, None
            
        sample_activations = token_activations[sample_idx]
        if sample_activations is None:
            return None, None
            
        if isinstance(sample_activations, torch.Tensor):
            if latent_idx < sample_activations.shape[1]:
                latent_activations = sample_activations[:, latent_idx].numpy()
            else:
                return sample_tokens, None
        elif isinstance(sample_activations, np.ndarray):
            if latent_idx < sample_activations.shape[1]:
                latent_activations = sample_activations[:, latent_idx]
            else:
                return sample_tokens, None
        else:
            return sample_tokens, None
            
        if len(sample_tokens) != len(latent_activations):
            min_len = min(len(sample_tokens), len(latent_activations))
            sample_tokens = sample_tokens[:min_len]
            latent_activations = latent_activations[:min_len]
            
        return sample_tokens, latent_activations
        
    except Exception as e:
        print(f"Error processing sample token activations: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def clean_token(token):
    """Remove spaces, 'Ġ', 'Ċ', and extra spaces from tokens."""
    if token is None:
        return ""
    token_str = str(token)
    cleaned = re.sub(r'\s+', ' ', token_str.replace('Ġ', '').replace('Ċ', '')).strip()
    return cleaned

def load_data(npz_path, data_type):
    """Load data from an .npz file based on the specified data type."""
    if not os.path.exists(npz_path):
        return None, None
        
    data = np.load(npz_path, allow_pickle=True)
    
    if data_type == "latent":
        if "activations" not in data:
            print(f"No activations found in {npz_path}")
            return None, None
            
        activations = data["activations"]
        if len(activations.shape) == 2:
            activations = activations.reshape(activations.shape[0], 1, activations.shape[1])
        array_data = activations
        
    elif data_type == "edge":
        if "violations" not in data:
            print(f"No violations found in {npz_path}")
            return None, None
            
        violations = data["violations"]
        array_data = violations
    
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    original_texts = data["original_texts"] if "original_texts" in data else None
    
    return array_data, original_texts

def load_tokenizer(model_path):
    """Load a tokenizer from the given model path."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    
    return tokenizer

def load_token_level_data(file_path, data_type="edge"):
    """
    Load token-level data from an .npz file.
    
    Args:
        file_path (str): Path to the .npz file
        data_type (str): Type of data to load - "edge" for violations or "latent" for activations
    
    Returns:
        tuple: Based on data_type:
            - For "edge": (token_data, token_violations, token_texts)
            - For "latent": (token_data, token_activations, token_texts)
    """
    try:
        if not os.path.exists(file_path):
            return None, None, None
            
        loaded = np.load(file_path, allow_pickle=True)
        
        token_data = loaded.get('tokens', None)
        token_texts = loaded.get('texts', None)
        
        if data_type == "edge":
            token_violations = loaded.get('violations', None)
            return token_data, token_violations, token_texts
        else:  # "latent"
            token_activations = loaded.get('activations', None)
            return token_data, token_activations, token_texts
            
    except Exception as e:
        print(f"Error loading token-level data: {str(e)}")
        return None, None, None 

def process_sample_tokens_parallel(sample_idx, token_data, token_activations, data_array_width):
    """Process token-level data for a single sample across all latent dimensions."""
    result = {}
    tokens = token_data[sample_idx]
    if tokens is None:
        return {}, {}
    
    if isinstance(tokens, np.ndarray) and tokens.dtype == np.object_:
        tokens = tokens.tolist()
    
    sample_activations = {}
    unique_tokens = {}
    
    # Process each latent dimension for this sample
    for j in range(data_array_width):
        tokens_processed, activations_processed = process_sample_token_activations(token_data, token_activations, sample_idx, j)
        
        if tokens_processed is None or activations_processed is None:
            continue
            
        # Combine tokens with their activations
        for k, token in enumerate(tokens_processed):
            cleaned_token = clean_token(token)
            if cleaned_token not in sample_activations:
                sample_activations[cleaned_token] = []
                unique_tokens[cleaned_token] = token
            
            sample_activations[cleaned_token].append(activations_processed[k])
    
    # Process the collected activations
    token_stats = {}
    for token, activation in sample_activations.items():
        token_stats[token] = {
            'activation': np.mean(activation),  # Take mean for multiple occurrences
            'count': 1
        }
    
    return token_stats, unique_tokens

def compute_token_stats_parallel(token, data):
    """Compute statistics for a single token."""
    activations = np.vstack(data['activations'])
    activation_mean = np.mean(activations, axis=0)
    
    return token, {
        'mean': activation_mean,
        'std': np.std(activations, axis=0),
        'min': np.min(activations, axis=0),
        'max': np.max(activations, axis=0),
        'count': data['count'],
        'original_token': data['original_token']
    }

def precompute_latent_statistics(data_array, token_data=None, token_activations=None, output_dir='precomputed_stats', n_processes=None):
    """
    Precompute only activation density and top logits for all latent dimensions,
    matching the computation method used in latent_inspector.py.
    
    Args:
        data_array: The latent activations array (samples x 1 x dimensions)
        token_data: The tokenized text data
        token_activations: The token-level activations
        output_dir: Directory to save the precomputed statistics
        n_processes: Number of processes to use (defaults to CPU count)
    
    Returns:
        Dictionary containing only token density and top logits
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Reshape data array if necessary
    if len(data_array.shape) > 2:
        data_array = data_array.reshape(data_array.shape[0], -1)
    
    # Initialize only the essential dictionaries
    token_density = {}
    top_positive_logits = {}
    top_negative_logits = {}
    num_top_tokens = 15  # Number of top tokens to store as shown in latent_inspector
    
    # Only compute the bare minimum stats we need
    stats = {}
    
    # Process token-level data if available
    if token_data is not None and token_activations is not None:
        # Process each latent dimension
        num_dimensions = data_array.shape[1]
        
        for dim_idx in tqdm(range(num_dimensions), desc="Processing latent dimensions"):
            # Gather all token activations for this dimension across all samples
            # This matches how latent_inspector.py processes tokens
            all_tokens = []
            all_token_activations = []
            
            for sample_idx in range(len(token_data)):
                sample_tokens, sample_latent_activations = process_sample_token_activations(
                    token_data, token_activations, sample_idx, dim_idx
                )
                
                if sample_tokens is not None and sample_latent_activations is not None:
                    all_tokens.extend(sample_tokens)
                    all_token_activations.extend(sample_latent_activations)
            
            # Skip dimension if no tokens processed
            if not all_token_activations:
                continue
                
            # Calculate token-level activation density
            token_activations_array = np.array(all_token_activations)
            non_zero_activations = token_activations_array[np.abs(token_activations_array) > 1e-6]
            
            if len(token_activations_array) > 0:
                token_density[dim_idx] = (len(non_zero_activations) / len(token_activations_array)) * 100
            else:
                token_density[dim_idx] = 0.0
            
            # Create a dictionary to store the average activation per token
            token_avg_activations = {}
            token_counts = {}
            
            # Calculate average activation for each token (matching latent_inspector.py)
            for token, activation in zip(all_tokens, all_token_activations):
                clean_tok = clean_token(token)
                if not clean_tok:  # Skip empty tokens
                    continue
                    
                if clean_tok not in token_avg_activations:
                    token_avg_activations[clean_tok] = 0
                    token_counts[clean_tok] = 0
                
                token_avg_activations[clean_tok] += activation
                token_counts[clean_tok] += 1
            
            # Calculate averages
            for token in token_avg_activations:
                token_avg_activations[token] /= token_counts[token]
            
            # Sort tokens by activation values
            sorted_tokens = sorted(token_avg_activations.items(), key=lambda x: x[1])
            
            # Store only if we have tokens
            if sorted_tokens:
                # Store top negative logits (lowest values) - limit to num_top_tokens
                top_negative_logits[dim_idx] = sorted_tokens[:min(num_top_tokens, len(sorted_tokens)//2)]
                
                # Store top positive logits (highest values) - limit to num_top_tokens
                top_positive_logits[dim_idx] = sorted_tokens[max(len(sorted_tokens)-num_top_tokens, len(sorted_tokens)//2):][::-1]
    
    # Store only the metrics requested
    stats['token_density'] = token_density
    stats['top_positive_logits'] = top_positive_logits
    stats['top_negative_logits'] = top_negative_logits
    
    # Save to pickle file
    with open(os.path.join(output_dir, 'basic_stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"Saved token densities and top logits for {len(token_density)} latent dimensions")
    return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Precompute latent statistics for SAP Interpret')
    parser.add_argument('--data_file', type=str, required=True, help='Path to data array file (.npy or .npz)')
    parser.add_argument('--data_type', type=str, default='latent', choices=['latent', 'edge'], 
                       help='Type of data to process (latent or edge)')
    parser.add_argument('--token_data_file', type=str, help='Path to token data file (.npz)')
    parser.add_argument('--token_data_type', type=str, default='latent', choices=['latent', 'edge'],
                       help='Type of token data to process (latent or edge)')
    parser.add_argument('--output_dir', type=str, default='precomputed_stats', help='Output directory for precomputed stats')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_file}...")
    data_array, original_texts = load_data(args.data_file, args.data_type)
    
    token_data = None
    token_activations = None
    
    if args.token_data_file:
        print(f"Loading token data from {args.token_data_file}...")
        token_data_loaded = load_token_level_data(args.token_data_file, args.token_data_type)
        if token_data_loaded:
            token_data, token_activations, token_texts = token_data_loaded
            print(f"Loaded token data with {len(token_data)} samples.")
    
    print(f"Computing statistics for {data_array.shape[0]} samples...")
    stats = precompute_latent_statistics(data_array, token_data, token_activations, args.output_dir)
    
    print(f"Statistics saved to {args.output_dir}") 