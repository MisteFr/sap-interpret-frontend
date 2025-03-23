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

def get_edge_violation_stats(data_array):
    """Calculate violation statistics for each edge."""
    n_samples = data_array.shape[0]
    n_edges = data_array.shape[1]
    
    stats = {
        'edge': [],
        'violation_count': [],
        'violation_percentage': [],
        'max_violation': [],
        'mean_violation': []
    }
    
    for edge_idx in range(n_edges):
        edge_violations = data_array[:, edge_idx]
        # Count positive violations
        positive_violations = edge_violations > 0
        violation_count = np.sum(positive_violations)
        
        # Calculate statistics
        stats['edge'].append(edge_idx + 1)  # 1-indexed for display
        stats['violation_count'].append(violation_count)
        stats['violation_percentage'].append((violation_count / n_samples) * 100)
        
        if violation_count > 0:
            positive_values = edge_violations[positive_violations]
            stats['max_violation'].append(np.max(positive_values))
            stats['mean_violation'].append(np.mean(positive_values))
        else:
            stats['max_violation'].append(0.0)
            stats['mean_violation'].append(0.0)
    
    return pd.DataFrame(stats)

def precompute_edge_statistics(data_array, token_data=None, token_violations=None, output_dir='precomputed_stats', n_processes=None):
    """
    Precompute edge violation statistics and top logits for all edges,
    matching the computation method used in edge_inspector.py.
    
    Args:
        data_array: The edge violations array (samples x edges)
        token_data: The tokenized text data
        token_violations: The token-level violations
        output_dir: Directory to save the precomputed statistics
        n_processes: Number of processes to use (defaults to CPU count)
    
    Returns:
        Dictionary containing edge statistics, token density and top logits
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dictionaries for statistics
    edge_stats = get_edge_violation_stats(data_array)
    token_density = {}
    top_positive_logits = {}
    top_negative_logits = {}
    num_top_tokens = 15  # Number of top tokens to store
    
    # Process token-level data if available
    if token_data is not None and token_violations is not None:
        # Process each edge
        num_edges = data_array.shape[1]
        
        for edge_idx in tqdm(range(num_edges), desc="Processing edges"):
            # Gather all token violations for this edge across all samples
            all_tokens = []
            all_token_violations = []
            
            for sample_idx in range(len(token_data)):
                sample_tokens, sample_edge_violations = process_sample_token_violations(
                    token_data, token_violations, sample_idx, edge_idx
                )
                
                if sample_tokens is not None and sample_edge_violations is not None:
                    all_tokens.extend(sample_tokens)
                    all_token_violations.extend(sample_edge_violations)
            
            # Skip edge if no tokens processed
            if not all_token_violations:
                continue
                
            # Calculate token-level violation density
            token_violations_array = np.array(all_token_violations)
            non_zero_violations = token_violations_array[np.abs(token_violations_array) > 1e-6]
            
            if len(token_violations_array) > 0:
                token_density[edge_idx] = (len(non_zero_violations) / len(token_violations_array)) * 100
            else:
                token_density[edge_idx] = 0.0
            
            # Create a dictionary to store the average violation per token
            token_avg_violations = {}
            token_counts = {}
            
            # Calculate average violation for each token
            for token, violation in zip(all_tokens, all_token_violations):
                clean_tok = clean_token(token)
                if not clean_tok:  # Skip empty tokens
                    continue
                    
                if clean_tok not in token_avg_violations:
                    token_avg_violations[clean_tok] = 0
                    token_counts[clean_tok] = 0
                
                token_avg_violations[clean_tok] += violation
                token_counts[clean_tok] += 1
            
            # Calculate averages
            for token in token_avg_violations:
                token_avg_violations[token] /= token_counts[token]
            
            # Sort tokens by violation values
            sorted_tokens = sorted(token_avg_violations.items(), key=lambda x: x[1])
            
            # Store only if we have tokens
            if sorted_tokens:
                # Store top negative logits (lowest values) - limit to num_top_tokens
                top_negative_logits[edge_idx] = sorted_tokens[:min(num_top_tokens, len(sorted_tokens)//2)]
                
                # Store top positive logits (highest values) - limit to num_top_tokens
                top_positive_logits[edge_idx] = sorted_tokens[max(len(sorted_tokens)-num_top_tokens, len(sorted_tokens)//2):][::-1]
    
    # Store metrics
    stats = {
        'edge_stats': edge_stats.to_dict('list'),
        'token_density': token_density,
        'top_positive_logits': top_positive_logits,
        'top_negative_logits': top_negative_logits
    }
    
    # Save to pickle file
    with open(os.path.join(output_dir, 'edge_stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"Saved edge statistics, token densities, and top logits for {len(token_density)} edges")
    return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Precompute edge statistics for SAP Interpret')
    parser.add_argument('--data_file', type=str, required=True, help='Path to data array file (.npy or .npz)')
    parser.add_argument('--data_type', type=str, default='edge', choices=['latent', 'edge'], 
                       help='Type of data to process (latent or edge)')
    parser.add_argument('--token_data_file', type=str, help='Path to token data file (.npz)')
    parser.add_argument('--token_data_type', type=str, default='edge', choices=['latent', 'edge'],
                       help='Type of token data to process (latent or edge)')
    parser.add_argument('--output_dir', type=str, default='precomputed_stats', help='Output directory for precomputed stats')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_file}...")
    data_array, original_texts = load_data(args.data_file, args.data_type)
    
    token_data = None
    token_violations = None
    
    if args.token_data_file:
        print(f"Loading token data from {args.token_data_file}...")
        token_data_loaded = load_token_level_data(args.token_data_file, args.token_data_type)
        if token_data_loaded:
            token_data, token_violations, token_texts = token_data_loaded
            print(f"Loaded token data with {len(token_data)} samples.")
    
    print(f"Computing edge statistics for {data_array.shape[0]} samples and {data_array.shape[1]} edges...")
    stats = precompute_edge_statistics(data_array, token_data, token_violations, args.output_dir)
    
    print(f"Edge statistics saved to {args.output_dir}") 