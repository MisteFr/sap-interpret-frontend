import numpy as np
import pandas as pd
import torch

def get_top_features_for_latent(activations, latent_idx, top_k=5, ascending=False):
    """Get samples with highest or lowest activations for a latent dimension.
    
    Args:
        activations (np.ndarray): Array of activations
        latent_idx (int): Index of the latent dimension to analyze
        top_k (int): Number of top samples to return
        ascending (bool): If True, returns samples with lowest activations
                         If False (default), returns samples with highest activations
    
    Returns:
        tuple: (indices, activation_values) of the top samples
    """
    if len(activations.shape) == 3:
        max_activations = np.max(activations[:, :, latent_idx], axis=1)
    elif len(activations.shape) == 2:
        max_activations = activations[:, latent_idx]
    else:
        raise ValueError(f"Unexpected activation shape: {activations.shape}")
    
    if ascending:
        top_indices = np.argsort(max_activations)[:top_k]
        top_activations = max_activations[top_indices]
    else:
        top_indices = np.argsort(max_activations)[-top_k:][::-1]
        top_activations = max_activations[top_indices]
    
    return top_indices, top_activations

def get_top_violations_for_edge(violations, edge_idx, top_k=5, ascending=False):
    """Get samples with highest or lowest violation scores for an edge.
    
    Args:
        violations (np.ndarray): Array of violation scores
        edge_idx (int): Index of the edge to analyze
        top_k (int): Number of top samples to return
        ascending (bool): If True, returns samples with lowest violations
                         If False (default), returns samples with highest violations
    
    Returns:
        tuple: (indices, violation_scores) of the top samples
    """
    edge_violations = violations[:, edge_idx]
    
    if ascending:
        top_indices = np.argsort(edge_violations)[:top_k]
    else:
        top_indices = np.argsort(edge_violations)[-top_k:][::-1]
        
    top_violation_scores = edge_violations[top_indices]
    
    return top_indices, top_violation_scores

def get_edge_violation_stats(violations):
    """Calculate violation statistics for each edge."""
    n_samples, n_edges = violations.shape
    stats = []
    
    for edge_idx in range(n_edges):
        edge_data = violations[:, edge_idx]
        positive_violations = np.sum(edge_data > 0)
        max_violation = np.max(edge_data)
        mean_violation = np.mean(edge_data)
        
        stats.append({
            'edge': edge_idx + 1,
            'violation_count': positive_violations,
            'violation_percentage': (positive_violations / n_samples) * 100,
            'max_violation': max_violation,
            'mean_violation': mean_violation
        })
    
    return pd.DataFrame(stats)

def get_latent_activation_stats(activations):
    """Calculate activation statistics for each latent dimension.
    
    Args:
        activations (np.ndarray): Array of shape (n_samples, n_timesteps, n_latents) 
            or (n_samples, n_latents)
    
    Returns:
        pd.DataFrame: Statistics for each latent dimension including:
            - latent: index of the latent dimension (1-based)
            - activation_count: number of samples with positive activations
            - activation_percentage: percentage of samples with positive activations
            - max_activation: maximum activation value
            - mean_activation: mean activation value
    """
    if len(activations.shape) == 3:
        activations = np.max(activations, axis=1)
    
    n_samples, n_latents = activations.shape
    stats = []
    
    for latent_idx in range(n_latents):
        latent_data = activations[:, latent_idx]
        positive_activations = np.sum(latent_data > 0)
        max_activation = np.max(latent_data)
        mean_activation = np.mean(latent_data)
        
        stats.append({
            'latent': latent_idx + 1,
            'activation_count': positive_activations,
            'activation_percentage': (positive_activations / n_samples) * 100,
            'max_activation': max_activation,
            'mean_activation': mean_activation
        })
    
    return pd.DataFrame(stats)

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