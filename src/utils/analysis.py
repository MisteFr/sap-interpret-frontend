import numpy as np
import pandas as pd
import torch
import re

def clean_text(input_text):
    """Remove unwanted characters like 'Ċ', 'Ġ' and extra spaces from the text."""
    if input_text is None:
        return ""
    # Remove special tokenizer characters and normalize spaces
    cleaned = input_text.replace('Ġ', '').replace('Ċ', '')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

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

def get_top_violations_for_edge(violations, edge_idx, top_k=5, ascending=False, original_texts=None):
    """Get samples with highest or lowest violation scores for an edge.
    
    Args:
        violations (np.ndarray): Array of violation scores
        edge_idx (int): Index of the edge to analyze
        top_k (int): Number of top samples to return
        ascending (bool): If True, returns samples with lowest violations
                         If False (default), returns samples with highest violations
        original_texts (list): List of original texts to use for deduplication
    
    Returns:
        tuple: (indices, violation_scores) of the top samples
    """
    edge_violations = violations[:, edge_idx]
    
    # Get more samples initially to ensure we have enough unique ones
    initial_k = top_k * 4 if original_texts is not None else top_k
    
    if ascending:
        top_indices = np.argsort(edge_violations)[:initial_k]
    else:
        top_indices = np.argsort(edge_violations)[-initial_k:][::-1]
        
    top_violation_scores = edge_violations[top_indices]
    
    # If no original texts provided, return all samples
    if original_texts is None:
        return top_indices, top_violation_scores
    
    # Deduplicate based on cleaned text
    unique_indices = []
    unique_scores = []
    seen_texts = set()
    
    for idx, score in zip(top_indices, top_violation_scores):
        if idx < len(original_texts):
            text = clean_text(original_texts[idx])
            if text not in seen_texts:
                seen_texts.add(text)
                unique_indices.append(idx)
                unique_scores.append(score)
                if len(unique_indices) >= top_k:
                    break
    
    return np.array(unique_indices), np.array(unique_scores)

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

def get_token_associations(token_data, token_violations, target_token, edge_idx, n_associations=5, max_samples=1000):
    """Calculate the most associated next and previous tokens for a given token in the context of edge violations.
    
    Args:
        token_data: List of tokenized samples
        token_violations: Token-level violation scores
        target_token: The token to find associations for
        edge_idx: Index of the edge to analyze
        n_associations: Number of top associations to return
        max_samples: Maximum number of samples to process to prevent memory growth
    
    Returns:
        tuple: (top_next_tokens, top_prev_tokens) where each is a list of (token, count) tuples
    """
    if not target_token or not isinstance(target_token, str):
        return [], []
        
    target_token = clean_text(target_token)
    if not target_token:
        return [], []
    
    # First pass: quickly scan all samples to find ones containing our target token
    relevant_indices = []
    for idx, tokens in enumerate(token_data):
        if tokens is not None and any(clean_text(t) == target_token for t in tokens):
            relevant_indices.append(idx)
    
    # If we have too many relevant samples, randomly sample from them
    if len(relevant_indices) > max_samples:
        # Shuffle the indices and take the first max_samples
        relevant_indices = list(np.random.permutation(relevant_indices)[:max_samples])
    
    # Use dictionaries with size limits to prevent unbounded growth    
    next_token_counts = {}
    prev_token_counts = {}
    max_dict_size = max(n_associations * 20, 100)  # Increased buffer for more complete collection
    
    # Process only the relevant samples
    for sample_idx in relevant_indices:
        sample_tokens, sample_edge_violations = process_sample_token_violations(
            token_data, token_violations, sample_idx, edge_idx
        )
        
        if sample_tokens is None or sample_edge_violations is None:
            continue
        
        # Find all occurrences of the target token in this sample
        for i, token in enumerate(sample_tokens):
            if not token or not isinstance(token, str):
                continue
                
            if clean_text(token) == target_token:
                # Process next tokens (can be multiple)
                for j in range(1, min(3, len(sample_tokens) - i)):  # Look at next 2 tokens
                    if i + j < len(sample_tokens):
                        next_token = clean_text(sample_tokens[i + j])
                        if next_token:
                            next_token_counts[next_token] = next_token_counts.get(next_token, 0) + (3-j)  # Weight closer tokens higher
                
                # Process previous tokens (can be multiple)
                for j in range(1, min(3, i + 1)):  # Look at previous 2 tokens
                    if i - j >= 0:
                        prev_token = clean_text(sample_tokens[i - j])
                        if prev_token:
                            prev_token_counts[prev_token] = prev_token_counts.get(prev_token, 0) + (3-j)  # Weight closer tokens higher
                
                # Prune dictionaries if they get too large
                if len(next_token_counts) > max_dict_size:
                    next_token_counts = dict(sorted(next_token_counts.items(), 
                                                  key=lambda x: x[1], 
                                                  reverse=True)[:max_dict_size])
                if len(prev_token_counts) > max_dict_size:
                    prev_token_counts = dict(sorted(prev_token_counts.items(), 
                                                  key=lambda x: x[1], 
                                                  reverse=True)[:max_dict_size])
    
    # Get final top associations
    top_next = sorted(next_token_counts.items(), key=lambda x: x[1], reverse=True)[:n_associations]
    top_prev = sorted(prev_token_counts.items(), key=lambda x: x[1], reverse=True)[:n_associations]
    
    # Clear dictionaries to help with memory
    next_token_counts.clear()
    prev_token_counts.clear()
    
    return top_next, top_prev