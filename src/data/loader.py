import os
import numpy as np
from transformers import AutoTokenizer

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