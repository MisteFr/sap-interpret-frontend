import os
import numpy as np
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

def load_data(npz_path: str, data_type: str):
    """Load latent or edge arrays from an `.npz` stored in the HF dataset repo."""
    # Resolve – will always download (or reuse cached) file.
    npz_path_resolved = _hf_resolve(npz_path)

    data = np.load(npz_path_resolved, allow_pickle=True)
    
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

def load_token_level_data(file_path: str, data_type: str = "edge"):
    """Load token-level activations/violations from the HF dataset repo."""
    file_path_resolved = _hf_resolve(file_path)

    try:
        # (no local existence check needed – resolution would have raised if missing)
        
        file_path_str = str(file_path_resolved)
        
        # Handle different compression formats
        if file_path_str.endswith('.npz'):
            # Standard NPZ format
            loaded = np.load(file_path_resolved, allow_pickle=True)
            
            token_data = loaded.get('tokens', None)
            token_texts = loaded.get('original_texts', loaded.get('texts', None))
            
            if data_type == "edge":
                token_violations = loaded.get('violations', None)
                return token_data, token_violations, token_texts
            else:  # "latent"
                token_activations = loaded.get('activations', None)
                return token_data, token_activations, token_texts
                
        elif file_path_str.endswith('.pkl.gz'):
            # Pickle with gzip compression
            import pickle
            import gzip
            
            with gzip.open(file_path_resolved, 'rb') as f:
                data = pickle.load(f)
            
            token_data = data.get('tokens', None)
            token_texts = data.get('original_texts', None)
            
            if data_type == "edge":
                token_violations = data.get('violations', None)
                return token_data, token_violations, token_texts
            else:  # "latent"
                token_activations = data.get('activations', None)
                return token_data, token_activations, token_texts
                
        elif file_path_str.endswith('.json.gz'):
            # JSON with gzip compression
            import json
            import gzip
            
            with gzip.open(file_path_resolved, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            
            token_data = data.get('tokens', None)
            token_texts = data.get('original_texts', None)
            
            if data_type == "edge":
                violations = data.get('violations', None)
                # Convert back to numpy arrays
                token_violations = [np.array(v) for v in violations] if violations else None
                return token_data, token_violations, token_texts
            else:  # "latent"
                activations = data.get('activations', None)
                # Convert back to numpy arrays
                token_activations = [np.array(v) for v in activations] if activations else None
                return token_data, token_activations, token_texts
                
        elif '_sparse.npz' in file_path_str:
            # Sparse format
            loaded = np.load(file_path_resolved, allow_pickle=True)
            
            token_data = loaded.get('tokens', None)
            token_texts = loaded.get('original_texts', None)
            violations_sparse = loaded.get('violations_sparse', None)
            
            if data_type == "edge" and violations_sparse is not None:
                # Reconstruct full violation matrices
                token_violations = []
                for sparse_data in violations_sparse:
                    shape = sparse_data['shape']
                    violation_matrix = np.zeros(shape)
                    violation_matrix[sparse_data['token_indices'], sparse_data['edge_indices']] = sparse_data['values']
                    token_violations.append(violation_matrix)
                return token_data, token_violations, token_texts
            else:
                return token_data, None, token_texts
                
        elif '_quantized.npz' in file_path_str:
            # Quantized format
            loaded = np.load(file_path_resolved, allow_pickle=True)
            
            token_data = loaded.get('tokens', None)
            token_texts = loaded.get('original_texts', None)
            
            if data_type == "edge":
                violations = loaded.get('violations', None)
                # Convert back to float32 (approximate)
                token_violations = [v.astype(np.float32) for v in violations] if violations else None
                return token_data, token_violations, token_texts
            else:  # "latent"
                activations = loaded.get('activations', None)
                # Convert back to float32 (approximate)
                token_activations = [v.astype(np.float32) for v in activations] if activations else None
                return token_data, token_activations, token_texts
                
        else:
            print(f"Unsupported file format: {file_path}")
            return None, None, None
            
    except Exception as e:
        print(f"Error loading token-level data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None 

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _hf_resolve(path: str) -> str:
    """Return a local (cached) path to *path* inside the dataset repo defined by
    the `HF_DATASET_REPO` environment variable.

    The function always triggers `hf_hub_download`, which will reuse the local
    cache on subsequent calls, so there is no need to manually check if the file
    is already present on disk.
    """
    repo_id = os.getenv("HF_DATASET_REPO")
    if repo_id is None:
        raise RuntimeError(
            "Environment variable `HF_DATASET_REPO` is not set — cannot fetch "
            f"dataset file '{path}'."
        )

    return hf_hub_download(repo_id=repo_id, filename=path, repo_type="dataset") 