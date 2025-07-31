import streamlit as st
import os
from data.loader import load_data, load_tokenizer, load_token_level_data
from visualization.latent_inspector import display_latent_inspector
from visualization.edge_inspector import display_edge_inspector
from visualization.token_analyzer import display_token_analyzer
from visualization.latent_token_analyzer import display_latent_token_analyzer
from visualization.sample_edge_analyzer import display_sample_edge_analyzer
from visualization.edge_correlation_analyzer import display_edge_correlation_analyzer
from huggingface_hub import HfApi

def get_available_data_files():
    """List available .npz data files from the Hugging Face dataset repo defined by `HF_DATASET_REPO`."""
    repo_id = os.getenv("HF_DATASET_REPO")

    if not repo_id:
        st.error("Environment variable `HF_DATASET_REPO` is not set — the app cannot list data files.")
        return {k: [] for k in ['latent_token', 'latent', 'token', 'edge']}

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as exc:
        st.error(f"Could not list files in Hugging Face dataset `{repo_id}`: {exc}")
        return {k: [] for k in ['latent_token', 'latent', 'token', 'edge']}

    available_files = {
        'latent_token': [f for f in files if f.endswith('.npz') and 'token_level_activations' in f],
        'latent': [f for f in files if f.endswith('.npz') and 'sae_activations' in f],
        'token': [f for f in files if f.endswith('.npz') and 'token_level_violations' in f],
        'edge': [f for f in files if f.endswith('.npz') and 'edge_violations' in f],
    }
    return available_files

def format_file_option(filepath):
    """Format file option for display."""
    filename = os.path.basename(filepath)
    parent = os.path.dirname(filepath)
    return f"{filename} ({parent})" if parent else filename

def main():
    # We no longer pre-download the entire dataset; individual files are fetched on-demand.
    
    st.set_page_config(
        page_title="Latent Space & Edge Violations Explorer",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    html, body, [class*="st-"] {
        font-size: 0.9rem !important;
    }
    .stMarkdown, .stText, .stCode, .stDataFrame {
        font-size: 0.9rem !important;
    }
    /* Make tables more compact */
    table {
        font-size: 0.85rem !important;
    }
    /* Make expanders more compact */
    .streamlit-expanderHeader {
        font-size: 0.9rem !important;
    }
    /* More compact headers */
    h1 {
        font-size: 1.8rem !important;
    }
    h2 {
        font-size: 1.5rem !important;
    }
    h3 {
        font-size: 1.2rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Latent Space & Edge Violations Explorer")
    
    mode = st.sidebar.radio(
        "Exploration Mode", 
        ["Latent Inspection", "Edge Violation Inspection"],
        index=0
    )
    
    tabs = st.tabs(["Explorer", "Token Analysis", "Sample Edge Analysis", "Edge Correlation", "Settings"])
    
    with tabs[4]:  # Settings tab
        st.header("Data Settings")
        
        # Get available files
        available_files = get_available_data_files()
        
        # Create dropdowns for file selection
        if mode == "Latent Inspection":
            if available_files['latent_token']:
                latent_token_file = st.selectbox(
                    "Select token-level activations file",
                    available_files['latent_token'],
                    format_func=format_file_option
                )
            else:
                st.warning("No token-level activation files found")
                latent_token_file = None
                
            if available_files['latent']:
                latent_file = st.selectbox(
                    "Select latent activations file",
                    available_files['latent'],
                    format_func=format_file_option
                )
            else:
                st.warning("No latent activation files found")
                latent_file = None
        else:  # Edge Violation mode
            if available_files['token']:
                token_file = st.selectbox(
                    "Select token-level violations file",
                    available_files['token'],
                    format_func=format_file_option
                )
            else:
                st.warning("No token-level violation files found")
                token_file = None
                
            if available_files['edge']:
                edge_file = st.selectbox(
                    "Select edge violations file",
                    available_files['edge'],
                    format_func=format_file_option
                )
            else:
                st.warning("No edge violation files found")
                edge_file = None
        
        model_path = st.text_input("Path or identifier for tokenizer", "mistralai/Ministral-8B-Instruct-2410")
    
    if mode == "Latent Inspection":
        file_to_use = latent_file
        data_type = "latent"
    else:  # Edge Violation mode
        file_to_use = edge_file
        data_type = "edge"
        
    if not file_to_use:
        st.error(f"Please select a {data_type} data file in the Settings tab.")
        return

    data_array, original_texts = load_data(file_to_use, data_type=data_type)
    
    if data_array is None:
        st.error(f"Could not load {data_type} data from {file_to_use}.")
        return

    # tokenizer = load_tokenizer(model_path)

    edge_token_data, edge_token_violations, token_texts = None, None, None
    latent_token_data, latent_token_activations = None, None
    
    if mode == "Edge Violation Inspection":
        if token_file:
            edge_token_data, edge_token_violations, token_texts = load_token_level_data(token_file, data_type="edge")
            has_token_data = edge_token_data is not None

            if has_token_data:
                st.sidebar.success(f"Token-level violation data loaded from {os.path.basename(token_file)}")
            else:
                st.sidebar.info("No token-level violation data loaded.")
        else:
            st.sidebar.info("No token-level violation data file selected.")
    else:  # Latent Inspection mode
        if latent_token_file:
            latent_token_data, latent_token_activations, token_texts = load_token_level_data(latent_token_file, data_type="latent")
            has_latent_token_data = latent_token_data is not None
            if has_latent_token_data:
                st.sidebar.success(f"Token-level activation data loaded from {os.path.basename(latent_token_file)}")
            else:
                st.sidebar.info("No token-level activation data loaded.")
        else:
            st.sidebar.info("No token-level activation data file selected.")
    
    st.sidebar.write(f"Dataset contains {data_array.shape[0]} samples")
    if original_texts is not None:
        st.sidebar.write(f"Original texts available: {len(original_texts)}")
    
    with tabs[0]:  # Explorer tab
        if mode == "Latent Inspection":
            display_latent_inspector(
                data_array, 
                original_texts,
                token_data=latent_token_data,
                token_activations=latent_token_activations
            )
        else:  # Edge Violation mode
            display_edge_inspector(
                data_array, 
                original_texts, 
                token_data=edge_token_data, 
                token_violations=edge_token_violations
            )
    
    with tabs[1]:  # Token Analysis tab
        if mode == "Edge Violation Inspection":
            if edge_token_data is None:
                st.warning("No token-level violation data loaded.")
            else:
                display_token_analyzer(
                    edge_token_data, 
                    edge_token_violations, 
                    token_texts, 
                    original_texts
                )
        else:  # Latent Inspection mode
            if latent_token_data is None:
                st.warning("No token-level activation data loaded.")
            else:
                display_latent_token_analyzer(
                    latent_token_data,
                    latent_token_activations,
                    original_texts=original_texts
                )
    
    with tabs[2]:  # Sample Edge Analysis tab
        if mode == "Edge Violation Inspection":
            if data_array is None:
                st.warning("No edge violation data loaded.")
            else:
                display_sample_edge_analyzer(
                    data_array,
                    original_texts
                )
        else:
            st.info("Sample edge analysis is only available in Edge Violation Inspection mode.")

    with tabs[3]:  # Edge Correlation tab
        if mode == "Edge Violation Inspection":
            if data_array is None:
                st.warning("No edge violation data loaded.")
            else:
                display_edge_correlation_analyzer(data_array)
        else:
            st.info("Edge correlation analysis is only available in Edge Violation Inspection mode.")

if __name__ == "__main__":
    main() 