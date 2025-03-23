import streamlit as st
import os
from data.loader import load_data, load_tokenizer, load_token_level_data
from visualization.latent_inspector import display_latent_inspector
from visualization.edge_inspector import display_edge_inspector
from visualization.token_analyzer import display_token_analyzer
from visualization.latent_token_analyzer import display_latent_token_analyzer
from visualization.sample_edge_analyzer import display_sample_edge_analyzer

def main():
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
    
    tabs = st.tabs(["Explorer", "Token Analysis", "Sample Edge Analysis", "Settings"])
    
    with tabs[3]:  # Settings tab
        st.header("Data Settings")
        # latent_token_file = st.text_input("Path to token-level activations (.npz) file", "./data/test_token_level_activations.npz")
        # latent_file = st.text_input("Path to latent activations (.npz) file", "./data/test_sae_activations.npz")
        # token_file = st.text_input("Path to token-level violations (.npz) file", "./data/token_level_violations.npz")
        latent_token_file = st.text_input("Path to token-level activations (.npz) file", "./data/token_level_activations.npz")
        latent_file = st.text_input("Path to latent activations (.npz) file", "./data/sae_activations.npz")
        token_file = st.text_input("Path to token-level violations (.npz) file", "./data/token_level_violations.npz")
        
        
        edge_file = st.text_input("Path to edge violations (.npz) file", "./data/edge_violations.npz")
        model_path = st.text_input("Path or identifier for tokenizer", "mistralai/Ministral-8B-Instruct-2410")
    
    if mode == "Latent Inspection":
        file_to_use = latent_file
        data_type = "latent"
    else:  # Edge Violation mode
        file_to_use = edge_file
        data_type = "edge"
        
    if not os.path.exists(file_to_use):
        st.error(f"The file '{file_to_use}' does not exist. Please provide a valid file path in the Settings tab.")
        return

    data_array, original_texts = load_data(file_to_use, data_type=data_type)
    
    if data_array is None:
        st.error(f"Could not load {data_type} data from {file_to_use}.")
        return
        
    tokenizer = load_tokenizer(model_path)
    
    edge_token_data, edge_token_violations, token_texts = None, None, None
    latent_token_data, latent_token_activations = None, None
    
    if mode == "Edge Violation Inspection":
        edge_token_data, edge_token_violations, token_texts = load_token_level_data(token_file, data_type="edge")
        has_token_data = edge_token_data is not None

        if has_token_data:
            st.sidebar.success(f"Token-level violation data loaded from {token_file}")
        else:
            st.sidebar.info("No token-level violation data loaded.")
    else:  # Latent Inspection mode
        if os.path.exists(latent_token_file):
            latent_token_data, latent_token_activations, token_texts = load_token_level_data(latent_token_file, data_type="latent")
            has_latent_token_data = latent_token_data is not None
            if has_latent_token_data:
                st.sidebar.success(f"Token-level activation data loaded from {latent_token_file}")
            else:
                st.sidebar.info("No token-level activation data loaded.")
    
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
                    original_texts,
                    tokenizer
                )
        else:
            st.info("Sample edge analysis is only available in Edge Violation Inspection mode.")

if __name__ == "__main__":
    main() 