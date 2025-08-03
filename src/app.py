import streamlit as st
import os
import numpy as np
from data.loader import load_data, load_token_level_data
from visualization.latent_inspector import display_latent_inspector
from visualization.edge_inspector import display_edge_inspector
from visualization.token_analyzer import display_token_analyzer
from visualization.latent_token_analyzer import display_latent_token_analyzer
from visualization.sample_edge_analyzer import display_sample_edge_analyzer
from visualization.edge_correlation_analyzer import display_edge_correlation_analyzer

def get_available_data_files():
    """Detect available data files in the data directory."""
    data_dir = "./data"
    available_files = {
        'latent_token': [],
        'latent': [],
        'token': [],
        'edge': []
    }
    
    # Look for all directories
    for item in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, item)):
            dir_path = os.path.join(data_dir, item)
            for file in os.listdir(dir_path):
                if file.endswith('.npz'):
                    if 'token_level_activations' in file:
                        available_files['latent_token'].append(os.path.join(dir_path, file))
                    elif 'sae_activations' in file:
                        available_files['latent'].append(os.path.join(dir_path, file))
                    elif 'token_level_violations' in file:
                        available_files['token'].append(os.path.join(dir_path, file))
                    elif 'edge_violations' in file:
                        available_files['edge'].append(os.path.join(dir_path, file))
    
    return available_files

# -----------------------------------------------------------------------------
# Pre-load datasets into Streamlit's cache at server start
# -----------------------------------------------------------------------------

# Caching this function ensures it executes only once (per server lifetime). It
# eagerly loads every dataset found in ./data into the existing cache helpers
# so that later calls to `load_data` / `load_token_level_data` return instantly.

@st.cache_resource(show_spinner=False)
def _preload_all_datasets():
    """Eagerly load all .npz datasets so the first user doesn't wait."""

    available = get_available_data_files()

    # Latent and edge-level arrays ------------------------------------------------
    for path in available.get("latent", []):
        load_data(path, data_type="latent")

    for path in available.get("edge", []):
        load_data(path, data_type="edge")

    # Token-level data ------------------------------------------------------------
    for path in available.get("latent_token", []):
        load_token_level_data(path, data_type="latent")

    for path in available.get("token", []):
        load_token_level_data(path, data_type="edge")

    # Return a trivial value so Streamlit keeps the resource around
    return True


# Kick off the preload process immediately when the module is imported. Because
# the function above is cached as a resource, the heavy work is performed only
# once, during server start-up, and subsequent script re-runs reuse the cache.

_ = _preload_all_datasets()

def format_file_option(filepath):
    """Format the filepath to show filename (folder_name)"""
    filename = os.path.basename(filepath)
    folder_name = os.path.basename(os.path.dirname(filepath))
    return f"{filename} ({folder_name})"

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
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    button[data-testid="stToolbarStop"] {display: none !important;}
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

    # ---------------------------------------------------------------------
    # Overview / About section (displayed prominently below the main title)
    # ---------------------------------------------------------------------
    st.markdown(
        """

Safety Polytope (SaP) is a geometric safety layer that defines a set of half-space constraints ("facets") in the hidden-state representation space of a language model (see *[Learning Safety Constraints for Large Language Models](https://arxiv.org/abs/2505.24445)* for the technical details).

This dashboard lets you inspect how data from the [BeaverTail](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) terrorism category interacts with those constraints:

• **Latent Dimension Explorer** – browse features of the SaP Concept Encoder and view samples/logits that maximally activate them.  
• **Facet / Edge Violation Inspector** – identify inputs/logits that violate safety facets and drill down to token level.  
• **Token-level Heat-maps** – see which tokens contribute most to each violation.

Made as a part of a semester research project by [Arthur Bigot](https://github.com/MisteFr).  
A project report is coming soon.
        """
    )
    
    mode = st.sidebar.radio(
        "Exploration Mode", 
        ["Latent Inspection", "Edge Violation Inspection"],
        index=1
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
        

    if mode == "Latent Inspection":
        file_to_use = latent_file
        data_type = "latent"
    else:  # Edge Violation mode
        file_to_use = edge_file
        data_type = "edge"
        
    if not file_to_use or not os.path.exists(file_to_use):
        st.error(f"Please select a valid {data_type} data file in the Settings tab.")
        return

    data_array, original_texts = load_data(file_to_use, data_type=data_type)
    
    if data_array is None:
        st.error(f"Could not load {data_type} data from {file_to_use}.")
        return
        

    
    edge_token_data, edge_token_violations, token_texts = None, None, None
    latent_token_data, latent_token_activations = None, None
    
    if mode == "Edge Violation Inspection":
        if token_file and os.path.exists(token_file):
            edge_token_data, edge_token_violations, token_texts = load_token_level_data(token_file, data_type="edge")
            has_token_data = edge_token_data is not None

            if has_token_data:
                st.sidebar.success(f"Token-level violation data loaded from {os.path.basename(token_file)}")
            else:
                st.sidebar.info("No token-level violation data loaded.")
        else:
            st.sidebar.info("No token-level violation data file selected.")
    else:  # Latent Inspection mode
        if latent_token_file and os.path.exists(latent_token_file):
            latent_token_data, latent_token_activations, token_texts = load_token_level_data(latent_token_file, data_type="latent")
            has_latent_token_data = latent_token_data is not None
            if has_latent_token_data:
                st.sidebar.success(f"Token-level activation data loaded from {os.path.basename(latent_token_file)}")
            else:
                st.sidebar.info("No token-level activation data loaded.")
        else:
            st.sidebar.info("No token-level activation data file selected.")
    
    st.sidebar.write(f"Dataset contains {data_array.shape[0]} samples")
    
    with st.sidebar:
        with st.expander("Help", expanded=False):
            st.markdown("""
            **Key Concepts:**
            - **Edge/Facet**: Safety constraint boundary that defines safe vs unsafe content
            - **Violation**: When text crosses a safety boundary (positive scores = potentially unsafe)
            - **Latent Dimension**: Hidden features the AI model learned to recognize
            - **Positive Logits**: Tokens that contribute to safety violations
            - **Negative Logits**: Tokens that indicate safe content
            - **Activation**: How strongly a feature responds to input text
            
            **Score Interpretation:**
            - **Higher positive values** = stronger unsafe signal
            - **Higher negative values** = stronger safe signal
            - **Values near zero** = neutral/ambiguous content
            
            **BeaverTail Dataset**: Contains examples from terrorism category for safety analysis
            """)
    
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

    # -----------------------------------------------------------------
    # Custom footer (source code link & report notice)
    # -----------------------------------------------------------------
    st.markdown(
        """
---
View the source code on [GitHub](https://github.com/MisteFr/sap-interpret-frontend)  """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main() 