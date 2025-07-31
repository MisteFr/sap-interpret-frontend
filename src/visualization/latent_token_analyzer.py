import streamlit as st
from utils.analysis import process_sample_token_activations
from utils.text_processing import clean_token
from visualization.plotting import create_activation_histogram, create_colored_token_html
from visualization.latent_inspector import display_token_details

def display_latent_token_analyzer(token_data, token_activations, token_texts=None, original_texts=None):
    """Display the token-level analyzer UI for latent dimensions."""
    st.header("Token-Level Activation Explorer")
    
    latent_idx = st.session_state.get('current_latent_idx', 0)
    latent_display = st.session_state.get('current_latent_display', 1)
    
    st.subheader(f"Analyzing Latent Dimension {latent_display}")
    
    default_idx = st.session_state.get(
        'selected_sample_idx',
        st.session_state.get('top_activation_indices', [0])[0]
        if len(st.session_state.get('top_activation_indices', [])) > 0 else 0
    )

    max_idx = len(token_data) - 1 if token_data is not None else 0
    if default_idx > max_idx:
        default_idx = max_idx

    sample_idx = st.number_input(
        "Select sample index to explore",
        min_value=0,
        max_value=max_idx,
        value=default_idx,
        key="sample_idx_input"
    )
    
    if sample_idx < len(token_data):
        st.markdown("**Sample text:**")
        if token_texts is not None and sample_idx < len(token_texts):
            st.code(token_texts[sample_idx], language=None)
        elif original_texts is not None and sample_idx < len(original_texts):
            st.code(original_texts[sample_idx], language=None)
        
        sample_tokens, sample_latent_activations = process_sample_token_activations(
            token_data, token_activations, sample_idx, latent_idx
        )
        
        if sample_tokens is not None:
            st.markdown("**Activation Score Distribution:**")
            fig = create_activation_histogram(sample_tokens, sample_latent_activations)
            st.plotly_chart(fig, use_container_width=True, key=f"explorer_hist_sample_{sample_idx}_latent_{latent_display}")

            if sample_latent_activations is not None:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("**Color-coded token activations:**", unsafe_allow_html=True)
                html_content = create_colored_token_html(sample_tokens, sample_latent_activations)
                st.markdown(html_content, unsafe_allow_html=True)
                st.caption("Tokens are colored by activation intensity (darker red = higher activation). Hover over tokens to see exact scores.")
                
                display_token_details(sample_tokens, sample_latent_activations)
            else:
                st.warning("No activation data available for this sample.")
        else:
            st.warning(f"Sample index {sample_idx} is out of range or contains no token data.") 