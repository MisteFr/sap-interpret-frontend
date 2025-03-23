import streamlit as st
from utils.analysis import process_sample_token_violations
from utils.text_processing import clean_token
from visualization.plotting import create_violation_histogram, create_colored_token_html
from visualization.edge_inspector import display_token_details

def display_token_analyzer(token_data, token_violations, token_texts=None, original_texts=None):
    """Display the token-level analyzer UI."""
    st.header("Token-Level Violation Explorer")
    
    edge_idx = st.session_state.get('current_edge_idx', 0)
    edge_display = st.session_state.get('current_edge_display', 1)
    
    st.subheader(f"Analyzing Edge {edge_display}")
    
    default_idx = st.session_state.get('selected_sample_idx', 
                                     st.session_state.get('top_violation_indices', [0])[0] 
                                     if len(st.session_state.get('top_violation_indices', [])) > 0 else 0)
    
    sample_idx = st.number_input("Select sample index to explore", 
                                 min_value=0, 
                                 max_value=len(token_data)-1 if token_data is not None else 0, 
                                 value=default_idx)
    
    if sample_idx < len(token_data):
        st.markdown("**Sample text:**")
        if token_texts is not None and sample_idx < len(token_texts):
            st.code(token_texts[sample_idx], language=None)
        elif original_texts is not None and sample_idx < len(original_texts):
            st.code(original_texts[sample_idx], language=None)
        
        sample_tokens, sample_edge_violations = process_sample_token_violations(
            token_data, token_violations, sample_idx, edge_idx
        )
        
        if sample_tokens is not None:
            st.markdown("**Violation Score Distribution:**")
            fig = create_violation_histogram(sample_tokens, sample_edge_violations)
            st.plotly_chart(fig, use_container_width=True, key=f"explorer_hist_sample_{sample_idx}_edge_{edge_display}")

            if sample_edge_violations is not None:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("**Color-coded token violations:**", unsafe_allow_html=True)
                html_content = create_colored_token_html(sample_tokens, sample_edge_violations)
                st.markdown(html_content, unsafe_allow_html=True)
                st.caption("Tokens are colored by violation intensity (darker red = higher violation). Hover over tokens to see exact scores.")
                

                display_token_details(sample_tokens, sample_edge_violations)
            else:
                st.warning("No violation data available for this sample.")
        else:
            st.warning(f"Sample index {sample_idx} is out of range or contains no token data.") 