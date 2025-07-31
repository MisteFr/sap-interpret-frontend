import streamlit as st
from utils.analysis import get_top_features_for_latent, get_latent_activation_stats, process_sample_token_activations
from utils.text_processing import extract_relevant_text, clean_text, clean_token
from visualization.plotting import create_activation_histogram, create_colored_token_html
import pandas as pd
import plotly.figure_factory as ff
import numpy as np
import plotly.express as px
import os
import pickle

def display_latent_inspector(data_array, original_texts, token_data=None, token_activations=None, precomputed_dir='basic_stats'):
    """Display the latent space inspector UI."""
    st.markdown(
        """
        This interface lets you explore the latent space by selecting a latent dimension 
        and viewing the samples that maximally activate that dimension.
        """
    )
    
    print(data_array.shape)
    
    latents_stats = get_latent_activation_stats(data_array)
    
    print(data_array.shape)
    
    with st.sidebar:
        with st.expander("📊 Latents Activation Statistics", expanded=False):
            sorted_stats = latents_stats.sort_values('activation_count', ascending=False)
            
            display_df = sorted_stats.copy()
            
            display_df['activation_percentage'] = display_df['activation_percentage'].round(2)
            display_df['max_activation'] = display_df['max_activation'].round(3)
            display_df['mean_activation'] = display_df['mean_activation'].round(3)
            
            display_df.columns = ['Latent', 'Activation', '% of Samples', 'Max Score', 'Mean Score']
            # Remove the default numeric index so it doesn't appear as an unnamed column
            display_df = display_df.reset_index(drop=True)
            
            st.dataframe(display_df, hide_index=True)
            
            min_latent = int(display_df['Latent'].min())
            max_latent = int(display_df['Latent'].max())
            
            selected_latent = st.number_input(
                "Select a latent dimension",
                min_value=min_latent,
                max_value=max_latent,
                value=min_latent,
                step=1
            )

    n_features = data_array.shape[-1]
    st.sidebar.write(f"Total latent dimensions: {n_features}")
    
    dimension_idx = int(selected_latent) - 1
    
    if dimension_idx >= n_features:
        st.error(f"Latent dimension {selected_latent} is out of bounds. The maximum dimension is {n_features}.")
        dimension_idx = 0
        selected_latent = 1
        
        st.sidebar.warning(f"Selected dimension was reset to {selected_latent} because the previous selection was invalid.")
        st.session_state['selected_latent'] = selected_latent
        st.experimental_rerun()
    
    st.session_state['current_latent_idx'] = dimension_idx
    st.session_state['current_latent_display'] = selected_latent

    neg_col, pos_col, density_col = st.columns(3)
    
    dimension_activations = data_array[:, 0, dimension_idx].flatten()
    
    if token_data is not None and token_activations is not None:
        all_token_activations = []

        for sample_idx in range(len(token_data)):
            _, sample_latent_activations = process_sample_token_activations(
                token_data, token_activations, sample_idx, dimension_idx
            )
            
            if sample_latent_activations is not None:
                all_token_activations.extend(sample_latent_activations)
        
        if all_token_activations:
            token_activations_array = np.array(all_token_activations)
            non_zero_activations = token_activations_array[np.abs(token_activations_array) > 1e-6]
            activation_density = (len(non_zero_activations) / len(token_activations_array)) * 100
            dimension_activations = token_activations_array
        else:
            non_zero_activations = dimension_activations[np.abs(dimension_activations) > 1e-6]
            activation_density = (len(non_zero_activations) / len(dimension_activations)) * 100
    else:
        non_zero_activations = dimension_activations[np.abs(dimension_activations) > 1e-6]
        activation_density = (len(non_zero_activations) / len(dimension_activations)) * 100
    
    with density_col:
        st.markdown(f"# Activations Density: {activation_density:.3f}%", unsafe_allow_html=True)
        
        max_samples = 10000
        print("Total activations collected", len(non_zero_activations))
        if len(non_zero_activations) > max_samples:
            sampled_activations = np.random.choice(non_zero_activations, size=max_samples, replace=True)
        else:
            sampled_activations = non_zero_activations
        
        fig = px.histogram(
            pd.DataFrame({
                "Activation Score": sampled_activations
            }), 
            x="Activation Score",
            nbins=50,
            title="",
            color_discrete_sequence=["rgba(100, 100, 255, 0.7)"]
        )
        
        fig.add_vline(
            x=0, 
            line_dash="dash", 
            line_color="gray", 
            annotation_text=""
        )
        
        fig.update_layout(
            plot_bgcolor="white",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    if token_data is not None and token_activations is not None:
        all_tokens = []
        all_token_activations = []
        
        for sample_idx in range(len(token_data)):
            sample_tokens, sample_latent_activations = process_sample_token_activations(
                token_data, token_activations, sample_idx, dimension_idx
            )
            
            if sample_tokens is not None and sample_latent_activations is not None:
                all_tokens.extend(sample_tokens)
                all_token_activations.extend(sample_latent_activations)
        
        token_avg_activations = {}
        token_counts = {}
        
        for token, activation in zip(all_tokens, all_token_activations):
            clean_tok = clean_token(token)
            if clean_tok not in token_avg_activations:
                token_avg_activations[clean_tok] = 0
                token_counts[clean_tok] = 0
            
            token_avg_activations[clean_tok] += activation
            token_counts[clean_tok] += 1
        
        for token in token_avg_activations:
            token_avg_activations[token] /= token_counts[token]
        
        filtered_tokens = {token: value for token, value in token_avg_activations.items() 
                          if token_counts.get(token, 0) >= 1}
        
        sorted_tokens = sorted(filtered_tokens.items(), key=lambda x: x[1])
        
        display_count = 10
        
        top_negative = sorted_tokens[:min(display_count, len(sorted_tokens)//2)]
        
        top_positive = sorted_tokens[max(len(sorted_tokens)-display_count, len(sorted_tokens)//2):][::-1]
        
        with neg_col:
            st.markdown(f"# Negative Logits")
            for token, value in top_negative:
                count = token_counts.get(token, 0)
                st.markdown(
                    f"""<div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="background-color: rgba(255, 100, 100, 0.3); padding: 2px 8px; border-radius: 4px;">{token}</span>
                        <span style="display: flex; gap: 10px;">
                            <span title="Frequency">{count}×</span>
                            <span>{value:.2f}</span>
                        </span>
                    </div>""",
                    unsafe_allow_html=True
                )
        
        with pos_col:
            st.markdown(f"# Positive Logits")
            for token, value in top_positive:
                if value > 0:
                    count = token_counts.get(token, 0)
                    st.markdown(
                        f"""<div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span style="background-color: rgba(100, 100, 255, 0.3); padding: 2px 8px; border-radius: 4px;">{token}</span>
                            <span style="display: flex; gap: 10px;">
                                <span title="Frequency">{count}×</span>
                                <span>{value:.2f}</span>
                            </span>
                        </div>""",
                        unsafe_allow_html=True
                    )
    
    top_k = st.sidebar.number_input("Number of top samples to display", min_value=1, max_value=20, value=5, step=1)
    
    top_indices, top_values = get_top_features_for_latent(data_array, dimension_idx, top_k)
    
    st.session_state['top_activation_indices'] = top_indices
    
    bottom_indices, bottom_values = get_top_features_for_latent(data_array, dimension_idx, top_k, ascending=True)
    
    has_token_data = token_data is not None and token_activations is not None
    
    st.subheader(f"Top {top_k} highest activating samples for latent dimension {dimension_idx + 1}")
    
    if original_texts is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Top {top_k} highest activating samples")
            for i, (idx, val) in enumerate(zip(top_indices, top_values)):
                if idx < len(original_texts):
                    with st.expander(f"Sample {i+1} - Activation: {val:.3f}"):
                        original_text = original_texts[idx]
                        relevant_part = extract_relevant_text(original_text)
                        
                        cleaned_text = clean_text(original_text)
                        st.code(cleaned_text, language=None)
                        
                        if relevant_part:
                            cleaned_relevant_part = clean_text(relevant_part)
                            st.markdown("<hr>", unsafe_allow_html=True)
                            st.markdown("**Extracted relevant part:**", unsafe_allow_html=True)
                            st.markdown(f"<div style='color: yellow; font-family: monospace; font-size: 0.9em;'>{cleaned_relevant_part}</div>", 
                                        unsafe_allow_html=True)
                        
                        if has_token_data and idx < len(token_data):
                            st.markdown("<hr>", unsafe_allow_html=True)
                            
                            sample_tokens, sample_latent_activations = process_sample_token_activations(
                                token_data, token_activations, idx, dimension_idx
                            )
                            
                            if sample_tokens is not None and sample_latent_activations is not None:
                                fig = create_activation_histogram(sample_tokens, sample_latent_activations)
                                st.plotly_chart(fig, use_container_width=True, key=f"main_hist_sample_{idx}_latent_{selected_latent}")

                                if sample_latent_activations is not None:
                                    st.markdown("<hr>", unsafe_allow_html=True)
                                    st.markdown("**Color-coded token activations:**", unsafe_allow_html=True)
                                    html_content = create_colored_token_html(sample_tokens, sample_latent_activations)
                                    st.markdown(html_content, unsafe_allow_html=True)
                                    st.caption("Tokens are colored by activation intensity (darker red = higher activation). Hover over tokens to see exact scores.")
                                
                                display_token_details(sample_tokens, sample_latent_activations)
                else:
                    st.write(f"Sample index {idx} is out of range for available texts")
        
        with col2:
            st.subheader(f"Top {top_k} lowest activating samples")
            for i, (idx, val) in enumerate(zip(bottom_indices, bottom_values)):
                if idx < len(original_texts):
                    with st.expander(f"Sample {i+1} - Activation: {val:.3f}"):
                        original_text = original_texts[idx]
                        relevant_part = extract_relevant_text(original_text)
                        
                        cleaned_text = clean_text(original_text)
                        st.code(cleaned_text, language=None)
                        
                        if relevant_part:
                            cleaned_relevant_part = clean_text(relevant_part)
                            st.markdown("<hr>", unsafe_allow_html=True)
                            st.markdown("**Extracted relevant part:**", unsafe_allow_html=True)
                            st.markdown(f"<div style='color: yellow; font-family: monospace; font-size: 0.9em;'>{cleaned_relevant_part}</div>", 
                                        unsafe_allow_html=True)
                        
                        if has_token_data and idx < len(token_data):
                            st.markdown("<hr>", unsafe_allow_html=True)
                            
                            sample_tokens, sample_latent_activations = process_sample_token_activations(
                                token_data, token_activations, idx, dimension_idx
                            )
                            
                            if sample_tokens is not None and sample_latent_activations is not None:
                                fig = create_activation_histogram(sample_tokens, sample_latent_activations)
                                st.plotly_chart(fig, use_container_width=True, key=f"bottom_hist_sample_{idx}_latent_{selected_latent}")

                                if sample_latent_activations is not None:
                                    st.markdown("<hr>", unsafe_allow_html=True)
                                    st.markdown("**Color-coded token activations:**", unsafe_allow_html=True)
                                    html_content = create_colored_token_html(sample_tokens, sample_latent_activations)
                                    st.markdown(html_content, unsafe_allow_html=True)
                                    st.caption("Tokens are colored by activation intensity (darker red = higher activation). Hover over tokens to see exact scores.")
                                
                                display_token_details(sample_tokens, sample_latent_activations)
                else:
                    st.write(f"Sample index {idx} is out of range for available texts")
    else:
        st.warning("No original texts available to display samples")
    
    st.markdown("---")
    st.markdown("**Note:** Each sample shows the text with the highest/lowest activation for the selected latent dimension.")

def display_token_details(tokens, activations):
    """Display detailed token activation information in a table."""
    st.markdown("**Token activation details:**")
    
    token_table = pd.DataFrame({
        "Token": [clean_token(token) for token in tokens],
        "Activation Score": activations,
        "Position": range(len(tokens))
    })
    
    sorted_table = token_table.sort_values("Activation Score", ascending=False)
    
    max_positive = max(max(activations), 0.001) if any(v > 0 for v in activations) else 1.0
    min_negative = min(min(activations), -0.001) if any(v < 0 for v in activations) else -1.0
    
    def highlight_activations(val):
        if isinstance(val, (int, float)):
            if val > 0:
                intensity = min(val / max_positive * 0.9, 0.9)
                return f'background-color: rgba(255, 0, 0, {intensity})'
            elif val < 0:
                intensity = min(abs(val / min_negative) * 0.9, 0.9)
                return f'background-color: rgba(0, 0, 255, {intensity})'
        return ''
    
    styled_table = sorted_table.style.applymap(
        highlight_activations, subset=['Activation Score']
    ).format({"Activation Score": "{:.4f}"})
    
    st.dataframe(styled_table, use_container_width=True)
    
    st.caption("Color coding: Red indicates tokens with positive activation scores (strongly activating this latent), " 
            "blue indicates negative scores (inhibiting this latent), white indicates neutral scores.")