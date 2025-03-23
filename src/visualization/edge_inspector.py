import streamlit as st
import pandas as pd
import numpy as np
from utils.analysis import get_top_violations_for_edge, get_edge_violation_stats, process_sample_token_violations
from utils.text_processing import extract_relevant_text, clean_text, clean_token
from visualization.plotting import create_violation_histogram, create_colored_token_html
import plotly.express as px

def display_edge_inspector(data_array, original_texts, token_data=None, token_violations=None):
    """Display the edge violation inspector UI."""
    st.markdown(
        """
        This interface lets you explore edge violations by selecting an edge 
        and viewing the samples that maximally violate that edge constraint.
        """
    )
    
    n_edges = data_array.shape[1]
    st.sidebar.write(f"Total edges: {n_edges}")
    
    edge_stats = get_edge_violation_stats(data_array)
    
    with st.sidebar:
        with st.expander("📊 Edge Violation Statistics", expanded=False):
            # Sort by violation count descending
            sorted_stats = edge_stats.sort_values('violation_count', ascending=False)
            
            # Format the dataframe for display
            display_df = sorted_stats.copy()
            display_df['violation_percentage'] = display_df['violation_percentage'].round(2)
            display_df['max_violation'] = display_df['max_violation'].round(3)
            display_df['mean_violation'] = display_df['mean_violation'].round(3)
            
            # Rename columns for better display
            display_df.columns = ['Edge', 'Violations', '% of Samples', 'Max Score', 'Mean Score']
            
            # Display as interactive table
            selected_rows = st.data_editor(
                display_df,
                hide_index=True,
                use_container_width=True,
                disabled=True,
                key='edge_stats'
            )
            
            if selected_rows is not None and len(selected_rows) > 0:
                # Update the edge selection based on clicked row
                edge_display = selected_rows['Edge'].iloc[0]
                st.session_state['edge_selection'] = edge_display
    
    edge_display = st.slider("Select edge", 1, n_edges, 
                           value=st.session_state.get('edge_selection', 1))
    edge_idx = edge_display - 1
    
    st.session_state['current_edge_idx'] = edge_idx
    st.session_state['current_edge_display'] = edge_display
    
    neg_col, pos_col, density_col = st.columns(3)
    
    edge_violations = data_array[:, edge_idx].flatten()
    
    if token_data is not None and token_violations is not None:
        all_token_violations = []
        
        for sample_idx in range(len(token_data)):
            _, sample_edge_violations = process_sample_token_violations(
                token_data, token_violations, sample_idx, edge_idx
            )
            
            if sample_edge_violations is not None:
                all_token_violations.extend(sample_edge_violations)
        
        if all_token_violations:
            token_violations_array = np.array(all_token_violations)
            non_zero_violations = token_violations_array[np.abs(token_violations_array) > 1e-6]
            violation_density = (len(non_zero_violations) / len(token_violations_array)) * 100
            edge_violations = token_violations_array
        else:
            non_zero_violations = edge_violations[np.abs(edge_violations) > 1e-6]
            violation_density = (len(non_zero_violations) / len(edge_violations)) * 100
    else:
        non_zero_violations = edge_violations[np.abs(edge_violations) > 1e-6]
        violation_density = (len(non_zero_violations) / len(edge_violations)) * 100
    
    # Display violation distribution in the third column
    # How frequently different violation score values occur
    with density_col:
        st.markdown(f"# Distribution of Violation Scores", unsafe_allow_html=True)
        
        # Randomly sample non-zero violations if there are too many
        max_samples = 10000
        if len(non_zero_violations) > max_samples:
            sampled_violations = np.random.choice(non_zero_violations, size=max_samples, replace=True)
        else:
            sampled_violations = non_zero_violations
        
        fig1 = px.histogram(
            pd.DataFrame({
                "Violation Score": sampled_violations
            }), 
            x="Violation Score",
            nbins=50,
            title="",
            color_discrete_sequence=["rgba(253, 208, 162, 0.8)"]  # Orange/peach color
        )
        
        # Add a vertical line at 0 as a reference point
        fig1.add_vline(
            x=0, 
            line_dash="dash", 
            line_color="gray", 
            annotation_text=""
        )
        
        # Update layout for first histogram
        fig1.update_layout(
            plot_bgcolor="white",
            height=300,
            margin=dict(l=40, r=40, t=10, b=30)
        )
        
        # Display the first histogram
        st.plotly_chart(fig1, use_container_width=True)
        
    
    # Process token data to calculate and display top positive and negative logits
    if token_data is not None and token_violations is not None:
        # Gather all token violations for this edge
        all_tokens = []
        all_token_violations = []
        
        # Process all samples to extract token violations for this edge
        for sample_idx in range(len(token_data)):
            sample_tokens, sample_edge_violations = process_sample_token_violations(
                token_data, token_violations, sample_idx, edge_idx
            )
            
            if sample_tokens is not None and sample_edge_violations is not None:
                all_tokens.extend(sample_tokens)
                all_token_violations.extend(sample_edge_violations)
        
        token_avg_violations = {}
        token_counts = {}
        
        for token, violation in zip(all_tokens, all_token_violations):
            clean_tok = clean_token(token)
            if clean_tok not in token_avg_violations:
                token_avg_violations[clean_tok] = 0
                token_counts[clean_tok] = 0
            
            token_avg_violations[clean_tok] += violation
            token_counts[clean_tok] += 1
        
        for token in token_avg_violations:
            token_avg_violations[token] /= token_counts[token]
        
        filtered_tokens = {token: value for token, value in token_avg_violations.items() 
                          if token_counts.get(token, 0) >= 20}
        
        sorted_tokens = sorted(filtered_tokens.items(), key=lambda x: x[1])
        
        display_count = 15
        
        top_negative = sorted_tokens[:min(display_count, len(sorted_tokens)//2)]
        
        top_positive = sorted_tokens[max(len(sorted_tokens)-display_count, len(sorted_tokens)//2):][::-1]
        
        with neg_col:
            st.markdown(f"# Negative Logits")
            for token, value in top_negative:
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
        
        with pos_col:
            st.markdown(f"# Positive Logits")
            for token, value in top_positive:
                if value > 0:  # Only display positive values
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
    
    
    top_k = st.number_input("Number of top violating samples to display", min_value=1, max_value=20, value=5, step=1)
    
    top_indices, top_violations = get_top_violations_for_edge(data_array, edge_idx, top_k)
    
    st.session_state['top_violation_indices'] = top_indices
    
    bottom_indices, bottom_violations = get_top_violations_for_edge(data_array, edge_idx, top_k, ascending=True)
    
    positive_violations = data_array[:, edge_idx] > 0
    positive_count = np.sum(positive_violations)
    st.write(f"**{positive_count}** out of **{data_array.shape[0]}** samples violate this edge constraint")
    
    edge_data = data_array[:, edge_idx]
    st.write(f"Edge statistics - Min: {edge_data.min():.3f}, Max: {edge_data.max():.3f}, Mean: {edge_data.mean():.3f}")
    
    has_token_data = token_data is not None and token_violations is not None
    
    if original_texts is not None:
        col1, col2 = st.columns(2)
        
        # Left column: highest violating samples
        with col1:
            st.subheader(f"Top {top_k} highest violating samples")
            for i, (idx, violation) in enumerate(zip(top_indices, top_violations)):
                if idx < len(original_texts):
                    with st.expander(f"Sample {i+1} - Violation score: {violation:.3f}"):
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
                            
                            sample_tokens, sample_edge_violations = process_sample_token_violations(
                                token_data, token_violations, idx, edge_idx
                            )
                            
                            if sample_tokens is not None and sample_edge_violations is not None:
                                fig = create_violation_histogram(sample_tokens, sample_edge_violations)
                                st.plotly_chart(fig, use_container_width=True, key=f"main_hist_sample_{idx}_edge_{edge_display}")

                                if sample_edge_violations is not None:
                                    st.markdown("<hr>", unsafe_allow_html=True)
                                    st.markdown("**Color-coded token violations:**", unsafe_allow_html=True)
                                    html_content = create_colored_token_html(sample_tokens, sample_edge_violations)
                                    st.markdown(html_content, unsafe_allow_html=True)
                                    st.caption("Tokens are colored by violation intensity (darker red = higher violation). Hover over tokens to see exact scores.")
                                
                                display_token_details(sample_tokens, sample_edge_violations)
                else:
                    st.write(f"Sample index {idx} is out of range for available texts")
        
        # Right column: lowest violating samples  
        with col2:
            st.subheader(f"Top {top_k} lowest violating samples")
            for i, (idx, violation) in enumerate(zip(bottom_indices, bottom_violations)):
                if idx < len(original_texts):
                    with st.expander(f"Sample {i+1} - Violation score: {violation:.3f}"):
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
                            
                            sample_tokens, sample_edge_violations = process_sample_token_violations(
                                token_data, token_violations, idx, edge_idx
                            )
                            
                            if sample_tokens is not None and sample_edge_violations is not None:
                                fig = create_violation_histogram(sample_tokens, sample_edge_violations)
                                st.plotly_chart(fig, use_container_width=True, key=f"bottom_hist_sample_{idx}_edge_{edge_display}")

                                if sample_edge_violations is not None:
                                    st.markdown("<hr>", unsafe_allow_html=True)
                                    st.markdown("**Color-coded token violations:**", unsafe_allow_html=True)
                                    html_content = create_colored_token_html(sample_tokens, sample_edge_violations)
                                    st.markdown(html_content, unsafe_allow_html=True)
                                    st.caption("Tokens are colored by violation intensity (darker red = higher violation). Hover over tokens to see exact scores.")
                                
                                display_token_details(sample_tokens, sample_edge_violations)
                else:
                    st.write(f"Sample index {idx} is out of range for available texts")
    else:
        st.warning("No original texts available to display samples")
    
    st.markdown("---")
    st.markdown("**Note:** Each sample shows the text with the highest/lowest violation score for the selected edge constraint.")

def display_token_details(tokens, violations):
    """Display detailed token violation information in a table."""
    st.markdown("**Token violation details:**")
    
    token_table = pd.DataFrame({
        "Token": [clean_token(token) for token in tokens],
        "Violation Score": violations,
        "Position": range(len(tokens))
    })
    
    sorted_table = token_table.sort_values("Violation Score", ascending=False)
    
    max_positive = max(max(violations), 0.001) if any(v > 0 for v in violations) else 1.0
    min_negative = min(min(violations), -0.001) if any(v < 0 for v in violations) else -1.0
    
    def highlight_violations(val):
        if isinstance(val, (int, float)):
            if val > 0:
                intensity = min(val / max_positive * 0.9, 0.9)  # Scale based on max positive
                return f'background-color: rgba(255, 0, 0, {intensity})'
            elif val < 0:
                intensity = min(abs(val / min_negative) * 0.9, 0.9)  # Scale based on min negative
                return f'background-color: rgba(0, 0, 255, {intensity})'
        return ''
    
    styled_table = sorted_table.style.applymap(
        highlight_violations, subset=['Violation Score']
    ).format({"Violation Score": "{:.4f}"})
    
    st.dataframe(styled_table, use_container_width=True)
    
    st.caption("Color coding: Red indicates tokens with positive violation scores (potentially unsafe), " 
            "blue indicates negative scores (likely safe), white indicates neutral scores.")