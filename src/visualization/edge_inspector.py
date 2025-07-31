import streamlit as st
import pandas as pd
import numpy as np
from utils.analysis import get_top_violations_for_edge, get_edge_violation_stats, process_sample_token_violations, get_token_associations
from utils.text_processing import extract_relevant_text, clean_text, clean_token
from visualization.plotting import create_violation_histogram, create_colored_token_html
import plotly.express as px
import html
from collections import defaultdict

# Cache-heavy aggregation so it's computed only once per edge and cached by Streamlit
@st.cache_data(show_spinner="Aggregating token violation statistics…")
def _get_edge_token_stats(token_data, token_violations, edge_idx):
    """Return (avg_dict, count_dict) for the selected edge."""
    token_sum = defaultdict(float)
    token_count = defaultdict(int)

    for sample_idx in range(len(token_data)):
        sample_tokens, sample_edge_violations = process_sample_token_violations(
            token_data, token_violations, sample_idx, edge_idx
        )
        if sample_tokens is None or sample_edge_violations is None:
            continue

        for tok, v in zip(sample_tokens, sample_edge_violations):
            tok_clean = clean_token(tok)
            token_sum[tok_clean] += v
            token_count[tok_clean] += 1

    token_avg = {t: token_sum[t] / token_count[t] for t in token_sum}
    return token_avg, token_count

def inject_token_tooltip_style():
    """Inject the CSS style for token tooltips."""
    st.markdown("""
        <style>
            .token-row {
                display: flex;
                align-items: center;
                position: relative;
                margin-bottom: 5px;
                width: 100%;
            }
            .token-info-btn {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 24px;
                height: 24px;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.1);
                color: rgba(255, 255, 255, 0.8);
                font-size: 12px;
                margin-left: 8px;
                cursor: help;
                border: none;
                position: relative;
                z-index: 1;
            }
            .token-info-container {
                position: relative;
                display: inline-flex;
                align-items: center;
            }
            .token-info-container:hover .token-tooltip {
                opacity: 1;
                visibility: visible;
                transform: translate(8px, -50%);
            }
            .token-tooltip {
                position: absolute;
                left: 100%;
                top: 50%;
                transform: translate(0, -50%);
                background: #1e1e1e;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                padding: 12px;
                width: 280px;
                z-index: 1000;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
                opacity: 0;
                visibility: hidden;
                transition: all 0.2s ease;
                pointer-events: none;
            }
            .token-tooltip.visible {
                opacity: 1;
                visibility: visible;
                pointer-events: auto;
            }
            .token-section {
                margin-bottom: 12px;
            }
            .token-section:last-child {
                margin-bottom: 0;
            }
            .section-title {
                font-weight: bold;
                color: rgba(255, 255, 255, 0.9);
                margin-bottom: 6px;
                font-size: 0.9em;
            }
            .token-list {
                display: flex;
                flex-direction: column;
                gap: 4px;
                max-height: 150px;
                overflow-y: auto;
            }
            .token-list::-webkit-scrollbar {
                width: 6px;
            }
            .token-list::-webkit-scrollbar-track {
                background: rgba(255, 255, 255, 0.05);
            }
            .token-list::-webkit-scrollbar-thumb {
                background: rgba(255, 255, 255, 0.2);
                border-radius: 3px;
            }
            .token-item {
                color: rgba(255, 255, 255, 0.7);
                font-size: 0.9em;
                padding: 4px 8px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 4px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .token-main {
                background-color: rgba(255, 100, 100, 0.3);
                padding: 4px 10px;
                border-radius: 4px;
                font-size: 0.95em;
            }
        </style>
    """, unsafe_allow_html=True)

def display_edge_inspector(data_array, original_texts, token_data=None, token_violations=None):
    """Display the edge violation inspector UI."""
    # Always inject the tooltip style first
    inject_token_tooltip_style()
    
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
        with st.expander("Edge Violation Statistics", expanded=False):
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
        # Compute token statistics once per edge and reuse them across reruns
        token_avg_violations, token_counts = _get_edge_token_stats(
            token_data, token_violations, edge_idx
        )

        # Sidebar controls
        with st.sidebar:
            with st.expander("Display Settings", expanded=False):
                display_count = st.slider("Number of tokens to display", min_value=1, max_value=50, value=15, step=1)
                min_token_freq = st.slider("Minimum token frequency", min_value=1, max_value=200, value=20, step=1)

        # Filter & sort after user interaction
        filtered_tokens = {token: value for token, value in token_avg_violations.items()
                           if token_counts.get(token, 0) >= min_token_freq}
        sorted_tokens = sorted(filtered_tokens.items(), key=lambda x: x[1])
        
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
            # Reset token display state when edge changes
            current_edge_key = f"edge_{edge_idx}"
            if 'last_edge_key' not in st.session_state or st.session_state.last_edge_key != current_edge_key:
                st.session_state.last_edge_key = current_edge_key
                if 'token_states' in st.session_state:
                    del st.session_state.token_states
            
            st.markdown(f"# Positive Logits")
            
            # Process tokens in batches to prevent memory issues
            batch_size = 10
            for batch_start in range(0, len(top_positive), batch_size):
                batch_end = min(batch_start + batch_size, len(top_positive))
                batch = top_positive[batch_start:batch_end]
                
                for token_idx, (token, value) in enumerate(batch, start=batch_start):
                    if value > 0:  # Only display positive values
                        count = token_counts.get(token, 0)
                        
                        # Create unique key for this token in this edge
                        token_key = f"{current_edge_key}_{token_idx}"
                        
                        # Get token associations with limited sample size
                        top_next, top_prev = get_token_associations(
                            token_data, 
                            token_violations, 
                            token, 
                            edge_idx,
                            n_associations=5,
                            max_samples=1000
                        )
                        
                        # Only show non-empty associations
                        if top_next or top_prev:
                            # Create the tooltip content with proper HTML escaping
                            prev_tokens_html = "".join([
                                f'<div class="token-item">{html.escape(str(t))} ({c}×)</div>'
                                for t, c in top_prev
                            ]) or '<div class="token-item">No frequent previous tokens found</div>'
                            
                            next_tokens_html = "".join([
                                f'<div class="token-item">{html.escape(str(t))} ({c}×)</div>'
                                for t, c in top_next
                            ]) or '<div class="token-item">No frequent next tokens found</div>'
                            
                            col1, col2 = st.columns([4, 1], gap="small")
                            with col1:
                                with st.container():
                                    st.markdown(
                                        f"""<div class="token-row" id="row_{token_key}">
                                            <span class="token-main">{html.escape(str(token))}</span>
                                            <div class="token-info-container">
                                                <button class="token-info-btn" id="btn_{token_key}">i</button>
                                                <div class="token-tooltip" id="tooltip_{token_key}">
                                                    <div class="token-section">
                                                        <div class="section-title">Previous tokens:</div>
                                                        <div class="token-list">
                                                            {prev_tokens_html}
                                                        </div>
                                                    </div>
                                                    <div class="token-section">
                                                        <div class="section-title">Next tokens:</div>
                                                        <div class="token-list">
                                                            {next_tokens_html}
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>""",
                                        unsafe_allow_html=True
                                    )
                            with col2:
                                with st.container():
                                    st.markdown(
                                        f"""<div style="display: flex; gap: 10px; justify-content: flex-end;">
                                            <span title="Frequency">{count}×</span>
                                            <span>{value:.2f}</span>
                                        </div>""",
                                        unsafe_allow_html=True
                                    )
                
                # Force garbage collection after each batch
                import gc
                gc.collect()
    
    top_k = st.number_input("Number of top violating samples to display", min_value=1, max_value=20, value=5, step=1)
    
    # Get unique top violating samples
    top_indices, top_violations = get_top_violations_for_edge(data_array, edge_idx, top_k, original_texts=original_texts)
    st.session_state['top_violation_indices'] = top_indices
    
    # Get unique bottom violating samples
    bottom_indices, bottom_violations = get_top_violations_for_edge(data_array, edge_idx, top_k, ascending=True, original_texts=original_texts)
    
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