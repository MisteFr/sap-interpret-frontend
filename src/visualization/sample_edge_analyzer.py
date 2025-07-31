import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

def display_sample_edge_analyzer(edge_data, original_texts):
    """
    Display a sample-level analyzer for edge violations.
    
    Args:
        edge_data: The edge violation data array (samples x edges)
        original_texts: The original text samples
        tokenizer: The tokenizer to use for text processing
    """
    st.header("Sample-Level Edge Violation Analysis")
    
    max_samples = edge_data.shape[0] - 1
    selected_sample_idx = st.slider("Select sample:", 
                                   min_value=0, 
                                   max_value=max_samples, 
                                   value=0)
    
    if original_texts is not None and len(original_texts) > 0 and selected_sample_idx < len(original_texts):
        with st.expander("Full Sample Text", expanded=True):
            st.write(original_texts[selected_sample_idx])
    
    sample_violations = edge_data[selected_sample_idx]
    
    sort_by = st.radio("Sort edges by:", ["Violation Magnitude", "Edge Index"])
    
    total_edges = len(sample_violations)
    violated_edges = np.sum(sample_violations > 0)
    violation_percentage = (violated_edges / total_edges) * 100 if total_edges > 0 else 0
    
    st.subheader("Edge Violation Statistics")
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    with stats_col1:
        st.metric("Total Edges", total_edges)
    with stats_col2:
        st.metric("Violated Edges", violated_edges)
    with stats_col3:
        st.metric("Violation Percentage", f"{violation_percentage:.2f}%")
    
    edge_df = pd.DataFrame({
        'Edge Index': range(len(sample_violations)),
        'Violation Magnitude': sample_violations
    })
    
    if sort_by == "Violation Magnitude":
        edge_df = edge_df.sort_values('Violation Magnitude', ascending=False)
    
    edge_df['Is Violated'] = edge_df['Violation Magnitude'] > 0
    edge_df['Color'] = edge_df['Is Violated'].apply(lambda x: 'red' if x else 'blue')
    
    st.subheader("Edge Violations")
    
    if sort_by == "Edge Index":
        x_values = edge_df['Edge Index']
        x_title = 'Edge Index'
    else:
        x_values = list(range(len(edge_df)))
        x_title = 'Sorted Position'
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=x_values,
        y=edge_df['Violation Magnitude'],
        marker_color=edge_df['Color'],
        hovertemplate='<b>Edge Index:</b> %{customdata}<br><b>Violation:</b> %{y:.4f}<extra></extra>',
        customdata=edge_df['Edge Index']
    ))
    
    # Add a line at y=0
    fig.add_shape(
        type="line",
        x0=min(x_values),
        y0=0,
        x1=max(x_values),
        y1=0,
        line=dict(color="black", width=1, dash="solid"),
    )
    
    fig.update_layout(
        title=f'Edge Violations for Sample {selected_sample_idx}',
        xaxis=dict(
            title='',
            showticklabels=False,
            showgrid=False,
        ),
        yaxis=dict(
            title='Violation Magnitude',
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial",
            font_color="black",
            bordercolor="darkgray",
            namelength=-1
        ),
        height=500,
    )
    
    st.plotly_chart(fig, use_container_width=True) 