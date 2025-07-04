import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
from itertools import combinations

def analyze_edge_groups(data_array, threshold=0):
    """
    Analyze relationships between groups of edges that are highly correlated.
    
    Args:
        data_array: numpy array of shape (n_samples, n_edges) containing violation scores
        threshold: violation threshold for considering a violation
    
    Returns:
        groups: list of edge groups with high correlation
        group_stats: statistics about the relationships between groups
    """
    violations = data_array > threshold
    n_samples = data_array.shape[0]
    
    # Find sets of edges that share many of the same violations
    edge_sets = {}
    for i in range(violations.shape[1]):
        edge_sets[i] = set(np.where(violations[:, i])[0])
    
    # Calculate Jaccard similarity between all pairs
    edge_pairs = []
    for i, j in combinations(range(violations.shape[1]), 2):
        intersection = len(edge_sets[i] & edge_sets[j])
        union = len(edge_sets[i] | edge_sets[j])
        if union > 0:
            similarity = intersection / union
            if similarity > 0.5:  # Only consider highly correlated pairs
                edge_pairs.append((i, j, similarity))
    
    return edge_sets, edge_pairs

def find_active_edges(data_array, threshold=0):
    """
    Find edges that have violations and sort them by violation count.
    
    Args:
        data_array: numpy array of shape (n_samples, n_edges) containing violation scores
        threshold: violation threshold for considering a violation
    
    Returns:
        list of tuples (edge_index, violation_count) sorted by violation count
    """
    violations = data_array > threshold
    violation_counts = np.sum(violations, axis=0)
    active_edges = [(i, count) for i, count in enumerate(violation_counts) if count > 0]
    return sorted(active_edges, key=lambda x: x[1], reverse=True)

def calculate_edge_violation_overlap(data_array, threshold=0):
    """
    Calculate the normalized overlap matrix of violations between edges.
    The overlap score is normalized to be between 0 and 1:
    - 1 means edges i and j are violated by exactly the same samples
    - 0 means edges i and j have no common violating samples
    
    Args:
        data_array: numpy array of shape (n_samples, n_edges) containing violation scores
        threshold: violation threshold (default 0)
    
    Returns:
        overlap_matrix: numpy array of shape (n_edges, n_edges) containing normalized overlap scores
    """
    n_edges = data_array.shape[1]
    overlap_matrix = np.zeros((n_edges, n_edges))
    
    # Create binary violation matrix
    violations = data_array > threshold
    
    # Calculate normalized overlap for each pair of edges
    for i in range(n_edges):
        for j in range(n_edges):
            intersection = np.sum(np.logical_and(violations[:, i], violations[:, j]))
            union = np.sum(np.logical_or(violations[:, i], violations[:, j]))
            # Use Jaccard index (intersection over union) if there are any violations
            overlap_matrix[i, j] = intersection / union if union > 0 else 0
    
    return overlap_matrix, violations

def analyze_edge_violation_relationships(data_array, threshold=0):
    """
    Analyze the relationship between samples violating different edges.
    
    Args:
        data_array: numpy array of shape (n_samples, n_edges) containing violation scores
        threshold: violation threshold for considering a violation
    
    Returns:
        tuple: (edge_rankings, sample_overlaps) where:
            - edge_rankings: list of (edge_idx, violation_count) sorted by violation count
            - sample_overlaps: dict mapping edge pairs to their sample overlap statistics
    """
    violations = data_array > threshold
    n_samples = data_array.shape[0]
    
    # Calculate violation counts for each edge
    violation_counts = np.sum(violations, axis=0)
    edge_rankings = sorted([(i, count) for i, count in enumerate(violation_counts) if count > 0],
                          key=lambda x: x[1], reverse=True)
    
    # Calculate sample overlaps between edges
    sample_overlaps = {}
    for i, (edge1_idx, count1) in enumerate(edge_rankings):
        for edge2_idx, count2 in edge_rankings[i+1:]:
            # Get samples violating each edge
            edge1_samples = set(np.where(violations[:, edge1_idx])[0])
            edge2_samples = set(np.where(violations[:, edge2_idx])[0])
            
            # Calculate overlap statistics
            intersection = len(edge1_samples & edge2_samples)
            union = len(edge1_samples | edge2_samples)
            
            if union > 0:
                jaccard = intersection / union
                overlap_pct1 = (intersection / count1) * 100 if count1 > 0 else 0
                overlap_pct2 = (intersection / count2) * 100 if count2 > 0 else 0
                
                sample_overlaps[(edge1_idx, edge2_idx)] = {
                    'edge1_count': count1,
                    'edge2_count': count2,
                    'intersection': intersection,
                    'jaccard': jaccard,
                    'overlap_pct1': overlap_pct1,
                    'overlap_pct2': overlap_pct2
                }
    
    return edge_rankings, sample_overlaps

def display_edge_correlation_analyzer(data_array):
    """Display the edge violation correlation analysis interface."""
    st.header("Edge Violation Correlation Analysis")
    
    st.markdown("""
    This analysis shows how strongly different edges are correlated in their violations.
    The heatmap shows correlation scores between 0 and 1:
    - Score of 1 means the edges are violated by exactly the same samples
    - Score of 0 means the edges have no common violating samples
    - Intermediate values indicate partial overlap in violating samples
    """)
    
    # Add controls
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider(
            "Violation threshold",
            min_value=float(data_array.min()),
            max_value=float(data_array.max()),
            value=0.0,
            step=0.1,
            help="Minimum violation score to consider as a violation"
        )
    
    # Calculate overlap matrix and get violations
    overlap_matrix, violations = calculate_edge_violation_overlap(data_array, threshold)
    
    # Create DataFrame for the heatmap
    df = pd.DataFrame(
        overlap_matrix,
        index=[f"Edge {i+1}" for i in range(overlap_matrix.shape[0])],
        columns=[f"Edge {i+1}" for i in range(overlap_matrix.shape[1])]
    )
    
    # Create heatmap using plotly
    fig = px.imshow(
        df,
        title="Edge Violation Correlation (Jaccard Index)",
        labels=dict(x="Edge", y="Edge", color="Correlation"),
        color_continuous_scale="Viridis",
        aspect="auto",
        zmin=0,
        zmax=1
    )
    
    # Update layout for better visibility
    fig.update_layout(
        height=800,
        width=800,
        title_x=0.5,
        xaxis_title="Edge",
        yaxis_title="Edge"
    )
    
    # Display the heatmap
    st.plotly_chart(fig, use_container_width=True)
    
    # Add detailed analysis section
    st.subheader("Edge Violation Relationships")
    
    # Get edge rankings and sample overlaps
    edge_rankings, sample_overlaps = analyze_edge_violation_relationships(data_array, threshold)
    
    if edge_rankings:
        # Show relationships between most violated edge and other edges
        st.markdown("#### Relationships Between Most Violated Edge and Other Edges")
        most_violated_edge = edge_rankings[0][0]  # Get the most violated edge index
        
        # Get top 10 edges to compare with
        comparison_edges = [idx for idx, _ in edge_rankings[1:11]]  # Skip the most violated edge itself
        
        relationship_data = []
        for edge2_idx in comparison_edges:
            if (most_violated_edge, edge2_idx) in sample_overlaps:
                stats = sample_overlaps[(most_violated_edge, edge2_idx)]
                relationship_data.append({
                    "Edge Pair": f"Edge {most_violated_edge+1} & Edge {edge2_idx+1}",
                    "Other Edge Count": stats['edge2_count'],
                    "Common Samples": stats['intersection'],
                    "% of Other Edge Samples": f"{stats['overlap_pct2']:.1f}%"
                })
        
        if relationship_data:
            st.table(pd.DataFrame(relationship_data))
        else:
            st.info("No significant relationships found between the most violated edge and other edges.")
    else:
        st.warning("No edges with violations found at the current threshold.") 