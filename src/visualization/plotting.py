import pandas as pd
import plotly.express as px
import torch
import numpy as np
from utils.text_processing import clean_token
import html  # Add this for HTML escaping

def visualize_token_violations(tokens, violations, edge_idx):
    """Create a visualization of token-level violations for a specific edge."""
    if tokens is None or violations is None:
        return None
    
    try:
        if isinstance(violations, torch.Tensor):
            edge_violations = violations[:, edge_idx].numpy()
        elif isinstance(violations, np.ndarray):
            if violations.ndim == 2:
                edge_violations = violations[:, edge_idx]
            else:
                edge_violations = violations
        else:
            return None
        
        string_tokens = [str(t) for t in tokens]
        
        token_df = pd.DataFrame({
            "token": string_tokens,
            "violation": edge_violations,
            "position": range(len(string_tokens)),
            "token_text": string_tokens
        })
        
        token_df["hover_text"] = token_df.apply(
            lambda row: f"Token: {row['token']}<br>Violation: {row['violation']:.4f}", 
            axis=1
        )
        
        fig = px.bar(
            token_df,
            x="violation",
            y="position",
            hover_data=["token_text", "violation"],
            custom_data=["token", "violation"],
            labels={"violation": "Violation Score", "position": "Token Position"},
            orientation='h',
            height=max(400, len(string_tokens) * 15)
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color="red", 
                    annotation_text="Safety Threshold")
        
        fig.update_layout(yaxis=dict(autorange="reversed"))
        
        fig.update_traces(
            hovertemplate="<b>Token:</b> %{customdata[0]}<br><b>Violation:</b> %{customdata[1]:.4f}<extra></extra>"
        )
        
        return fig
    except Exception as e:
        print(f"Error in visualize_token_violations: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_violation_histogram(tokens, violations):
    """Create a histogram of token violation scores."""
    if tokens is None or violations is None:
        return None
        
    fig = px.histogram(
        pd.DataFrame({
            "Token": [clean_token(token) for token in tokens],
            "Violation Score": violations
        }), 
        x="Violation Score",
        nbins=50,
        title="Distribution of Token Violation Scores"
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", 
                annotation_text="Safety Threshold")
    
    return fig

def create_colored_token_html(tokens, scores):
    """Create HTML with colored tokens based on violation scores.
    
    Uses a color scale where:
    - Positive scores: white → red (higher = brighter red)
    - Negative scores: white → blue (lower = brighter blue)
    - Scores near 0: minimal or no color
    """
    if tokens is None or scores is None:
        return ""
    
    scores_array = np.array(scores)
    max_positive = np.max(scores_array) if np.any(scores_array > 0) else 1.0
    min_negative = np.min(scores_array) if np.any(scores_array < 0) else -1.0
    
    html_parts = []
    for token, score in zip(tokens, scores):
        # Properly escape the token to prevent HTML interpretation
        cleaned_token = html.escape(clean_token(token))
        
        if score > 0:
            # Scale from white to red for positive scores
            intensity = min(score / max_positive * 0.9, 0.9)  # Scale based on max positive
            html_parts.append(f'<span title="Score: {score:.4f}" style="background-color: rgba(255, 0, 0, {intensity}); padding: 2px; border-radius: 3px; margin: 0px; display: inline-block;">{cleaned_token}</span>')
        elif score < 0:
            # Scale from white to blue for negative scores
            intensity = min(abs(score / min_negative) * 0.9, 0.9)  # Scale based on min negative
            html_parts.append(f'<span title="Score: {score:.4f}" style="background-color: rgba(0, 0, 255, {intensity}); padding: 2px; border-radius: 3px; margin: 0px; display: inline-block;">{cleaned_token}</span>')
        else:
            html_parts.append(f'<span title="Score: {score:.4f}" style="margin: 0px; display: inline-block;">{cleaned_token}</span>')
    
    return '<div style="font-family: monospace; font-size: 0.9em; line-height: 1.8em; overflow-wrap: anywhere;">' + ''.join(html_parts) + '</div>' 

def create_activation_histogram(tokens, activations):
    """Create a histogram of token activation scores."""
    if tokens is None or activations is None:
        return None
        
    fig = px.histogram(
        pd.DataFrame({
            "Token": [clean_token(token) for token in tokens],
            "Activation Score": activations
        }), 
        x="Activation Score",
        nbins=50,
        title="Distribution of Token Activation Scores"
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray", 
                annotation_text="Zero Activation")
    
    return fig 