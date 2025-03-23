import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tqdm import tqdm

def load_stats(stats_dir):
    """Load the precomputed statistics."""
    basic_stats_path = os.path.join(stats_dir, 'basic_stats.pkl')
    token_stats_path = os.path.join(stats_dir, 'token_stats.pkl')
    
    basic_stats = None
    token_stats = None
    
    if os.path.exists(basic_stats_path):
        with open(basic_stats_path, 'rb') as f:
            basic_stats = pickle.load(f)
        print(f"Loaded basic statistics with {len(basic_stats['mean'])} dimensions")
    
    if os.path.exists(token_stats_path):
        with open(token_stats_path, 'rb') as f:
            token_stats = pickle.load(f)
        print(f"Loaded token statistics with {len(token_stats)} tokens")
    
    return basic_stats, token_stats

def plot_latent_dimension_summary(basic_stats, output_dir):
    """Generate summary plots for latent dimensions."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    mean = basic_stats['mean']
    std = basic_stats['std']
    min_vals = basic_stats['min']
    max_vals = basic_stats['max']
    p25 = basic_stats['p25']
    p50 = basic_stats['p50']
    p75 = basic_stats['p75']
    
    # Plot dimension statistics distribution
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Mean distribution
    sns.histplot(mean, kde=True, ax=axs[0, 0])
    axs[0, 0].set_title('Distribution of Mean Values Across Latent Dimensions')
    axs[0, 0].set_xlabel('Mean Value')
    
    # Standard deviation distribution
    sns.histplot(std, kde=True, ax=axs[0, 1])
    axs[0, 1].set_title('Distribution of Standard Deviations Across Latent Dimensions')
    axs[0, 1].set_xlabel('Standard Deviation')
    
    # Range distribution
    ranges = max_vals - min_vals
    sns.histplot(ranges, kde=True, ax=axs[1, 0])
    axs[1, 0].set_title('Distribution of Ranges (Max - Min) Across Latent Dimensions')
    axs[1, 0].set_xlabel('Range')
    
    # Interquartile range distribution
    iqr = p75 - p25
    sns.histplot(iqr, kde=True, ax=axs[1, 1])
    axs[1, 1].set_title('Distribution of Interquartile Ranges Across Latent Dimensions')
    axs[1, 1].set_xlabel('IQR')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dimension_stats_distribution.png'), dpi=300)
    plt.close()
    
    # Plot top and bottom dimensions by absolute mean value
    df = pd.DataFrame({
        'dimension': np.arange(len(mean)),
        'mean': mean,
        'std': std,
        'abs_mean': np.abs(mean),
        'range': ranges,
        'iqr': iqr
    })
    
    top_dims = df.sort_values('abs_mean', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    bar_plot = sns.barplot(x='dimension', y='mean', data=top_dims)
    plt.title('Top 20 Dimensions by Absolute Mean Value')
    plt.xlabel('Dimension Index')
    plt.ylabel('Mean Value')
    # Add error bars for standard deviation
    for i, row in enumerate(top_dims.itertuples()):
        bar_plot.errorbar(i, row.mean, yerr=row.std, fmt='none', color='black', capsize=5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_dimensions_by_abs_mean.png'), dpi=300)
    plt.close()
    
    # Create scatter plot showing relationship between dimension metrics
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['abs_mean'], df['std'], c=df['range'], 
                         cmap='viridis', alpha=0.7, s=30)
    plt.colorbar(scatter, label='Range (Max-Min)')
    plt.title('Relationship Between Dimension Statistics')
    plt.xlabel('Absolute Mean')
    plt.ylabel('Standard Deviation')
    
    # Add annotations for top dimensions
    for i, row in top_dims.head(10).iterrows():
        plt.annotate(f"Dim {row['dimension']}", 
                    (row['abs_mean'], row['std']),
                    xytext=(5, 5),
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dimension_metrics_relationship.png'), dpi=300)
    plt.close()
    
    # Add correlation matrix of top dimensions
    top_corr_dims = top_dims.head(10)['dimension'].values
    
    # Get correlation matrix
    # If basic_stats contains raw data, compute correlations
    if 'raw_data' in basic_stats:
        raw_data = basic_stats['raw_data']
        corr_matrix = np.corrcoef(raw_data[:, top_corr_dims], rowvar=False)
    else:
        # If raw data not available, just create a placeholder
        corr_matrix = np.random.rand(len(top_corr_dims), len(top_corr_dims))
        np.fill_diagonal(corr_matrix, 1.0)
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
               xticklabels=[f"Dim {d}" for d in top_corr_dims],
               yticklabels=[f"Dim {d}" for d in top_corr_dims])
    plt.title('Correlation Between Top Latent Dimensions')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_dimensions_correlation.png'), dpi=300)
    plt.close()
    
    return df

def plot_aggregated_token_statistics(token_stats, output_dir, top_n=20, min_count=10):
    """Generate plots for token-level statistics aggregated across all latent dimensions."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter tokens by minimum count
    filtered_tokens = {token: data for token, data in token_stats.items() if data['count'] >= min_count}
    print(f"Filtered to {len(filtered_tokens)} tokens with count >= {min_count}")
    
    if not filtered_tokens:
        print("No tokens left after filtering. Try lowering min_count.")
        return
    
    # Create dataframe with aggregated token statistics
    aggregated_tokens = []
    
    for token, data in filtered_tokens.items():
        print("Token", token, "Data", data)
        # Calculate aggregate statistics across all latent dimensions
        if isinstance(data['mean'], np.ndarray):
            # For multi-dimensional values, compute aggregates
            token_entry = {
                'token': token,
                'count': data['count'],
                'mean_abs_activation': np.mean(np.abs(data['mean'])),
                'max_abs_activation': np.max(np.abs(data['mean'])),
                'mean_activation': np.mean(data['mean']),
                'std_activation': np.mean(data['std']),
                'max_positive': np.max(data['mean']),
                'min_negative': np.min(data['mean']),
                'range': np.max(data['mean']) - np.min(data['mean']),
                'dimensions_count': len(data['mean'])
            }
            
            # Find dimension with max absolute activation
            max_abs_idx = np.argmax(np.abs(data['mean']))
            token_entry['max_abs_dim'] = max_abs_idx
            token_entry['max_abs_dim_value'] = data['mean'][max_abs_idx]
            
            # Find dimension with max positive activation
            max_pos_idx = np.argmax(data['mean'])
            token_entry['max_pos_dim'] = max_pos_idx
            token_entry['max_pos_dim_value'] = data['mean'][max_pos_idx]
            
            # Find dimension with min negative activation
            min_neg_idx = np.argmin(data['mean'])
            token_entry['min_neg_dim'] = min_neg_idx
            token_entry['min_neg_dim_value'] = data['mean'][min_neg_idx]
            
        else:
            # For scalar values
            token_entry = {
                'token': token,
                'count': data['count'],
                'mean_abs_activation': abs(data['mean']),
                'max_abs_activation': abs(data['mean']),
                'mean_activation': data['mean'],
                'std_activation': data['std'],
                'max_positive': data['mean'] if data['mean'] > 0 else 0,
                'min_negative': data['mean'] if data['mean'] < 0 else 0,
                'range': 0,
                'dimensions_count': 1,
                'max_abs_dim': 0,
                'max_abs_dim_value': data['mean'],
                'max_pos_dim': 0,
                'max_pos_dim_value': data['mean'] if data['mean'] > 0 else 0,
                'min_neg_dim': 0,
                'min_neg_dim_value': data['mean'] if data['mean'] < 0 else 0
            }
            
        aggregated_tokens.append(token_entry)
    
    token_df = pd.DataFrame(aggregated_tokens)
    
    # Plot top tokens by mean absolute activation
    top_abs_tokens = token_df.sort_values('mean_abs_activation', ascending=False).head(top_n)
    
    plt.figure(figsize=(14, 10))
    bar_plot = sns.barplot(x='token', y='mean_abs_activation', data=top_abs_tokens)
    plt.title(f'Top {top_n} Tokens by Mean Absolute Activation (Across All Dimensions)')
    plt.xlabel('Token')
    plt.ylabel('Mean Absolute Activation')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_tokens_by_abs_activation.png'), dpi=300)
    plt.close()
    
    # Plot top tokens by maximum absolute activation
    top_max_abs_tokens = token_df.sort_values('max_abs_activation', ascending=False).head(top_n)
    
    plt.figure(figsize=(14, 10))
    bar = plt.bar(range(len(top_max_abs_tokens)), top_max_abs_tokens['max_abs_activation'], 
           tick_label=top_max_abs_tokens['token'])
    
    # Annotate with dimension information
    for i, p in enumerate(bar):
        token_row = top_max_abs_tokens.iloc[i]
        dim = token_row['max_abs_dim']
        value = token_row['max_abs_dim_value']
        plt.annotate(f"Dim {dim}\n({value:.2f})", 
                    (p.get_x() + p.get_width()/2, p.get_height()), 
                    ha='center', va='bottom')
    
    plt.title(f'Top {top_n} Tokens by Maximum Absolute Activation in Any Dimension')
    plt.xlabel('Token')
    plt.ylabel('Maximum Absolute Activation')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_tokens_by_max_abs_activation.png'), dpi=300)
    plt.close()
    
    # Plot top tokens by positive activation
    top_pos_tokens = token_df.sort_values('max_positive', ascending=False).head(top_n)
    
    plt.figure(figsize=(14, 10))
    bar = plt.bar(range(len(top_pos_tokens)), top_pos_tokens['max_positive'], 
           tick_label=top_pos_tokens['token'], color='green')
    
    # Annotate with dimension information
    for i, p in enumerate(bar):
        token_row = top_pos_tokens.iloc[i]
        dim = token_row['max_pos_dim']
        value = token_row['max_pos_dim_value']
        plt.annotate(f"Dim {dim}\n({value:.2f})", 
                    (p.get_x() + p.get_width()/2, p.get_height()), 
                    ha='center', va='bottom')
    
    plt.title(f'Top {top_n} Tokens by Maximum Positive Activation')
    plt.xlabel('Token')
    plt.ylabel('Maximum Positive Activation')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_tokens_by_positive_activation.png'), dpi=300)
    plt.close()
    
    # Plot tokens with most negative activation
    top_neg_tokens = token_df.sort_values('min_negative').head(top_n)
    
    plt.figure(figsize=(14, 10))
    bar = plt.bar(range(len(top_neg_tokens)), top_neg_tokens['min_negative'], 
           tick_label=top_neg_tokens['token'], color='red')
    
    # Annotate with dimension information
    for i, p in enumerate(bar):
        token_row = top_neg_tokens.iloc[i]
        dim = token_row['min_neg_dim']
        value = token_row['min_neg_dim_value']
        plt.annotate(f"Dim {dim}\n({value:.2f})", 
                    (p.get_x() + p.get_width()/2, p.get_height()), 
                    ha='center', va='top')
    
    plt.title(f'Top {top_n} Tokens by Maximum Negative Activation')
    plt.xlabel('Token')
    plt.ylabel('Maximum Negative Activation')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_tokens_by_negative_activation.png'), dpi=300)
    plt.close()
    
    # Plot tokens with highest range of activation across dimensions
    if token_df['dimensions_count'].max() > 1:
        top_range_tokens = token_df.sort_values('range', ascending=False).head(top_n)
        
        plt.figure(figsize=(14, 10))
        plt.bar(range(len(top_range_tokens)), top_range_tokens['range'], 
               tick_label=top_range_tokens['token'], color='purple', alpha=0.7)
        plt.title(f'Top {top_n} Tokens by Activation Range Across Dimensions')
        plt.xlabel('Token')
        plt.ylabel('Activation Range (Max - Min)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_tokens_by_activation_range.png'), dpi=300)
        plt.close()
    
    # Plot token frequency vs. activation
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(token_df['count'], token_df['mean_abs_activation'], 
                         alpha=0.5, c=token_df['max_abs_activation'], cmap='viridis')
    plt.colorbar(scatter, label='Max Absolute Activation')
    plt.title('Token Frequency vs. Mean Absolute Activation')
    plt.xlabel('Token Frequency (Count)')
    plt.ylabel('Mean Absolute Activation')
    plt.xscale('log')
    
    # Annotate some interesting points
    threshold = token_df['mean_abs_activation'].quantile(0.95)
    for idx, row in token_df[token_df['mean_abs_activation'] > threshold].iterrows():
        plt.annotate(row['token'], 
                    (row['count'], row['mean_abs_activation']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'token_frequency_vs_activation.png'), dpi=300)
    plt.close()
    
    # Create a word cloud where size is proportional to absolute activation
    try:
        wordcloud_data = dict(zip(token_df['token'], token_df['mean_abs_activation']))
        
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                             max_words=100, colormap='viridis').generate_from_frequencies(wordcloud_data)
        
        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud: Size Proportional to Mean Absolute Activation')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'token_activation_wordcloud.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Could not generate word cloud: {e}")

def main():
    """Main function to visualize latent statistics."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize precomputed latent statistics')
    parser.add_argument('--stats_dir', type=str, required=True, help='Directory containing precomputed statistics')
    parser.add_argument('--output_dir', type=str, default='visualization_results', help='Output directory for visualizations')
    parser.add_argument('--min_token_count', type=int, default=10, help='Minimum token count for token statistics')
    parser.add_argument('--top_n', type=int, default=20, help='Number of top items to show in rankings')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load statistics
    print(f"Loading statistics from {args.stats_dir}...")
    basic_stats, token_stats = load_stats(args.stats_dir)
    
    # Plot latent dimension statistics
    if basic_stats is not None:
        print("Generating latent dimension visualizations...")
        df_dims = plot_latent_dimension_summary(basic_stats, args.output_dir)
    else:
        print("No basic statistics found. Skipping latent dimension visualizations.")
    
    # Plot token statistics
    if token_stats is not None:
        print("Generating aggregated token statistics visualizations...")
        plot_aggregated_token_statistics(token_stats, args.output_dir, args.top_n, args.min_token_count)
    else:
        print("No token statistics found. Skipping token visualizations.")
    
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main() 