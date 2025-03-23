# Latent Space & Edge Violations Explorer

A Streamlit application for visualizing and analyzing latent space activations and edge violations in language models.

## Features

- **Latent Space Inspection**: Explore latent dimensions and view samples that maximally activate them
- **Edge Violation Analysis**: Identify problematic examples that violate safety constraints
- **Token-Level Analysis**: Visualize which specific tokens in a sequence contribute to violations

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application using:

```bash
python -m streamlit run src/app.py
```

### Data Files

The application requires NPZ files with the following formats:

1. **Latent activations** (.npz): Contains arrays with keys:
   - `activations`: Numpy array of shape [n_samples, seq_len, n_features] or [n_samples, n_features]
   - `original_texts`: Optional array with original text samples

2. **Edge violations** (.npz): Contains arrays with keys:
   - `violations`: Numpy array of shape [n_samples, n_edges]
   - `original_texts`: Optional array with original text samples

3. **Token-level violations** (.npz): Contains arrays with keys:
   - `tokens`: Array of token lists for each sample
   - `violations`: Array of violation matrices for each sample
   - `original_texts`: Optional array with original text samples 