# Safety Polytope – Latent Space & Edge Violations Explorer

An interactive Streamlit dashboard for exploring latent-space activations and Safety Polytope facet (edge) violations in language models.

## Overview

[Safety Polytope](https://github.com/lasgroup/SafetyPolytope/tree/main) (SaP) is a geometric safety layer that defines a set of half-space constraints ("facets") in the hidden-state representation space of a language model ([Learning Safety Constraints for Large Language Models](https://arxiv.org/abs/2505.24445) the technical details).  

This dashboard lets you inspect how your data interacts with those constraints:

• **Latent Dimension Explorer** – browse features of the SaP Concept Encoder and view examples that maximally activate them.  
• **Facet / Edge Violation Inspector** – identify inputs that violate safety facets and drill down to token level.  
• **Token-level Heat-maps** – see which tokens contribute most to each violation.

## Quick Start

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the app:
   ```bash
   streamlit run src/app.py
   ```
3. Open <http://localhost:8501> in your browser.



## Generating Compatible Data with `sap-interpreter`

The dashboard consumes up to **three NumPy `.npz` archives**:

* `latent_activations.npz` – latent activations (sample- or token-level)
* `edge_violations.npz` – facet/edge violation scores (one row per sample)
* `token_level_violations.npz` – *optional* per-token violation scores

You can generate these files with the companion toolkit [`sap-interpreter`](https://github.com/MisteFr/sap-interpreter):

```bash
# install from GitHub (or "pip install sap-interpreter" once released)
pip install git+https://github.com/MisteFr/sap-interpreter.git

# 1) compute facet violations (edge-level & token-level)
compute-edge-violations \
  --model_path /path/to/model \
  --trained_weights_path /path/to/weights \
  --hidden_states_path /path/to/hidden_states \
  --output_dir outputs \
  --token_level            # optional – also save per-token violations

# 2) extract latent activations from the SaP Concept Encoder
extract-sae-activations \
  --model_path /path/to/model \
  --trained_weights_path /path/to/weights \
  --hidden_states_path /path/to/hidden_states \
  --output_dir outputs \
  --token_level
```

The commands above create `edge_violations.npz`, `token_level_violations.npz`, and `latent_activations.npz` in `outputs/`.  
Drop those files into `sap-interpret-frontend/` (or point the `--data_dir` flag once implemented) and relaunch the dashboard.


## Preview

### Edge Violation Inspector
![Edge Violation Inspector](screenshots/screen1.png)
*A view of the Edge Violation Inspector tab showing negative vs positive violation logits for a selected facet, token-level tooltips, and the distribution histogram.*

### Token-level Heat-map
![Token-level Heat-map](screenshots/screen2.png)
*Tokens are color-coded by violation intensity in a selected sample, with hoverable tooltips for exact scores.*

### Sample Edge Analysis
![Sample Edge Analysis](screenshots/screen5.png)
*Bar chart showing the magnitude of violations per token in a single sample.*

### Edge Correlation Heat-map
![Edge Correlation Heat-map](screenshots/screen0.png)
*Heat-map displaying pairwise correlations between edge violations across all facets.*