# Wikidata Query Logs KGQA

Wikidata Query Logs KGQA (WDQL) is SPARQL-based knowledge graph question answering dataset over the [Wikidata knowledge graph](https://www.wikidata.org) built from [Wikidata query logs](https://iccl.inf.tu-dresden.de/web/Wikidata_SPARQL_Logs/en).

## Embedding & Visualization Pipeline

This repository includes tools to generate embeddings for query-SPARQL samples and visualize them interactively.

### Setup

Install dependencies:
```bash
pip install -e .
```

### Development

Code quality tools are available via the Makefile:
```bash
make format       # Format code with ruff
make check        # Check code style with ruff
make type-check   # Run mypy and pyright type checking
make clean        # Remove cache files
```

### Usage

#### 1. Generate Embeddings

Generate embeddings for all query samples using Qwen3-Embedding-0.6B (averages all question variations per sample):

```bash
python generate_embeddings.py
```

This will:
- Load all JSON files from `data/organic-qwen3-next-80b-a3b/`
- **Check validity** of each sample according to `VALIDITY_RULES.md`
- Generate **L2-normalized embeddings** by averaging question variations
- Save results to `data/organic-qwen3-next-80b-a3b/embeddings/`:
  - `embeddings.npy` - NumPy array with all embeddings
  - `metadata.json` - JSON file with sample metadata (includes validity info)
  - `summary.json` - Summary statistics

Optional arguments:
```bash
python generate_embeddings.py --batch-size 256 --input-dir path/to/data --output-dir custom_output
```

#### 2. Build Clusters & Visualization

Apply UMAP + HDBSCAN clustering and generate 2D visualization:

```bash
python build_clusters.py
```

This will:
- Load embeddings from `data/organic-qwen3-next-80b-a3b/embeddings/`
- Apply **UMAP dimensionality reduction** (to 50d, cosine metric) for better clustering
- **Cluster valid samples with HDBSCAN** (invalid samples assigned cluster -1)
- Assign noise points to singleton clusters (preserves rare samples)
- Apply **UMAP for 2D visualization** (separate from clustering)
- Save results:
  - `cluster_labels.json` - Cluster assignment for each sample (-1 for invalid)
  - `umap_coords.json` - 2D coordinates for visualization
  - `cluster_stats.json` - Cluster statistics

Optional arguments:
```bash
# HDBSCAN parameters (control cluster density)
python build_clusters.py --min-cluster-size 2 --min-samples 1

# UMAP parameters for clustering (before HDBSCAN)
python build_clusters.py --umap-n-components 50 --umap-n-neighbors 30

# UMAP parameters for 2D visualization
python build_clusters.py --vis-umap-neighbors 15 --vis-umap-min-dist 0.1
```

**Note:** HDBSCAN uses a two-stage approach:
1. Dense clusters are found using HDBSCAN
2. Noise points (rare/isolated samples) are assigned to singleton clusters to preserve them for analysis

#### 3. Launch Interactive Visualization

Start the Streamlit app:

```bash
streamlit run visualize_app.py
```

Features:
- **Interactive 2D scatter plot** of UMAP-projected embeddings with 26 distinct colors
- **Click-to-select** samples directly from the visualization
- **Dropdown selector** with "idx: question" format, filtered by validity
- **Random sample button** for exploration
- **Validity filters** (All/Valid Only/Invalid Only) that update the sample dropdown
- **Cluster filters** with counts, sorted by cluster size
- **Sample details** with parsed sections:
  - Questions (markdown)
  - SPARQL query (syntax-highlighted code block)
  - Entities and execution results (markdown)
  - Invalid reason display for invalid samples
- **Cluster statistics** showing:
  - Algorithm name and total clusters
  - 5 largest and 5 smallest clusters in side-by-side tables
- **Validity breakdown** with pie chart and detailed reasons

The webapp automatically loads data from `data/organic-qwen3-next-80b-a3b/embeddings/`

**Note:** Cluster -1 indicates invalid samples that were excluded from clustering

## Validity Checking

All samples are automatically validated according to `VALIDITY_RULES.md`. Invalid samples:
- Are **included in embeddings and visualization** (cluster -1)
- Are **excluded from clustering** to ensure quality clusters
- Can be **filtered in the webapp** for analysis
