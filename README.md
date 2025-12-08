# Wikidata Query Logs KGQA

Wikidata Query Logs KGQA (WDQL) is SPARQL-based knowledge graph question answering dataset over the [Wikidata knowledge graph](https://www.wikidata.org) built from [Wikidata query logs](https://iccl.inf.tu-dresden.de/web/Wikidata_SPARQL_Logs/en).

## Embedding & Visualization Pipeline

This repository includes tools to generate embeddings for query-SPARQL samples and visualize them interactively.

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Generate Embeddings

Generate embeddings for all query samples using Qwen3-Embedding-0.6B (averages all question variations per sample):

```bash
python generate_embeddings.py
```

This will:
- Load all JSON files from `organic-qwen3-next-80b-a3b/`
- Generate embeddings by averaging question variations
- Save results to `embeddings_output/`:
  - `embeddings.npy` - NumPy array of embeddings
  - `metadata.pkl` - Pickle file with sample metadata
  - `summary.json` - Summary statistics

#### 2. Build Index & Cluster

Create usearch index and apply clustering:

```bash
python build_clusters.py
```

This will:
- Load embeddings from `embeddings_output/`
- Build usearch index for k-NN search
- Apply HDBSCAN clustering
- Reduce dimensions with UMAP for visualization
- Save clustered data to `embeddings_output/clusters.pkl`

#### 3. Launch Interactive Visualization

Start the Streamlit app:

```bash
streamlit run visualize_app.py
```

Features:
- Interactive 2D scatter plot of embeddings
- Color-coded by cluster
- Hover to preview questions/SPARQL
- Click to view full sample details
- Filter by cluster, query complexity, etc.
