# Wikidata Query Logs Dataset

A dataset of SPARQL-question pairs built from
[Wikidata Query Logs](https://iccl.inf.tu-dresden.de/web/Wikidata_SPARQL_Logs/en)
from 2017 to 2018.

## Downloads

Pre-generated files are available at https://ad-publications.cs.uni-freiburg.de/wikidata-query-logs:

- `organic.tar.gz` - Prepared SPARQL queries as JSONL
- `organic-qwen3-next-80b-a3b.tar.gz` - Generated question-SPARQL pairs
- `organic-qwen3-next-80b-a3b-dataset.tar.gz` - Dataset with embeddings and clusters
- `wdql-kgqa-dataset.tar.gz` - Deduplicated KGQA dataset (train/val/test splits)
- `items.tar.gz` - Wikidata item identifiers (needed for statistics script)

Download and extract these files into a subdirectory named `data/`
to avoid re-running the entire pipeline.

## Pipeline

### 1. Prepare input from query logs

```bash
python prepare_input.py logs/*.tsv data/
```

### 2. Generate question-SPARQL pairs

TODO

### 3. Generate dataset and embeddings

```bash
python generate_dataset.py
```

### 4. Build clusters from embeddings

```bash
python build_clusters.py
```

### 5. Export KGQA dataset using clusters

```bash
python export_kgqa_dataset.py
```

## Statistics

```bash
cat data/organic-qwen3-next-80b-a3b-dataset/samples.json \
  | jq -c '.[] | select(.valid == true) | .sparql' \
  | python sparql_statistics.py
```

## Visualization

```bash
streamlit run visualize_app.py
```
