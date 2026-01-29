# Wikidata Query Logs Dataset

A dataset of SPARQL-question pairs built from
[Wikidata Query Logs](https://iccl.inf.tu-dresden.de/web/Wikidata_SPARQL_Logs/en)
from 2017 to 2018.

## Downloads

Pre-generated files are available at https://ad-publications.cs.uni-freiburg.de/wikidata-query-logs:

- `organic.tar.gz` - Prepared SPARQL queries as JSONL
- `organic-qwen3-next-80b-a3b.tar.gz` - Generated question-SPARQL pairs
- `organic-qwen3-next-80b-a3b-dataset.tar.gz` - Raw dataset with embeddings and clusters
- `wdql.tar.gz` - Deduplicated KGQA dataset (train/val/test splits)
- `wdql-full.tar.gz` - Full KGQA dataset (train/val/test splits)
- `wikidata-benchmarks.tar.gz` - Other Wikidata benchmarks (for comparison)

Download and extract all of these files into a subdirectory named `data/` to skip
some or all of the steps below.

## Pipeline

### 1. Prepare input from query logs

```bash
python prepare_input.py data/*.tsv data/
```

### 2. Generate question-SPARQL pairs with GRASP

```bash
# Checkout wdql branch of GRASP
git clone -b wikidata-query-logs --single-branch git@github.com:ad-freiburg/grasp.git

# Install GRASP
cd grasp
pip install -e .

# Start vLLM server with Qwen-3-Next-80B-A3B
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tool-call-parser hermes \
  --enable-auto-tool-choice \
  --port 8337

# Start GRASP server with wdql config (runs on port 12345)
grasp serve configs/wikidata-query-logs/qwen3-next-80b-a3b.yaml

# Run generation script (more options available in the script)
python scripts/run_wikidata_query_logs.py \
  data/organic.jsonl \
  data/organic-qwen3-next-80b-a3b/ \
  http://localhost:12345/run
```

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
# Deduplicated WDQL dataset
python export_kgqa_dataset.py
# Full WDQL dataset
python export_kgqa_dataset.py --output-dir data/wdql-full \
  --samples-per-cluster -1
```

## Statistics

```bash
for bench in data/(wdql-full|wdql|spinach|simplequestions|qald7|wwq|qawiki|lcquad2|qald10); \
  do cat $bench/*.jsonl | jq '.sparql' | python sparql_statistics.py \
  > $bench/statistics.txt; \
done
```

## Visualization

```bash
streamlit run visualize_app.py
```
