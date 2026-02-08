# Wikidata Query Logs Dataset (WDQL)

A dataset of question-SPARQL-pairs built from
[Wikidata Query Logs](https://iccl.inf.tu-dresden.de/web/Wikidata_SPARQL_Logs/en)
from 2017 to 2018.

# Overview

The two most important files are:
- [wdql-all.tar.gz](https://ad-publications.cs.uni-freiburg.de/wikidata-query-logs/wdql-all.tar.gz): WDQL dataset for KGQA (train/val/test split by cluster, all samples per cluster)
- [wdql-uniq.tar.gz](https://ad-publications.cs.uni-freiburg.de/wikidata-query-logs/wdql-uniq.tar.gz): WDQL dataset for KGQA (train/val/test split by cluster, one sample per cluster)

Both archives contains three JSONL files: `train.jsonl`, `val.jsonl`, and `test.jsonl`.
Each line in these files is a JSON object with the following structure:

```jsonc
{
  "id": "train_132930",
  "question": "Works by Victor Hugo with French title \"Les Misérables\"",
  "sparql": "SELECT ?work WHERE { ?work wdt:P50 ?author . ?author rdfs:label \"Victor Hugo\"@fr . ?work wdt:P1476 \"Les Misérables\"@fr . }",
  "paraphrases": [
    "What works authored by Victor Hugo have the French title \"Les Misérables\"?",
    "List all works written by Victor Hugo that are titled \"Les Misérables\" in French."
  ],
  "info": {
    // Original SPARQL query from the query logs
    "raw_sparql": "SELECT ?var1 WHERE { ?var1 <http://www.wikidata.org/prop/direct/P50> ?var2 . ?var2 <http://www.w3.org/2000/01/rdf-schema#label> \"string1\"@fr . ?var1 <http://www.wikidata.org/prop/direct/P1476> ?var3 . ?var3 <http://www.w3.org/2000/01/rdf-schema#label> \"string2\"@fr . }"
  }
}
```

> Note: If you want to use WDQL for something else than KGQA, you can just
> concatenate all JSONL files after downloading and extracting `wdql-all.tar.gz` or
> `wdql-uniq.tar.gz` to get a single file with all question-SPARQL pairs.

## All Downloads

All assets are available for download at https://ad-publications.cs.uni-freiburg.de/wikidata-query-logs:

- `organic-query-logs.tar.gz`: Raw Wikidata SPARQL query logs as TSV files
- `organic.tar.gz`: Processed and deduplicated query logs in a single JSONL file
- `organic-qwen3-next-80b-a3b.tar.gz`: Generated question-SPARQL samples with GRASP
- `organic-qwen3-next-80b-a3b-dataset.tar.gz`: Processed GRASP samples with question embeddings and clusters
- `wdql-uniq.tar.gz`: WDQL dataset for KGQA (train/val/test split by cluster, one sample per cluster)
- `wdql-all.tar.gz`: WDQL dataset for KGQA (train/val/test split by cluster, all samples per cluster)
- `wikidata-benchmarks.tar.gz`: Other Wikidata benchmarks (for comparison)

Download and extract these files into a subdirectory named `data/` to skip
the corresponding steps in the pipeline below.

## Pipeline

### Setup

```bash
# Create data directory and install dependencies
mkdir -p data
pip install -r requirements.txt
```


### 1. Prepare input from query logs

```bash
# Download and extract raw query logs
curl -L https://ad-publications.cs.uni-freiburg.de/wikidata-query-logs/organic-query-logs.tar.gz \
  | tar -xzv -C data/
# Build organic.jsonl from TSV files
python prepare_input.py data/*.tsv data/
```

### 2. Generate question-SPARQL pairs with GRASP

```bash
# Checkout wdql branch of GRASP
git clone -b wikidata-query-logs --single-branch git@github.com:ad-freiburg/grasp.git

# Install and setup GRASP (see GRASP README for more details)
cd grasp
pip install -e .
export GRASP_INDEX_DIR=$(pwd)/grasp-indices
mkdir -p $GRASP_INDEX_DIR

# Download and extract GRASP Wikidata index
curl -L https://ad-publications.cs.uni-freiburg.de/grasp/kg-index/wikidata.tar.gz \
  | tar -xzv -C $GRASP_INDEX_DIR

# Install vLLM and start server with Qwen-3-Next-80B-A3B
pip install vllm
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tool-call-parser hermes \
  --enable-auto-tool-choice \
  --port 8336

# Start GRASP server with wdql config (runs on port 12345)
# By default, the config expects a Wikidata SPARQL endpoint at localhost:7001, but
# here we set it to the public QLever endpoint instead
KG_ENDPOINT=https://qlever.dev/api/wikidata \
  grasp serve configs/wikidata-query-logs/qwen3-next-80b-a3b.yaml

# Run generation script (more options available in the script)
python scripts/run_wikidata_query_logs.py \
  data/organic.jsonl \
  data/organic-qwen3-next-80b-a3b/ \
  http://localhost:12345/run # GRASP server URL
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
# WDQL uniq dataset (one sample per cluster)
python export_kgqa_dataset.py
# WDQL all dataset (all samples per cluster)
python export_kgqa_dataset.py --output-dir data/wdql-all \
  --samples-per-cluster -1
```

## Statistics

Generate some statistics about WDQL and other Wikidata datasets:

```bash
# Download and extract other Wikidata benchmarks
curl -L https://ad-publications.cs.uni-freiburg.de/wikidata-query-logs/wikidata-benchmarks.tar.gz \
  | tar -xzv -C data/

# Generate statistics
for bench in data/(wdql-all|wdql-uniq|spinach|simplequestions|qald7|wwq|qawiki|lcquad2|qald10); \
  do cat $bench/*.jsonl | jq '.sparql' | python sparql_statistics.py \
  > $bench/statistics.txt; \
done
```

> Note: To generate statistics for `wdql-all` and `wdql-uniq`, you need to
> complete step 5 above or download and extract the corresponding files first.

## Visualization

Run a Streamlit app to visualize the dataset:

```bash
streamlit run visualize_app.py
```

> Note: To run the app, you need to complete step 4 above or download
> and extract `organic-qwen3-next-80b-a3b-dataset.tar.gz` first.

## Declaration on AI

The code in this repository was written with the help of AI coding assistants, in
particular Claude Code. All AI-generated code was reviewed and edited by human
developers to ensure correctness and quality.
