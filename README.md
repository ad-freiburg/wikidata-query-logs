# Wikidata Query Logs Dataset (WDQL)

A dataset of question-SPARQL-pairs built from
[Wikidata Query Logs](https://iccl.inf.tu-dresden.de/web/Wikidata_SPARQL_Logs/en)
from 2017 to 2018.

See this [web app](https://wdql.cs.uni-freiburg.de/) to explore the dataset.

See the [paper](https://arxiv.org/pdf/2602.14594) for more details.

## News

- **2026-04-21**: Second WDQL release (`21-04-26`) with additional Qwen3.5 27B
  generations on top of the original Qwen3-Next 80B A3B generations. The new
  release grows WDQL from 200,186 to 335,450 samples (train/val/test
  268,913 / 33,149 / 33,388) and WDQL one-per-cluster from 103,327 to 173,766.
- **2026-04-02**: WDQL has been accepted to [SIGIR 2026](https://sigir2026.org/)
- **2026-02-02**: Initial WDQL release (`02-02-26`) with 200,186 samples
  generated from organic Wikidata query logs using Qwen3-Next 80B A3B.

# Overview

The two most important files are:

- [wdql.tar.gz](https://wdql.cs.uni-freiburg.de/data/latest/wdql.tar.gz): WDQL dataset for KGQA (train/val/test split by cluster, all samples per cluster)
- [wdql-one-per-cluster.tar.gz](https://wdql.cs.uni-freiburg.de/data/latest/wdql-one-per-cluster.tar.gz): WDQL dataset for KGQA (train/val/test split by cluster, one sample per cluster)

Both archives contain three JSONL files: `train.jsonl`, `val.jsonl`, and `test.jsonl`.
Each line in these files is a JSON object with the following structure:

```jsonc
{
  "id": "train_132930",
  "question": "Works by Victor Hugo with French title \"Les Misérables\"",
  "sparql": "PREFIX wdt: <http://www.wikidata.org/prop/direct/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> SELECT ?work WHERE { ?work wdt:P50 ?author . ?author rdfs:label \"Victor Hugo\"@fr . ?work wdt:P1476 \"Les Misérables\"@fr . }",
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
> concatenate all JSONL files after downloading and extracting `wdql.tar.gz` or
> `wdql-one-per-cluster.tar.gz` to get a single file with all question-SPARQL pairs.

> Note: Initially SPARQL queries were generated without PREFIX
> declarations. Affected queries were fixed post-hoc with this
> [fix_prefixes.py](https://github.com/ad-freiburg/grasp/blob/wikidata-query-logs/scripts/fix_prefixes.py)
> script.

## All Downloads

All assets are available for download at <https://wdql.cs.uni-freiburg.de/data/>.

Top-level (shared across releases):

- `organic-query-logs.tar.gz`: Raw Wikidata SPARQL query logs as TSV files
- `organic.tar.gz`: Processed and deduplicated query logs in a single JSONL file
- `wikidata-benchmarks.tar.gz`: Other Wikidata benchmarks (for comparison)

Per-release files live under a dated subdirectory (e.g. `21-04-26/`). The
`latest/` symlink always points to the most recent release.

- `latest/wdql.tar.gz`: WDQL dataset for KGQA (train/val/test split by cluster, all samples per cluster)
- `latest/wdql-one-per-cluster.tar.gz`: WDQL dataset for KGQA (train/val/test split by cluster, one sample per cluster)
- `latest/organic-qwen3-next-80b-a3b-and-qwen35-27b.tar.gz`: Generated question-SPARQL samples with GRASP (Qwen3-Next 80B A3B + Qwen3.5 27B; the `02-02-26` release uses `organic-qwen3-next-80b-a3b.tar.gz` since it only includes the Qwen3-Next 80B A3B generations)
- `latest/organic-qwen3-next-80b-a3b-and-qwen35-27b-dataset.tar.gz`: Processed GRASP samples with question embeddings and clusters (same naming caveat as above)

Download and extract these files into a subdirectory named `data/` to skip
the corresponding steps in the pipeline below.

## Dataset Creation Statistics (`02-02-26` release)

| Stage | Number |
|-------|--------|
| **Data Collection** | |
| &nbsp;&nbsp;&nbsp;&nbsp;Raw organic SPARQL logs | 3,530,955 |
| &nbsp;&nbsp;&nbsp;&nbsp;After deduplication | 859,305 |
| **SPARQL Fixing and Question Generation with GRASP** | |
| &nbsp;&nbsp;&nbsp;&nbsp;Processed samples | 314,430 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;With questions (68.5%) | 215,256 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Without questions (31.5%) | 99,174 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model API failure | 78,104 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model output failure | 18,280 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Cancelled via `CAN` | 2,770 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model stuck in loop | 20 |
| **Validation** | |
| &nbsp;&nbsp;&nbsp;&nbsp;Valid (93.0%) | 200,186 |
| &nbsp;&nbsp;&nbsp;&nbsp;Invalid (7.0%) | 15,070 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SPARQL parsing failed | 392 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SPARQL execution failed | 3,111 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Empty SPARQL result | 11,567 |
| **Clustering** | |
| &nbsp;&nbsp;&nbsp;&nbsp;Clustered samples (valid) | 200,186 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Num. clusters | 103,327 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Max. cluster size | 146 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Avg. cluster size | 1.94 |
| **KGQA Datasets** | |
| &nbsp;&nbsp;&nbsp;&nbsp;WDQL (one-per-cluster) | 103,327 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Train / Val / Test | 82,661 / 10,333 / 10,333 |
| &nbsp;&nbsp;&nbsp;&nbsp;WDQL | 200,186 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Train / Val / Test | 159,815 / 20,485 / 19,886 |

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
curl -L https://wdql.cs.uni-freiburg.de/data/organic-query-logs.tar.gz \
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
python export_kgqa_dataset.py --output-dir data/wdql \
  --samples-per-cluster -1
```

## Statistics

Generate some statistics about WDQL and other Wikidata datasets:

```bash
# Download and extract other Wikidata benchmarks
curl -L https://wdql.cs.uni-freiburg.de/data/wikidata-benchmarks.tar.gz \
  | tar -xzv -C data/

# Generate statistics
for bench in data/(wdql|wdql-one-per-cluster|spinach|simplequestions|qald7|wwq|qawiki|lcquad2|qald10); \
  do cat $bench/*.jsonl | jq '.sparql' | python sparql_statistics.py \
  > $bench/statistics.txt; \
done
```

> Note: To generate statistics for `wdql` and `wdql-one-per-cluster`, you need to
> complete step 5 above or download and extract the corresponding files first.

## Visualization

Run a Streamlit app to visualize the dataset:

```bash
streamlit run visualize_app.py
```

> Note: To run the app, you need to complete step 4 above or download
> and extract `latest/organic-qwen3-next-80b-a3b-and-qwen35-27b-dataset.tar.gz`
> first.
