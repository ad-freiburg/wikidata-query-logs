import argparse
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm
from universal_ml_utils.io import load_json
from universal_ml_utils.table import generate_table

from utils import validate_sample

GENERATION_REASON_LABELS = {
    "error_field_not_null (reason=api)": "Model API failure",
    "output_field_null": "Model output failure",
    "type_is_cancel": "Cancelled via `CAN`",
    "error_field_not_null (reason=loop)": "Model stuck in loop",
}

VALIDATION_REASON_LABELS = {
    "sparql_execution_failed (preprocessing)": "SPARQL parsing failed",
    "sparql_execution_failed (execution)": "SPARQL execution failed",
    "empty_result": "Empty SPARQL result",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the dataset creation statistics table from release files"
    )
    parser.add_argument(
        "--raw-logs",
        type=Path,
        nargs="+",
        required=True,
        help="Raw organic TSV query-log files",
    )
    parser.add_argument(
        "--organic-jsonl",
        type=Path,
        required=True,
        help="Path to the deduplicated organic.jsonl file",
    )
    parser.add_argument(
        "--generation-dir",
        type=Path,
        required=True,
        help="Directory containing the generation JSON files",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing samples.json and clusters/cluster_stats.json",
    )
    parser.add_argument(
        "--wdql-one-per-cluster-dir",
        type=Path,
        required=True,
        help="Directory containing train.jsonl, val.jsonl, and test.jsonl for WDQL one-per-cluster",
    )
    parser.add_argument(
        "--wdql-dir",
        type=Path,
        required=True,
        help="Directory containing train.jsonl, val.jsonl, and test.jsonl for WDQL",
    )
    return parser.parse_args()


def count_lines_in_file(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(
            chunk.count(b"\n") for chunk in iter(lambda: stream.read(1024 * 1024), b"")
        )


def count_raw_logs(tsv_files: list[Path]) -> int:
    total = 0
    for tsv_file in tqdm(tsv_files, desc="Raw logs", unit="file"):
        total += count_lines_in_file(tsv_file)
    return total


def compute_generation_stats(generation_dir: Path) -> dict[str, Any]:
    json_files = sorted(generation_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No generation JSON files found in {generation_dir}")

    processed = len(json_files)
    with_questions = 0
    without_questions = 0
    generation_failures = {label: 0 for label in GENERATION_REASON_LABELS.values()}
    validation_invalid = {label: 0 for label in VALIDATION_REASON_LABELS.values()}

    for json_file in tqdm(json_files, desc="Generation", unit="file"):
        data = load_json(json_file)

        output = data.get("output") if data else None
        has_questions = bool(output and output.get("questions"))
        is_valid, reason = validate_sample(data)

        if has_questions:
            with_questions += 1
            if not is_valid:
                label = VALIDATION_REASON_LABELS.get(reason)
                if label is None:
                    raise ValueError(
                        "Unhandled validation reason for samples with questions: "
                        f"{reason}"
                    )
                validation_invalid[label] += 1
            continue

        without_questions += 1
        label = GENERATION_REASON_LABELS.get(reason)
        if label is None:
            raise ValueError(
                "Unhandled generation failure reason for samples without questions: "
                f"{reason}"
            )
        generation_failures[label] += 1

    if with_questions + without_questions != processed:
        raise ValueError(
            "Generation counts do not add up: "
            f"{with_questions} + {without_questions} != {processed}"
        )

    return {
        "processed": processed,
        "with_questions": with_questions,
        "without_questions": without_questions,
        "generation_failures": generation_failures,
        "validation_invalid": validation_invalid,
    }


def count_split_jsonl(dataset_dir: Path) -> tuple[int, int, int]:
    train = count_lines_in_file(dataset_dir / "train.jsonl")
    val = count_lines_in_file(dataset_dir / "val.jsonl")
    test = count_lines_in_file(dataset_dir / "test.jsonl")
    return train, val, test


def format_int(value: int) -> str:
    return f"{value:,}"


def format_pct(part: int, whole: int) -> str:
    return f"{part / whole * 100:.1f}%"


def indent(label: str, level: int = 1) -> str:
    return f"{'  ' * level}{label}"


def render_table(stats: dict[str, Any]) -> str:
    raw_logs = stats["raw_logs"]
    deduplicated = stats["deduplicated"]
    generation = stats["generation"]
    cluster_stats = stats["cluster_stats"]
    one_train, one_val, one_test = stats["wdql_one_per_cluster_split"]
    wdql_train, wdql_val, wdql_test = stats["wdql_split"]

    processed = generation["processed"]
    with_questions = generation["with_questions"]
    without_questions = generation["without_questions"]
    valid = cluster_stats["num_valid_samples"]
    invalid = with_questions - valid
    generation_failures = generation["generation_failures"]
    validation_invalid = generation["validation_invalid"]
    max_cluster_size = max(cluster_stats["cluster_sizes"].values())
    avg_cluster_size = cluster_stats["num_valid_samples"] / cluster_stats["n_clusters"]

    rows = [
        ["**Data Collection**", ""],
        [indent("Raw organic SPARQL logs"), format_int(raw_logs)],
        [indent("After deduplication"), format_int(deduplicated)],
        ["**SPARQL Fixing and Question Generation with GRASP**", ""],
        [indent("Processed samples"), format_int(processed)],
        [
            indent(
                f"With questions ({format_pct(with_questions, processed)})", level=2
            ),
            format_int(with_questions),
        ],
        [
            indent(
                f"Without questions ({format_pct(without_questions, processed)})",
                level=2,
            ),
            format_int(without_questions),
        ],
        [
            indent("Model API failure", level=3),
            format_int(generation_failures["Model API failure"]),
        ],
        [
            indent("Model output failure", level=3),
            format_int(generation_failures["Model output failure"]),
        ],
        [
            indent("Cancelled via `CAN`", level=3),
            format_int(generation_failures["Cancelled via `CAN`"]),
        ],
        [
            indent("Model stuck in loop", level=3),
            format_int(generation_failures["Model stuck in loop"]),
        ],
        ["**Validation**", ""],
        [indent(f"Valid ({format_pct(valid, with_questions)})"), format_int(valid)],
        [
            indent(f"Invalid ({format_pct(invalid, with_questions)})"),
            format_int(invalid),
        ],
        [
            indent("SPARQL parsing failed", level=2),
            format_int(validation_invalid["SPARQL parsing failed"]),
        ],
        [
            indent("SPARQL execution failed", level=2),
            format_int(validation_invalid["SPARQL execution failed"]),
        ],
        [
            indent("Empty SPARQL result", level=2),
            format_int(validation_invalid["Empty SPARQL result"]),
        ],
        ["**Clustering**", ""],
        [indent("Clustered samples (valid)"), format_int(valid)],
        [indent("Num. clusters", level=2), format_int(cluster_stats["n_clusters"])],
        [indent("Max. cluster size", level=2), format_int(max_cluster_size)],
        [indent("Avg. cluster size", level=2), f"{avg_cluster_size:.2f}"],
        ["**KGQA Datasets**", ""],
        [
            indent("WDQL (one-per-cluster)"),
            format_int(sum((one_train, one_val, one_test))),
        ],
        [
            indent("Train / Val / Test", level=2),
            f"{format_int(one_train)} / {format_int(one_val)} / {format_int(one_test)}",
        ],
        [indent("WDQL"), format_int(sum((wdql_train, wdql_val, wdql_test)))],
        [
            indent("Train / Val / Test", level=2),
            f"{format_int(wdql_train)} / {format_int(wdql_val)} / {format_int(wdql_test)}",
        ],
    ]
    return generate_table(
        rows,
        headers=[["Stage", "Number"]],
        max_column_width=80,
    )


def main() -> None:
    args = parse_args()

    print("Counting raw logs...", file=sys.stderr)
    raw_logs = count_raw_logs(args.raw_logs)
    print("Counting deduplicated samples...", file=sys.stderr)
    deduplicated = count_lines_in_file(args.organic_jsonl)
    print("Scanning generation files...", file=sys.stderr)
    generation = compute_generation_stats(args.generation_dir)
    print("Loading cluster stats...", file=sys.stderr)
    cluster_stats = load_json(args.dataset_dir / "clusters" / "cluster_stats.json")
    print("Counting one-per-cluster WDQL splits...", file=sys.stderr)
    one_per_cluster_split = count_split_jsonl(args.wdql_one_per_cluster_dir)
    print("Counting WDQL splits...", file=sys.stderr)
    wdql_split = count_split_jsonl(args.wdql_dir)

    if cluster_stats["num_valid_samples"] != sum(wdql_split):
        raise ValueError(
            "WDQL split count does not match cluster_stats.num_valid_samples: "
            f"{sum(wdql_split)} != {cluster_stats['num_valid_samples']}"
        )
    if cluster_stats["n_clusters"] != sum(one_per_cluster_split):
        raise ValueError(
            "One-per-cluster split count does not match cluster_stats.n_clusters: "
            f"{sum(one_per_cluster_split)} != {cluster_stats['n_clusters']}"
        )
    if generation["with_questions"] != (
        cluster_stats["num_valid_samples"]
        + sum(generation["validation_invalid"].values())
    ):
        raise ValueError(
            "Validation counts do not add up to the number of samples with questions"
        )

    print("Rendering table...", file=sys.stderr)
    print(
        render_table(
            {
                "raw_logs": raw_logs,
                "deduplicated": deduplicated,
                "generation": generation,
                "cluster_stats": cluster_stats,
                "wdql_one_per_cluster_split": one_per_cluster_split,
                "wdql_split": wdql_split,
            }
        )
    )


if __name__ == "__main__":
    main()
