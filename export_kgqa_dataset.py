import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from universal_ml_utils.io import dump_jsonl, load_json


def sample_one_per_cluster(
    metadata: list[dict[str, Any]], cluster_labels: list[int], seed: int = 42
) -> list[dict[str, Any]]:
    """
    Sample one valid sample per cluster.

    Args:
        metadata: List of sample metadata
        cluster_labels: List of cluster IDs for each sample
        seed: Random seed for reproducibility

    Returns:
        List of sampled samples with cluster information
    """
    random.seed(seed)

    # Group valid samples by cluster
    cluster_to_samples: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)

    for idx, (sample, cluster_id) in enumerate(zip(metadata, cluster_labels)):
        # Skip invalid samples (cluster_id == -1 or not valid)
        if cluster_id == -1 or not sample.get("valid", False):
            continue

        cluster_to_samples[cluster_id].append((idx, sample))

    # Sample one per cluster
    sampled = []
    for cluster_id, samples in sorted(cluster_to_samples.items()):
        # Randomly sample one from the cluster
        idx, sample = random.choice(samples)
        # Prefer sparql_fixed if available and non-empty, otherwise use sparql
        sparql = sample.get("sparql_fixed", "").strip() or sample["sparql"]
        sampled.append(
            {
                "cluster_id": cluster_id,
                "sample_idx": idx,
                "file": sample["file"],
                "questions": sample["questions"],
                "sparql": sparql,
            }
        )

    return sampled


def split_by_cluster(
    samples: list[dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split samples into train, validation, and test sets by cluster.

    Args:
        samples: List of samples with cluster_id
        train_ratio: Ratio of clusters for training
        val_ratio: Ratio of clusters for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    random.seed(seed)

    # Get unique cluster IDs
    cluster_ids = sorted(set(s["cluster_id"] for s in samples))

    # Shuffle and split clusters
    random.shuffle(cluster_ids)
    train_split_idx = int(len(cluster_ids) * train_ratio)
    val_split_idx = int(len(cluster_ids) * (train_ratio + val_ratio))

    train_cluster_ids = set(cluster_ids[:train_split_idx])
    val_cluster_ids = set(cluster_ids[train_split_idx:val_split_idx])
    test_cluster_ids = set(cluster_ids[val_split_idx:])

    # Split samples
    train_samples = [s for s in samples if s["cluster_id"] in train_cluster_ids]
    val_samples = [s for s in samples if s["cluster_id"] in val_cluster_ids]
    test_samples = [s for s in samples if s["cluster_id"] in test_cluster_ids]

    return train_samples, val_samples, test_samples


def format_as_jsonl(samples: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """
    Format samples in JSONL format.

    Args:
        samples: List of samples to format
        split: "train" or "test"

    Returns:
        List of formatted samples
    """
    formatted = []

    for idx, sample in enumerate(samples):
        questions = sample["questions"]

        # Use first question as main question, rest as paraphrases
        main_question = questions[0] if questions else ""
        paraphrases = questions[1:] if len(questions) > 1 else []

        formatted.append(
            {
                "id": f"{split}_{idx}",
                "question": main_question,
                "sparql": sample["sparql"],
                "paraphrases": paraphrases,
                "info": {},
            }
        )

    return formatted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SPARQL QA dataset from clustered samples"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("data/organic-qwen3-next-80b-a3b/embeddings"),
        help="Path to embeddings directory with samples.json and cluster_labels.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/dataset"),
        help="Output directory for train.jsonl, val.jsonl, and test.jsonl",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Ratio of clusters for training (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Ratio of clusters for validation (default: 0.15)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.dataset_dir}")
    metadata = load_json(args.dataset_dir / "samples.json")
    cluster_labels = load_json(args.dataset_dir / "cluster_labels.json")

    print(f"Loaded {len(metadata)} samples with {len(cluster_labels)} cluster labels")

    # Validate lengths match
    if len(metadata) != len(cluster_labels):
        raise ValueError(
            f"Mismatch: {len(metadata)} samples but {len(cluster_labels)} cluster labels"
        )

    # Sample one per cluster
    print("\nSampling one sample per cluster...")
    sampled = sample_one_per_cluster(metadata, cluster_labels, seed=args.seed)
    print(
        f"Sampled {len(sampled)} samples from {len(set(s['cluster_id'] for s in sampled))} clusters"
    )

    # Count valid samples
    num_valid = sum(
        1 for s, c in zip(metadata, cluster_labels) if s.get("valid", False) and c != -1
    )
    num_invalid = sum(1 for s, c in zip(metadata, cluster_labels) if c == -1)
    print(f"  - Valid samples: {num_valid}")
    print(f"  - Invalid samples: {num_invalid}")

    # Split by cluster
    print(
        f"\nSplitting into train/val/test with ratios {args.train_ratio}/{args.val_ratio}/{1 - args.train_ratio - args.val_ratio:.2f}..."
    )
    train_samples, val_samples, test_samples = split_by_cluster(
        sampled,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"  - Train: {len(train_samples)} samples")
    print(f"  - Val: {len(val_samples)} samples")
    print(f"  - Test: {len(test_samples)} samples")

    # Format as JSONL
    print("\nFormatting as JSONL...")
    train_formatted = format_as_jsonl(train_samples, "train")
    val_formatted = format_as_jsonl(val_samples, "val")
    test_formatted = format_as_jsonl(test_samples, "test")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Write output files
    train_path = args.output_dir / "train.jsonl"
    val_path = args.output_dir / "val.jsonl"
    test_path = args.output_dir / "test.jsonl"

    print("\nWriting output files...")
    dump_jsonl(train_path, train_formatted)
    dump_jsonl(val_path, val_formatted)
    dump_jsonl(test_path, test_formatted)

    print(f"  - {train_path}: {len(train_formatted)} samples")
    print(f"  - {val_path}: {len(val_formatted)} samples")
    print(f"  - {test_path}: {len(test_formatted)} samples")
    print("\nDone!")


if __name__ == "__main__":
    main()
