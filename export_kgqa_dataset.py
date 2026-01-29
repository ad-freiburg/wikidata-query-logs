import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from universal_ml_utils.io import dump_jsonl, load_json


def sample_from_clusters(
    metadata: list[dict[str, Any]],
    cluster_labels: list[int],
    samples_per_cluster: int = 1,
    seed: int = 22,
) -> list[dict[str, Any]]:
    """
    Sample valid samples from each cluster.

    Args:
        metadata: List of sample metadata
        cluster_labels: List of cluster IDs for each sample
        samples_per_cluster: Number of samples to take per cluster
        seed: Random seed for reproducibility

    Returns:
        List of sampled samples with cluster information
    """
    random.seed(seed)

    # Group valid samples by cluster
    cluster_to_samples: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)

    for idx, (sample, cluster_id) in enumerate(zip(metadata, cluster_labels)):
        # Skip invalid samples
        if not sample["valid"]:
            assert cluster_id == -1, "Invalid points should not be assigned to clusters"
            continue

        cluster_to_samples[cluster_id].append((idx, sample))

    print(f"Found {len(cluster_to_samples)} clusters with valid samples")

    # Sample from each cluster
    sampled = []
    for cluster_id, samples in sorted(cluster_to_samples.items()):
        # Select samples: -1 means keep all, otherwise sample up to k
        if samples_per_cluster == -1:
            selected = samples
        else:
            k = min(samples_per_cluster, len(samples))
            selected = random.sample(samples, k)
        for idx, sample in selected:
            # Prefer sparql_fixed if available and non-empty, otherwise use sparql
            sparql = sample.get("sparql_fixed", "").strip() or sample["sparql"]
            sampled.append(
                {
                    "cluster_id": cluster_id,
                    "sample_idx": idx,
                    "questions": sample["questions"],
                    "sparql": sparql,
                }
            )

    return sampled


def split_by_cluster(
    samples: list[dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 22,
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
        "--dataset-dir",
        type=Path,
        default=Path("data/organic-qwen3-next-80b-a3b-dataset"),
        help="Path to embeddings directory with samples.json and cluster_labels.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/wdql-kgqa-dataset"),
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
        "--seed",
        type=int,
        default=22,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--samples-per-cluster",
        type=int,
        default=1,
        help="Number of samples to keep per cluster, -1 for all (default: 1)",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.dataset_dir}")
    metadata = load_json(args.dataset_dir / "samples.json")
    cluster_labels = load_json(args.dataset_dir / "clusters" / "cluster_labels.json")

    print(f"Loaded {len(metadata)} samples with {len(cluster_labels)} cluster labels")

    # Validate lengths match
    if len(metadata) != len(cluster_labels):
        raise ValueError(
            f"Mismatch: {len(metadata)} samples but {len(cluster_labels)} cluster labels"
        )

    # Sample from clusters
    print(f"\nSampling {args.samples_per_cluster} sample(s) per cluster...")
    sampled = sample_from_clusters(
        metadata, cluster_labels,
        samples_per_cluster=args.samples_per_cluster,
        seed=args.seed,
    )
    print(
        f"Sampled {len(sampled)} samples from {len(set(s['cluster_id'] for s in sampled))} clusters"
    )

    # Count valid samples
    num_valid = len(sampled)
    num_invalid = len(cluster_labels) - num_valid
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
    dump_jsonl(train_formatted, train_path)
    dump_jsonl(val_formatted, val_path)
    dump_jsonl(test_formatted, test_path)

    print(f"  - {train_path}: {len(train_formatted)} samples")
    print(f"  - {val_path}: {len(val_formatted)} samples")
    print(f"  - {test_path}: {len(test_formatted)} samples")
    print("\nDone!")


if __name__ == "__main__":
    main()
