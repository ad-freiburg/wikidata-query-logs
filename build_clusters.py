import argparse
from pathlib import Path

import hdbscan
import numpy as np
from umap import UMAP
from universal_ml_utils.io import dump_json, load_json


def load_embeddings_and_samples(
    dataset_dir: Path,
) -> tuple[np.ndarray, list[dict]]:
    """Load embeddings and samples"""
    # Load embeddings
    embeddings_path = dataset_dir / "embeddings.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    print(f"Loaded {len(embeddings)} vectors, dimension {embeddings.shape[1]}")

    # Load samples
    samples = load_json(dataset_dir / "samples.json")

    return embeddings, samples  # type: ignore


def cluster_hdbscan(
    vectors: np.ndarray,
    valid_mask: np.ndarray,
    min_cluster_size: int = 2,
    min_samples: int = 1,
    epsilon: float = 0.0,
    method: str = "eom",
    umap_n_components: int = 50,
    umap_n_neighbors: int = 30,
) -> np.ndarray:
    """
    Apply UMAP dimensionality reduction followed by HDBSCAN clustering to valid samples.

    Args:
        vectors: All vectors (valid and invalid)
        valid_mask: Boolean mask indicating valid samples
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples in a neighborhood for HDBSCAN
        epsilon: Cluster selection epsilon for HDBSCAN
        method: Cluster selection method for HDBSCAN
        umap_n_components: Target dimensionality for UMAP reduction (before clustering)
        umap_n_neighbors: Number of neighbors for UMAP

    Returns:
        labels: Cluster labels for all samples (-1 for invalid and noise)
    """
    print("\nApplying UMAP + HDBSCAN clustering pipeline...")
    print(f"  Using {valid_mask.sum()} valid samples out of {len(vectors)} total")
    print(
        f"  UMAP: reducing to {umap_n_components}d with n_neighbors={umap_n_neighbors}, metric=cosine"
    )
    print(
        f"  HDBSCAN: min_cluster_size={min_cluster_size}, min_samples={min_samples}, metric=euclidean"
    )

    # Cluster only valid samples
    valid_vectors = vectors[valid_mask]

    # Apply UMAP dimensionality reduction before clustering
    print("  Step 1: UMAP dimensionality reduction...")
    reducer = UMAP(
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=0.0,  # Preserve global structure for clustering
        metric="cosine",
    )
    valid_vectors_reduced = reducer.fit_transform(valid_vectors)
    print(
        f"  Reduced from {valid_vectors.shape[1]}d to {valid_vectors_reduced.shape[1]}d"
    )

    # Apply HDBSCAN clustering on reduced vectors
    print("  Step 2: HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_epsilon=epsilon,
        cluster_selection_method=method,
    )
    valid_labels = clusterer.fit_predict(valid_vectors_reduced)

    # Create full label array with -1 for invalid samples
    labels = np.full(len(vectors), -1, dtype=int)
    labels[valid_mask] = valid_labels

    # Count actual clusters (excluding noise which is -1)
    n_clusters_initial = len(set(valid_labels)) - (1 if -1 in valid_labels else 0)
    n_noise = np.sum(valid_labels == -1)

    print(f"HDBSCAN complete. Found {n_clusters_initial} dense clusters")
    print(f"Noise points (within valid samples): {n_noise}")

    # Assign each valid noise point to its own singleton cluster
    if n_noise > 0:
        next_cluster_id = max(valid_labels) + 1
        noise_indices = np.where((labels == -1) & valid_mask)[0]
        for idx in noise_indices:
            labels[idx] = next_cluster_id
            next_cluster_id += 1
        print(f"Assigned {n_noise} rare/isolated samples to singleton clusters")

    n_clusters_final = len(set(labels[valid_mask]))
    print(f"Total clusters (including singletons): {n_clusters_final}")
    print("Invalid samples assigned cluster -1")

    return labels


def reduce_dimensions_umap(
    vectors: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """Apply UMAP for 2D visualization."""
    print("\nApplying UMAP for 2D visualization...")
    print(f"  n_neighbors: {n_neighbors}, min_dist: {min_dist}")

    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
    )
    coords_2d = reducer.fit_transform(vectors)
    print(f"UMAP complete. Shape: {coords_2d.shape}")

    return coords_2d  # type: ignore


def save_results(
    output_dir: Path,
    labels: np.ndarray,
    coords_2d: np.ndarray,
    cluster_stats: dict,
) -> None:
    """Save clustering and visualization results."""
    print("\nSaving results...")

    # Save cluster labels
    labels_path = output_dir / "cluster_labels.json"
    dump_json(labels.tolist(), labels_path)
    print(f"Saved cluster labels to {labels_path}")

    # Save UMAP coordinates
    coords_path = output_dir / "umap_coords.json"
    dump_json(coords_2d.tolist(), coords_path)
    print(f"Saved UMAP coordinates to {coords_path}")

    # Save cluster statistics
    stats_path = output_dir / "cluster_stats.json"
    dump_json(cluster_stats, stats_path)
    print(f"Saved cluster statistics to {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build clusters from embeddings using UMAP + HDBSCAN and create 2D visualization"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/organic-qwen3-next-80b-a3b-dataset"),
        help="Directory containing embeddings (default: data/organic-qwen3-next-80b-a3b-dataset)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum cluster size for HDBSCAN (default: 2 - smallest possible, preserves rare samples)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum samples for HDBSCAN (default: 1 - very liberal clustering)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Cluster selection epsilon for HDBSCAN (default: 0.0 - no minimum distance required)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="eom",
        help="Cluster selection method for HDBSCAN (default: 'eom')",
    )
    parser.add_argument(
        "--umap-n-components",
        type=int,
        default=50,
        help="UMAP n_components for dimensionality reduction before clustering (default: 50)",
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=30,
        help="UMAP n_neighbors for dimensionality reduction before clustering (default: 30)",
    )
    parser.add_argument(
        "--vis-umap-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors for 2D visualization (default: 15)",
    )
    parser.add_argument(
        "--vis-umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist for 2D visualization (default: 0.1)",
    )

    args = parser.parse_args()

    # Load embeddings and samples
    print("Step 1: Loading embeddings and samples...")
    embeddings, samples = load_embeddings_and_samples(args.dataset_dir)

    # Create valid mask from samples
    valid_mask = np.array([sample["valid"] for sample in samples], dtype=bool)
    print(
        f"\nValid samples: {valid_mask.sum()} / {len(valid_mask)} ({valid_mask.sum() / len(valid_mask) * 100:.1f}%)"
    )

    # Apply UMAP + HDBSCAN clustering
    print("\nStep 2: Clustering with UMAP + HDBSCAN...")
    labels = cluster_hdbscan(
        embeddings,
        valid_mask,
        args.min_cluster_size,
        args.min_samples,
        args.epsilon,
        args.method,
        args.umap_n_components,
        args.umap_n_neighbors,
    )

    # Compute cluster statistics
    valid_labels = labels[labels != -1]
    unique_labels, counts = np.unique(valid_labels, return_counts=True)
    n_clusters = len(unique_labels)
    cluster_stats = {
        "algorithm": "hdbscan",
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "umap_n_components": args.umap_n_components,
        "umap_n_neighbors": args.umap_n_neighbors,
        "n_clusters": n_clusters,
        "num_valid_samples": int(valid_mask.sum()),
        "num_invalid_samples": int((~valid_mask).sum()),
        "cluster_sizes": {
            int(label): int(count)
            for label, count in zip(unique_labels, counts, strict=False)
        },
    }

    # Apply UMAP for 2D visualization
    print("\nStep 3: UMAP dimensionality reduction for visualization...")
    coords_2d = reduce_dimensions_umap(
        embeddings,
        n_neighbors=args.vis_umap_neighbors,
        min_dist=args.vis_umap_min_dist,
    )

    # Save results
    print("\nStep 4: Saving results...")
    save_results(args.dataset_dir / "clusters", labels, coords_2d, cluster_stats)

    print("\n✓ Clustering and visualization complete!")


if __name__ == "__main__":
    main()
