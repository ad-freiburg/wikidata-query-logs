#!/usr/bin/env python3
"""
Generate embeddings for Wikidata query-SPARQL samples using Qwen3-Embedding-0.6B.
Averages embeddings across all question variations per sample.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils import validate_sample


def load_json_samples(data_dir: Path) -> list[dict]:
    """Load all JSON samples that have questions and check validity."""
    samples = []
    json_files = sorted(data_dir.glob("*.json"))

    print(f"Found {len(json_files)} JSON files")

    validity_stats = {"total": 0, "valid": 0, "invalid_reasons": {}}

    for json_file in tqdm(json_files, desc="Loading samples"):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            validity_stats["total"] += 1

            # Check validity
            is_valid, reason = validate_sample(data)

            # Only include samples that have questions
            if (
                data
                and "output" in data
                and data["output"]
                and "questions" in data["output"]
                and data["output"]["questions"]
            ):
                output_data = data["output"]
                samples.append(
                    {
                        "file": json_file.name,
                        "questions": output_data["questions"],
                        "sparql": output_data.get("sparql", ""),
                        "formatted": output_data.get("formatted", ""),
                        "type": output_data.get("type", ""),
                        "error": data.get("error"),
                        "valid": is_valid,
                        "validity_reason": reason,
                    }
                )

                if is_valid:
                    validity_stats["valid"] += 1
                else:
                    validity_stats["invalid_reasons"][reason] = (
                        validity_stats["invalid_reasons"].get(reason, 0) + 1
                    )

        except Exception:
            continue

    # Print validity statistics
    print("\nValidity Statistics:")
    print(f"  Total files processed: {validity_stats['total']}")
    print(f"  Samples with questions: {len(samples)}")
    print(
        f"  Valid samples: {validity_stats['valid']} ({validity_stats['valid'] / len(samples) * 100:.1f}%)"
    )
    print(f"  Invalid samples: {len(samples) - validity_stats['valid']}")
    if validity_stats["invalid_reasons"]:
        print("\n  Invalid reasons:")
        for reason, count in sorted(
            validity_stats["invalid_reasons"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"    {reason}: {count}")

    return samples


def generate_embeddings(
    samples: list[dict],
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    batch_size: int = 128,
) -> tuple[np.ndarray, list[dict]]:
    """
    Generate embeddings for all samples by averaging question variations.
    Uses batching across samples for efficiency.

    Args:
        samples: List of sample dictionaries with 'questions' field
        model_name: HuggingFace model name
        batch_size: Number of samples to process in each batch

    Returns:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        metadata: list of metadata dicts for each sample
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading model: {model_name}")
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)

    embeddings_list = []
    metadata_list = []

    print("\nGenerating embeddings with batching...")
    for i in tqdm(range(0, len(samples), batch_size), desc="Processing batches"):
        batch = samples[i : i + batch_size]

        # Collect all questions from this batch
        all_questions = []
        sample_indices = []  # Track which questions belong to which sample

        for sample_idx, sample in enumerate(batch):
            questions = sample["questions"]
            all_questions.extend(questions)
            sample_indices.extend([sample_idx] * len(questions))

        # Encode all questions in the batch at once
        # normalize_embeddings=True: L2-normalize for cosine distance via Euclidean
        question_embeddings = model.encode(
            all_questions,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Average embeddings for each sample
        for sample_idx, sample in enumerate(batch):
            # Get embeddings for this sample's questions
            mask = [idx == sample_idx for idx in sample_indices]
            sample_embeddings = question_embeddings[mask]

            # Average across all variations
            avg_embedding = np.mean(sample_embeddings, axis=0)

            embeddings_list.append(avg_embedding)
            metadata_list.append(
                {
                    "file": sample["file"],
                    "questions": sample["questions"],
                    "sparql": sample["sparql"],
                    "formatted": sample["formatted"],
                    "type": sample["type"],
                    "error": sample["error"],
                    "num_questions": len(sample["questions"]),
                    "valid": sample["valid"],
                    "validity_reason": sample["validity_reason"],
                }
            )

    embeddings = np.vstack(embeddings_list)
    print(
        f"\nGenerated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}"
    )

    return embeddings, metadata_list


def save_results(
    embeddings: np.ndarray, metadata: list[dict], output_dir: Path, model_name: str
) -> None:
    """Save embeddings as numpy array and metadata to disk."""
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save embeddings as numpy array
    print("\nSaving embeddings...")
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")

    # Save metadata as JSON
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Save a summary JSON
    summary = {
        "num_samples": len(embeddings),
        "embedding_dim": embeddings.shape[1],
        "model": model_name,
        "normalized": True,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate embeddings for Wikidata query-SPARQL samples"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="HuggingFace model name (default: Qwen/Qwen3-Embedding-0.6B)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for processing samples (default: 128)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/organic-qwen3-next-80b-a3b",
        help="Input directory containing JSON files (default: "
        "data/organic-qwen3-next-80b-a3b)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for embeddings (default: <input-dir>/embeddings)",
    )

    args = parser.parse_args()

    data_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "embeddings"

    # Load samples
    print("Step 1: Loading JSON samples...")
    samples = load_json_samples(data_dir)
    print(f"Loaded {len(samples)} valid samples")

    if len(samples) == 0:
        print("No valid samples found. Exiting.")
        return

    # Generate embeddings
    print("\nStep 2: Generating embeddings...")
    embeddings, metadata = generate_embeddings(
        samples, model_name=args.model, batch_size=args.batch_size
    )

    # Save results
    print("\nStep 3: Saving results...")
    save_results(embeddings, metadata, output_dir, args.model)

    print("\n✓ Embedding generation complete!")


if __name__ == "__main__":
    main()
