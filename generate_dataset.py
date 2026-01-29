import argparse
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from universal_ml_utils.io import dump_json, load_json, load_jsonl

from utils import validate_sample


def load_json_samples(input_file: Path, data_dir: Path) -> list[dict]:
    """Load all JSON samples that have questions and check validity."""
    inputs = load_jsonl(input_file)

    samples = []
    json_files = sorted(data_dir.glob("*.json"))

    print(f"Found {len(json_files)} JSON files")

    validity_stats = {
        "total": 0,
        "none": 0,
        "error": 0,
        "valid": 0,
        "invalid_reasons": {},
    }

    # Track validity reasons for samples without proper output (no questions)
    no_output_reasons: dict[str, int] = {}

    for json_file in tqdm(json_files, desc="Loading samples"):
        try:
            data = load_json(json_file)
            input = inputs[int(json_file.stem)]

            validity_stats["total"] += 1

            # Check validity
            is_valid, reason = validate_sample(data)

            if is_valid:
                validity_stats["valid"] += 1
            else:
                cur = validity_stats["invalid_reasons"].get(reason, 0)
                validity_stats["invalid_reasons"][reason] = cur + 1

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
                        "origin": {
                            "file": json_file.name,
                            "input": input,
                        },
                        "questions": output_data["questions"],
                        "sparql": output_data.get("sparql", ""),
                        "sparql_fixed": output_data.get("sparql_fixed", ""),
                        "formatted": output_data.get("formatted", ""),
                        "type": output_data.get("type", ""),
                        "error": data.get("error"),
                        "valid": is_valid,
                        "validity_reason": reason,
                    }
                )
            else:
                # Track reasons for samples without proper output
                no_output_reasons[reason] = no_output_reasons.get(reason, 0) + 1

        except Exception:
            continue

    # Print validity statistics
    print("\nValidity Statistics:")
    print(f"  Total files processed: {validity_stats['total']}")
    print(f"  Samples with questions: {len(samples)}")
    no_output_total = sum(no_output_reasons.values())
    print(f"  Samples without questions: {no_output_total}")

    # Statistics for samples WITH questions
    if len(samples) > 0:
        valid_with_questions = sum(1 for s in samples if s["valid"])
        invalid_with_questions = len(samples) - valid_with_questions
        print("\n  Samples WITH questions:")
        print(
            f"    Valid: {valid_with_questions} ({valid_with_questions / len(samples) * 100:.1f}%)"
        )
        print(f"    Invalid: {invalid_with_questions}")
        if invalid_with_questions > 0:
            # Count invalid reasons for samples with questions
            reasons_with_questions = {}
            for s in samples:
                if not s["valid"]:
                    reason = s["validity_reason"]
                    reasons_with_questions[reason] = (
                        reasons_with_questions.get(reason, 0) + 1
                    )
            print("    Invalid reasons:")
            for reason, count in sorted(
                reasons_with_questions.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"      {reason}: {count}")

    # Statistics for samples WITHOUT questions
    if no_output_reasons:
        print("\n  Samples WITHOUT questions (reasons):")
        for reason, count in sorted(
            no_output_reasons.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"    {reason}: {count}")

    # Empty line at end
    print()

    return samples


def generate_embeddings(
    samples: list[dict],
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    batch_size: int = 128,
) -> np.ndarray:
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
    model = SentenceTransformer(model_name, device=device)

    embeddings_list = []

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
        question_embeddings = model.encode(
            all_questions,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Average embeddings for each sample
        for sample_idx in range(len(batch)):
            # Get embeddings for this sample's questions
            mask = [idx == sample_idx for idx in sample_indices]
            sample_embeddings = question_embeddings[mask]

            # Average across all variations
            avg_embedding = np.mean(sample_embeddings, axis=0)

            embeddings_list.append(avg_embedding)

    embeddings = np.vstack(embeddings_list)
    print(
        f"\nGenerated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}"
    )

    return embeddings


def save_results(
    embeddings: np.ndarray,
    samples: list[dict],
    output_dir: Path,
    model_name: str,
) -> None:
    """Save embeddings as numpy array and metadata to disk."""
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save embeddings as numpy array
    print("\nSaving embeddings...")
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")

    # Save metadata as JSON
    samples_path = output_dir / "samples.json"
    dump_json(samples, samples_path)
    print(f"Saved samples to {samples_path}")

    # Save a summary JSON
    summary = {
        "num_samples": len(embeddings),
        "embedding_dim": embeddings.shape[1],
        "model": model_name,
        "normalized": True,
    }

    summary_path = output_dir / "summary.json"
    dump_json(summary, summary_path)
    print(f"Saved summary to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, embed, and format Wikidata query-SPARQL samples"
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
        "--data-dir",
        type=Path,
        default=Path("data/organic-qwen3-next-80b-a3b"),
        help="Input directory containing JSON files (default: "
        "data/organic-qwen3-next-80b-a3b)",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/organic.jsonl"),
        help="Input JSONL file with original inputs (default: data/organic.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/organic-qwen3-next-80b-a3b-dataset"),
        help="Output directory for embeddings (default: data/organic-qwen3-next-80b-a3b-dataset)",
    )

    args = parser.parse_args()

    # Load samples
    print("Step 1: Loading JSON samples...")
    samples = load_json_samples(args.input_file, args.data_dir)
    print(f"Loaded {len(samples)} samples")

    if len(samples) == 0:
        print("No samples found. Exiting.")
        return

    # Generate embeddings
    print("\nStep 2: Generating embeddings...")
    embeddings = generate_embeddings(
        samples,
        model_name=args.model,
        batch_size=args.batch_size,
    )

    # Save results
    print("\nStep 3: Saving results...")
    save_results(embeddings, samples, args.output_dir, args.model)

    print("\n✓ Embedding generation complete!")


if __name__ == "__main__":
    main()
