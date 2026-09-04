import argparse
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from universal_ml_utils.io import dump_json, dump_jsonl, load_json, load_jsonl

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


def load_questions(questions_file: Path) -> list[list[str]]:
    """Load the questions written by --no-embed, in samples.json order."""
    questions = []
    for i, record in enumerate(load_jsonl(questions_file)):
        if record["index"] != i:
            raise ValueError(
                f"Expected index {i} in line {i + 1}, got {record['index']}. "
                "The questions must be in the same order as samples.json."
            )
        questions.append(record["questions"])

    return questions


def generate_embeddings(
    questions: list[list[str]],
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    batch_size: int = 128,
    devices: list[str] | None = None,
) -> np.ndarray:
    """
    Generate embeddings for all samples by averaging question variations.
    Uses batching across samples for efficiency.

    Args:
        questions: List of the questions of each sample
        model_name: HuggingFace model name
        batch_size: Number of samples to process in each batch
        devices: Devices to run the model on, more than one spreads the
            questions over all of them (default: all CUDA devices, else cpu)

    Returns:
        embeddings: numpy array of shape (n_samples, embedding_dim)
    """
    devices = devices or [
        f"cuda:{i}" for i in range(torch.cuda.device_count())
    ] or ["cpu"]
    print(f"\nLoading model: {model_name}")
    print(f"Using devices: {', '.join(devices)}")
    print(f"Batch size: {batch_size}")
    model = SentenceTransformer(model_name, device=devices[0])

    # A pool is only needed to use more than one device
    pool = model.start_multi_process_pool(devices) if len(devices) > 1 else None
    encode_kwargs = {"pool": pool} if pool is not None else {}

    embeddings_list = []

    print("\nGenerating embeddings with batching...")
    try:
        for i in tqdm(range(0, len(questions), batch_size), desc="Processing batches"):
            batch = questions[i : i + batch_size]

            # Collect all questions from this batch
            all_questions = [q for qs in batch for q in qs]

            # Encode all questions in the batch at once
            question_embeddings = model.encode(
                all_questions,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
                **encode_kwargs,  # type: ignore
            )

            # Average embeddings for each sample
            offset = 0
            for qs in batch:
                sample_embeddings = question_embeddings[offset : offset + len(qs)]
                offset += len(qs)

                # Average across all variations
                embeddings_list.append(np.mean(sample_embeddings, axis=0))
    finally:
        if pool is not None:
            model.stop_multi_process_pool(pool)

    embeddings = np.vstack(embeddings_list)
    print(
        f"\nGenerated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}"
    )

    return embeddings


def save_questions(samples: list[dict], questions_file: Path) -> None:
    """Save the questions of all samples for a separate embedding step."""
    questions_file.parent.mkdir(exist_ok=True, parents=True)
    dump_jsonl(
        (
            {"index": idx, "questions": sample["questions"]}
            for idx, sample in enumerate(samples)
        ),
        questions_file,
    )
    print(f"Saved questions to {questions_file}")


def save_samples(samples: list[dict], output_dir: Path) -> None:
    """Save sample metadata to disk."""
    output_dir.mkdir(exist_ok=True, parents=True)

    samples_path = output_dir / "samples.json"
    dump_json(samples, samples_path)
    print(f"Saved samples to {samples_path}")


def save_results(
    embeddings: np.ndarray,
    output_dir: Path,
    model_name: str,
) -> None:
    """Save embeddings as numpy array and their summary to disk."""
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save embeddings as numpy array
    print("\nSaving embeddings...")
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")

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

    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=None,
        help="Devices to run the embedding model on, more than one spreads the "
        "questions over all of them (default: all CUDA devices, else cpu)",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=None,
        help="Questions of the samples, written by --no-embed and read by "
        "--embed-only (default: <output-dir>/questions.jsonl)",
    )
    # Embedding the questions is the only step that wants a GPU. If the
    # generations live on a machine without one, --no-embed and --embed-only
    # split this script in two, with the questions file as the handover.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--no-embed",
        action="store_true",
        help="Only load the samples and write samples.json and the questions "
        "file, without embedding them",
    )
    mode.add_argument(
        "--embed-only",
        action="store_true",
        help="Only embed the questions written by a previous --no-embed run, "
        "without loading the generations again",
    )

    args = parser.parse_args()

    questions_file = args.questions_file or args.output_dir / "questions.jsonl"

    if args.embed_only:
        print(f"Step 1: Loading questions from {questions_file}...")
        questions = load_questions(questions_file)
        print(f"Loaded questions of {len(questions)} samples")
    else:
        print("Step 1: Loading JSON samples...")
        samples = load_json_samples(args.input_file, args.data_dir)
        print(f"Loaded {len(samples)} samples")

        if len(samples) == 0:
            print("No samples found. Exiting.")
            return

        questions = [sample["questions"] for sample in samples]
        save_samples(samples, args.output_dir)

        if args.no_embed:
            save_questions(samples, questions_file)
            print(
                f"\n✓ Samples ready. Next, embed them with --embed-only "
                f"--questions-file {questions_file}, then make sure "
                f"embeddings.npy ends up in {args.output_dir}"
            )
            return

    print("\nStep 2: Generating embeddings...")
    embeddings = generate_embeddings(
        questions,
        model_name=args.model,
        batch_size=args.batch_size,
        devices=args.devices,
    )

    print("\nStep 3: Saving results...")
    save_results(embeddings, args.output_dir, args.model)

    print("\n✓ Embedding generation complete!")


if __name__ == "__main__":
    main()
