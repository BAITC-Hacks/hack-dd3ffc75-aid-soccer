"""Export reviewed exploration choices to supervised fine-tuning JSONL, offline."""
import argparse
import json
import math
from pathlib import Path

from llm_advisor import (INSTRUCTIONS, PROMPT_VERSION, canonical_json, context_hash,
                         scenario_group, validate_recommendation)


def build_training_split(experiences, reviews, *, validation_fraction=0.2):
    """Require reviewed labels, independent groups and >=10 training examples.

    No model-generated suggestion or positive mock score is auto-approved.
    Identical contexts are deduplicated and related resource/seed variants
    stay in one split. This prepares files; it never uploads or trains a model.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    contexts = {}
    for item in experiences:
        context = item["context"]
        key = context_hash(context)
        if key != item["context_hash"] or context.get("prompt_version") != PROMPT_VERSION:
            raise ValueError("Context hash or prompt version mismatch")
        contexts[key] = context
    approved = {}
    for review in reviews:
        if review.get("approved") is not True:
            continue
        key = review.get("context_hash")
        reviewer = review.get("reviewer")
        if key not in contexts or not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("Approved review needs a known context and named reviewer")
        decision = validate_recommendation({"candidate_ids": review.get("candidate_ids"),
                                             "reason": review.get("reason")}, contexts[key])
        if key in approved and approved[key] != decision:
            raise ValueError("Conflicting approved labels for the same context")
        approved[key] = decision
    groups = {}
    for key in sorted(approved):
        groups.setdefault(scenario_group(contexts[key]), []).append(key)
    if len(groups) < 2:
        raise ValueError("Need reviewed examples from at least two independent scenario groups; repeated seeds do not count")
    validation_count = max(1, min(len(groups) - 1, math.ceil(len(groups) * validation_fraction)))
    validation_groups = set(sorted(groups)[:validation_count])
    train, validation, manifest = [], [], []
    for group, keys in sorted(groups.items()):
        destination = validation if group in validation_groups else train
        split = "validation" if group in validation_groups else "train"
        for key in keys:
            destination.append({"messages": [
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": canonical_json(contexts[key])},
                {"role": "assistant", "content": canonical_json(approved[key])},
            ]})
            manifest.append(dict(context_hash=key, scenario_group=group, split=split))
    if len(train) < 10:
        raise ValueError("At least 10 unique approved training examples are required after holding out validation groups")
    return train, validation, manifest


def _read_jsonl(paths):
    return [json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiences", type=Path, nargs="+", required=True)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/training"))
    args = parser.parse_args(argv)
    train, validation, manifest = build_training_split(_read_jsonl(args.experiences), _read_jsonl(args.reviews))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a new directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (("train.jsonl", train), ("validation.jsonl", validation)):
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(canonical_json(record) + "\n")
    (args.output_dir / "manifest.json").write_text(json.dumps(dict(
        prompt_version=PROMPT_VERSION, training_examples=len(train), validation_examples=len(validation),
        examples=manifest, uploaded=False, trained=False), indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(train)} training and {len(validation)} validation examples. No upload or training performed.")


if __name__ == "__main__":
    main()
