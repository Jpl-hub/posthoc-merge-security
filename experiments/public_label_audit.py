"""Audit whether public training labels reproduce frozen hard-label recoveries.

No new defense, teacher query, search, or full test is performed. Only identical
labels AND identical recomputed prototypes permit aliasing validated predictions.
The public-label attacker is reported separately from author AGA's protocol.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import struct
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from eight_task_benchmark import digest
from path_subspace import read
from prototype_recovery_gate import author_classifier
from query_partner_cost import runtime
from subspace_probe import save_json
from torchvision.datasets import MNIST


def tensor_digest(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def run(args):
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    runtime()
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "protocol.json", cfg)
    data = MNIST("data", train=True, download=False)
    label_file = Path("data/MNIST/raw/train-labels-idx1-ubyte.gz")
    idx = gzip.decompress(label_file.read_bytes())
    magic, count = struct.unpack(">II", idx[:8])
    assert magic == 2049 and count == len(data) == 60000
    independent = torch.tensor(list(idx[8:]), dtype=torch.int64)
    assert torch.equal(independent, data.targets)
    records, checks, input_pins = [], {}, {str(label_file): digest(label_file)}
    for name in cfg["parents"]:
        parent = Path(name)
        protocol, meta = read(parent / "protocol.json"), read(parent / "metadata.json")
        assert protocol["target"] == "MNIST" and meta["completed"]
        validation = read(parent / "validation.json")
        assert validation["status"] == "passed"
        assert digest(parent / "results.json") == validation["results_sha256"]
        assert digest(parent / "metadata.json") == validation["metadata_sha256"]
        for filename in [
            "protocol.json",
            "metadata.json",
            "validation.json",
            "results.json",
            "attack_indices.json",
        ]:
            input_pins[str(parent / filename)] = digest(parent / filename)
        for filename, sha in meta["raw_sha256"].items():
            assert digest(parent / "artifacts" / filename) == sha
            input_pins[str(parent / "artifacts" / filename)] = sha
        indices = read(parent / "attack_indices.json")
        assert len(indices) == len(set(indices)) == 256
        labels = data.targets[indices]
        certificate = torch.load(
            parent / "artifacts/certificate.pt", map_location="cpu", weights_only=True
        )
        features = certificate["features"]
        # Attack input contains public labels and frozen student features only.
        primitive = author_classifier(protocol)(10, features.shape[1], device="cuda")
        primitive.update(features.cuda(), labels.cuda())
        prototypes = primitive.get_prototypes().cpu()
        counts = torch.bincount(labels, minlength=10)
        assert counts.min() > 0
        sums = torch.zeros(10, features.shape[1], dtype=torch.float64)
        sums.index_add_(0, independent[indices], features)
        check = F.normalize(sums / counts[:, None], dim=1)
        independent_error = float((check - prototypes).abs().max())
        assert independent_error < 1e-6
        identical = {}
        for preset in ["original", "protected"]:
            evidence = certificate[preset]
            label_equal = torch.equal(labels, evidence["teacher_predictions"])
            prototype_equal = torch.equal(prototypes, evidence["prototypes"])
            identical[preset] = label_equal and prototype_equal
            checks[f"{name}/{preset}"] = dict(
                teacher_label_mismatches=int(labels.ne(evidence["teacher_predictions"]).sum()),
                prototype_bitwise_equal=prototype_equal,
                independent_index_aggregation_error=independent_error,
            )
        raw = torch.load(parent / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
        for row in read(parent / "results.json"):
            if row["preset"] != "original" or not identical[row["preset"]]:
                continue
            value = raw[row["key"]]
            assert row["n"] == len(value["labels"])
            assert row["correct"] == int(value["prediction"].eq(value["labels"]).sum())
            assert row["top1"] == row["correct"] / row["n"]
            records.append(
                dict(
                    parent=name,
                    parent_key=row["key"],
                    group=row["group"],
                    dataset=row["dataset"],
                    correct=row["correct"],
                    n=row["n"],
                    top1=row["top1"],
                    alias=True,
                    target_parameters_used_in_student=meta["target_parameters_used_in_student"],
                    new_teacher_queries=0,
                )
            )
        checks[name] = dict(
            samples=256,
            public_class_counts=counts.tolist(),
            public_labels_sha256=tensor_digest(labels),
            same_public_label_indices_both_teachers=True,
            identical_recovery_both_teachers=all(identical.values()),
            full_erasure_scope=not meta["target_parameters_used_in_student"],
        )
    save_json(out / "results.json", records)
    save_json(out / "label_checks.json", checks)
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            elapsed_seconds=time.perf_counter() - start,
            source_manifest=read(args.manifest)["source_manifest"],
            input_sha256=input_pins,
            original_idx_labels_independently_parsed=True,
            full_training_labels_sha256=tensor_digest(data.targets),
            full_training_images_sha256=tensor_digest(data.data),
            original_parents_previously_prefix_validated=True,
            prediction_aliases=len(records),
            fresh_full_evaluations=0,
            teacher_queries=0,
            public_training_labels_used=True,
            new_defense=False,
            defense_success_claim=False,
            results_sha256=digest(out / "results.json"),
            label_checks_sha256=digest(out / "label_checks.json"),
        ),
    )
    print("PUBLIC_LABEL_AUDIT", checks, "aliases", len(records), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/public_label_audit.json")
    parser.add_argument("--output", default="outputs/2026-10-12/public_label_audit")
    parser.add_argument("--manifest", required=True)
    run(parser.parse_args())
