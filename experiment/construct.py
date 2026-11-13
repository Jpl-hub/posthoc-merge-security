"""Construct a defense from the target/base; no partner or test-based selection."""

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import os
import platform
import resource
import statistics
import subprocess
import sys
import time

import torch

from _impl.baselines import params_default, verify_edits
from _impl.defenses import construct_bias
from _impl.io import check_files, digest, new_output, read_json, save_json
from _impl.models import apply, checked_models, initialize
from cfg import require_root


def benchmark_worker(a, cfg, out):
    """One fresh process: first construction, then one warm construction."""
    torch.set_num_interop_threads(1)
    assert torch.get_num_threads() == cfg["benchmark"]["CPU_threads"]
    assert len(os.sched_getaffinity(0)) == cfg["benchmark"]["CPU_threads"]
    anchor, expert, head, pins = checked_models(a.target)
    models = [anchor, expert, head]

    def fingerprint():
        h = hashlib.sha256()
        for model in models:
            for key, value in model.state_dict().items():
                assert value.device.type == "cpu"
                h.update(key.encode())
                h.update(value.contiguous().numpy().tobytes())
        return h.hexdigest()

    before = fingerprint()
    author = None
    samples = []
    previous_bias = None
    for mode in ["cold", "warm"]:
        gc.collect()
        cpu_start = time.process_time()
        start = time.perf_counter()
        if a.method == "bias":
            edits, details = construct_bias(expert, anchor, head, cfg["bias"])
        else:
            if author is None:
                # Keep this exact author module alive for the warm call/JIT caches.
                spec = importlib.util.spec_from_file_location(
                    "official_params_cost", cfg["params"]["path"])
                author = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = author
                spec.loader.exec_module(author)
            with torch.no_grad():
                state = author.params(
                    expert.state_dict(), anchor.state_dict(), num_layers=12,
                    enable_permutation=True, enable_scaling=True,
                    perm_config=None, scaling_config=None, rng_seed=0)
        seconds = time.perf_counter() - start
        cpu_seconds = time.process_time() - cpu_start
        # The author's output is materialized CPU torch tensors (JAX -> NumPy
        # occurs inside author.params), so no asynchronous JAX work remains.
        if a.method == "params":
            assert all(v.device.type == "cpu" and torch.isfinite(v).all()
                       for v in state.values())
            original = expert.state_dict()
            edits = {k: v for k, v in state.items() if not torch.equal(v, original[k])}
            details = verify_edits(expert, edits)
            del state, original
        elif previous_bias is not None:
            assert set(edits) == set(previous_bias)
            assert all(torch.equal(v, previous_bias[k]) for k, v in edits.items())
            details["cold_warm_exact_bias_equality"] = True
        if a.method == "bias":
            previous_bias = edits
        sample = dict(mode=mode, repetition=a.benchmark_worker,
                      kernel_seconds=seconds, process_cpu_seconds=cpu_seconds,
                      edited_tensors=len(edits),
                      edited_values=sum(v.numel() for v in edits.values()),
                      update_bytes=sum(v.numel() * v.element_size() for v in edits.values()),
                      process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      details=details)
        samples.append(sample)
        save_json(out / "timing.json", dict(samples=samples, complete=False))
        print("CONSTRUCTION_TIMING", a.method, a.target, mode,
              a.benchmark_worker, seconds, flush=True)
        del edits
    assert fingerprint() == before, "Construction modified its frozen inputs"
    packages = ["torch", "numpy", "scipy"] + (["jax", "jaxlib"] if a.method == "params" else [])
    save_json(out / "timing.json", dict(
        method=a.method, target=a.target, repetition=a.benchmark_worker,
        samples=samples, complete=True, checkpoint_sha256=pins,
        frozen_input_tensors_unchanged=True, construction_device="cpu",
        CPU_threads=torch.get_num_threads(), CPU_interop_threads=torch.get_num_interop_threads(),
        CPU_affinity=sorted(os.sched_getaffinity(0)), host=platform.node(),
        platform=platform.platform(), python=platform.python_version(),
        packages={name: importlib.metadata.version(name) for name in packages},
        images=0, expert_updates=0, GPU_kernels=0, new_publication_files=0))


def repeated_benchmark(a, cfg, out):
    """Five independent cold/warm pairs; child startup and model IO are untimed."""
    b = cfg["benchmark"]
    if a.target not in b["targets"] or a.method not in b["methods"]:
        raise ValueError("Method/target is outside the frozen benchmark protocol")
    cpus = sorted(os.sched_getaffinity(0))[:b["CPU_threads"]]
    if len(cpus) != b["CPU_threads"]:
        raise ValueError("The benchmark requires four available logical CPUs")
    env = os.environ | b["environment"]
    samples = []
    workers = []
    for repetition in range(b["repetitions"]):
        destination = out / f"repeat_{repetition:02d}"
        subprocess.run([
            "taskset", "--cpu-list", ",".join(map(str, cpus)),
            sys.executable, "-m", "experiment.construct", "--benchmark",
            "--method", a.method, "--target", a.target, "--protocol", a.protocol,
            "--output", str(destination), "--benchmark-worker", str(repetition),
        ], env=env, check=True)
        worker = read_json(destination / "timing.json")
        assert worker["complete"] and worker["CPU_affinity"] == cpus
        samples.extend(worker["samples"])
        workers.append(worker)
    summary = {}
    for mode in ["cold", "warm"]:
        values = [s["kernel_seconds"] for s in samples if s["mode"] == mode]
        assert len(values) == b["repetitions"]
        summary[mode] = dict(repetitions=len(values), seconds=values,
                             mean=statistics.mean(values), median=statistics.median(values),
                             sample_stddev=statistics.stdev(values),
                             minimum=min(values), maximum=max(values))
    save_json(out / "timing.json", dict(
        method=a.method, target=a.target, summary=summary, workers=workers,
        construction_device="cpu", CPU_threads=b["CPU_threads"], CPU_affinity=cpus,
        environment=b["environment"], timing_scope=b["timing_scope"],
        images=0, expert_updates=0, GPU_kernels=0, new_publication_files=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--method",
        choices=["bias", "params"],
        default="bias",
    )
    p.add_argument("--target", choices=["SUN397", "Cars", "RESISC45", "SVHN", "GTSRB", "MNIST", "DTD"], required=True)
    p.add_argument("--protocol", default="configs/construct.json")
    p.add_argument("--output", required=True)
    p.add_argument("--benchmark", action="store_true", help="Time the weight-only kernel; save metadata without a new publication")
    p.add_argument("--benchmark-worker", type=int, help=argparse.SUPPRESS)
    p.add_argument("--publication", help="Apply PaRaMS to an already frozen publication")
    p.add_argument("--publication-sha256")
    a = p.parse_args()
    if bool(a.publication) != bool(a.publication_sha256) or (a.publication and a.method != "params"):
        p.error("A frozen publication and its SHA256 must be supplied together, for params only")
    if a.benchmark and a.method not in {"bias", "params"}:
        p.error("Benchmark supports bias and params")
    require_root()
    initialize()
    cfg = read_json(a.protocol)
    check_files(cfg["source_sha256"])
    if a.benchmark and cfg.get("benchmark", {}).get("timing_modes") == ["cold", "warm"]:
        if a.publication:
            p.error("Repeated construction timings use the frozen expert checkpoint")
        if digest(cfg["params"]["path"]) != cfg["params"]["sha256"]:
            raise ValueError("PaRaMS source SHA256 mismatch")
        out = new_output(a.output, dict(method=a.method, target=a.target, configuration=cfg))
        if a.benchmark_worker is None:
            repeated_benchmark(a, cfg, out)
        else:
            benchmark_worker(a, cfg, out)
        return
    if a.benchmark_worker is not None:
        p.error("Benchmark workers require the cold/warm timing protocol")
    out = new_output(a.output, dict(method=a.method, target=a.target, configuration=cfg))
    anchor, expert, head, pins = checked_models(a.target)
    start = time.monotonic()
    publication_reference = "expert"
    if a.method == "bias":
        edits, details = construct_bias(expert, anchor, head, cfg["bias"])
        kernel_seconds = time.monotonic() - start
    else:
        c = cfg["params"]
        ref = dict(path=a.publication, sha256=a.publication_sha256) if a.publication else None
        source = apply(expert, ref) if ref else expert
        edits = params_default(source, anchor, c["path"], c["sha256"])
        kernel_seconds = time.monotonic() - start
        details = verify_edits(source, edits)
        if ref:
            details["input_publication"] = ref
            edits = {k: v for k, v in (source.state_dict() | edits).items()
                     if not torch.equal(v, expert.state_dict()[k])}
    if a.benchmark:
        save_json(out / "timing.json", dict(method=a.method, target=a.target,
            kernel_seconds=kernel_seconds, construction_device="cpu",
            CPU_threads=torch.get_num_threads(), repetitions=1,
            excludes="Checkpoint loading, correctness verification and serialization",
            edited_tensors=len(edits), edited_values=sum(v.numel() for v in edits.values()),
            update_bytes=sum(v.numel() * v.element_size() for v in edits.values()),
            checkpoint_sha256=pins, details=details, images=0, expert_updates=0,
            new_publication_files=0))
        print("CONSTRUCTION_TIMING", a.method, a.target, kernel_seconds, flush=True)
        return
    torch.save(edits, out / "release.pt")
    save_json(
        out / "before_images.json",
        dict(
            release_sha256=digest(out / "release.pt"),
            publication_reference=publication_reference,
            checkpoint_sha256=pins,
            method=a.method,
            target=a.target,
            seconds=time.monotonic() - start,
            details=details,
            expert_training_updates=0,
        ),
    )


if __name__ == "__main__":
    main()
