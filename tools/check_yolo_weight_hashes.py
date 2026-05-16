"""Compare YOLO checkpoint parameters by tensor content, not file hash."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class WeightSummary:
    """Tensor-level summary for one checkpoint."""

    index: int
    path: Path
    state_hash: str
    tensor_count: int
    total_numel: int
    selected_state: str


@dataclass
class PairComparison:
    """Pairwise parameter comparison summary."""

    weight_a: Path
    weight_b: Path
    state_hash_a: str
    state_hash_b: str
    same_keys: bool
    checked_tensors: int
    missing_or_extra_keys: int
    diff_tensors: int
    overall_max_abs_diff: float
    warning: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Check whether YOLO .pt checkpoints have identical tensor parameters."
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        required=True,
        help="One or more YOLO .pt checkpoint paths to compare.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path, for example results/tables/yolo_weight_hash_check.csv.",
    )
    return parser.parse_args()


def load_torch() -> Any:
    """Import torch lazily so --help and py_compile stay lightweight."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required to inspect .pt checkpoint tensors.") from exc
    return torch


def torch_load_checkpoint(path: Path) -> Any:
    """Load a checkpoint on CPU."""
    torch = load_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def is_tensor(value: Any) -> bool:
    """Return whether value is a torch tensor without importing torch at module load."""
    torch = load_torch()
    return isinstance(value, torch.Tensor)


def tensor_state_dict(mapping: dict[str, Any]) -> dict[str, Any]:
    """Keep only tensor values from a mapping."""
    return {str(key): value for key, value in mapping.items() if is_tensor(value)}


def extract_state_dict(checkpoint: Any) -> tuple[dict[str, Any], str]:
    """Extract the real model parameter state dict from common YOLO checkpoint forms."""
    if hasattr(checkpoint, "state_dict"):
        return tensor_state_dict(checkpoint.state_dict()), "checkpoint.state_dict"

    if isinstance(checkpoint, dict):
        direct_tensors = tensor_state_dict(checkpoint)
        if direct_tensors:
            return direct_tensors, "checkpoint"

        for key in ("ema", "model"):
            value = checkpoint.get(key)
            if value is not None and hasattr(value, "state_dict"):
                state = tensor_state_dict(value.state_dict())
                if state:
                    return state, key

        for key in ("state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                state = tensor_state_dict(value)
                if state:
                    return state, key

    raise ValueError("Could not find tensor state_dict in checkpoint.")


def tensor_bytes(tensor: Any) -> bytes:
    """Return deterministic raw tensor bytes from CPU contiguous memory."""
    tensor = tensor.detach().cpu().contiguous()
    return tensor.numpy().tobytes()


def compute_state_hash(state_dict: dict[str, Any]) -> tuple[str, int]:
    """Compute a hash from tensor names, dtype, shape, and tensor bytes."""
    digest = hashlib.sha256()
    total_numel = 0
    for key in sorted(state_dict):
        tensor = state_dict[key]
        total_numel += int(tensor.numel())
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor_bytes(tensor))
    return digest.hexdigest(), total_numel


def load_weight_summary(path: Path, index: int) -> tuple[WeightSummary, dict[str, Any]]:
    """Load one checkpoint and return its tensor summary plus state dict."""
    if not path.is_file():
        raise FileNotFoundError(f"Weight file does not exist: {path}")
    checkpoint = torch_load_checkpoint(path)
    state_dict, selected_state = extract_state_dict(checkpoint)
    if not state_dict:
        raise ValueError(f"No tensor parameters found in {path}")
    state_hash, total_numel = compute_state_hash(state_dict)
    return (
        WeightSummary(
            index=index,
            path=path,
            state_hash=state_hash,
            tensor_count=len(state_dict),
            total_numel=total_numel,
            selected_state=selected_state,
        ),
        state_dict,
    )


def tensor_max_abs_diff(tensor_a: Any, tensor_b: Any) -> float:
    """Return max absolute difference for two same-shape tensors."""
    if int(tensor_a.numel()) == 0:
        return 0.0
    diff = (tensor_a.detach().cpu().float() - tensor_b.detach().cpu().float()).abs()
    return float(diff.max().item())


def compare_states(
    summary_a: WeightSummary,
    state_a: dict[str, Any],
    summary_b: WeightSummary,
    state_b: dict[str, Any],
) -> PairComparison:
    """Compare two tensor state dicts."""
    keys_a = set(state_a)
    keys_b = set(state_b)
    common_keys = sorted(keys_a & keys_b)
    missing_or_extra_keys = len(keys_a ^ keys_b)
    same_keys = missing_or_extra_keys == 0

    checked_tensors = 0
    changed_tensors = missing_or_extra_keys
    overall_max_abs_diff = 0.0

    for key in common_keys:
        tensor_a = state_a[key]
        tensor_b = state_b[key]
        checked_tensors += 1

        if tuple(tensor_a.shape) != tuple(tensor_b.shape) or tensor_a.dtype != tensor_b.dtype:
            changed_tensors += 1
            continue

        max_abs_diff = tensor_max_abs_diff(tensor_a, tensor_b)
        overall_max_abs_diff = max(overall_max_abs_diff, max_abs_diff)
        if max_abs_diff != 0.0:
            changed_tensors += 1

    warning = ""
    if same_keys and checked_tensors > 0 and changed_tensors == 0 and overall_max_abs_diff == 0.0:
        warning = "IDENTICAL_WEIGHTS"

    return PairComparison(
        weight_a=summary_a.path,
        weight_b=summary_b.path,
        state_hash_a=summary_a.state_hash,
        state_hash_b=summary_b.state_hash,
        same_keys=same_keys,
        checked_tensors=checked_tensors,
        missing_or_extra_keys=missing_or_extra_keys,
        diff_tensors=changed_tensors,
        overall_max_abs_diff=overall_max_abs_diff,
        warning=warning,
    )


def write_csv(
    output_path: Path,
    summaries: list[WeightSummary],
    comparisons: list[PairComparison],
) -> None:
    """Write checkpoint summaries and pairwise comparisons to one CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_type",
        "weight_index",
        "weight_path",
        "state_hash",
        "tensor_count",
        "total_numel",
        "selected_state",
        "weight_a",
        "weight_b",
        "state_hash_a",
        "state_hash_b",
        "same_keys",
        "checked_tensors",
        "missing_or_extra_keys",
        "diff_tensors",
        "overall_max_abs_diff",
        "warning",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "record_type": "weight",
                    "weight_index": summary.index,
                    "weight_path": summary.path.as_posix(),
                    "state_hash": summary.state_hash,
                    "tensor_count": summary.tensor_count,
                    "total_numel": summary.total_numel,
                    "selected_state": summary.selected_state,
                }
            )
        for comparison in comparisons:
            writer.writerow(
                {
                    "record_type": "comparison",
                    "weight_a": comparison.weight_a.as_posix(),
                    "weight_b": comparison.weight_b.as_posix(),
                    "state_hash_a": comparison.state_hash_a,
                    "state_hash_b": comparison.state_hash_b,
                    "same_keys": str(comparison.same_keys).lower(),
                    "checked_tensors": comparison.checked_tensors,
                    "missing_or_extra_keys": comparison.missing_or_extra_keys,
                    "diff_tensors": comparison.diff_tensors,
                    "overall_max_abs_diff": f"{comparison.overall_max_abs_diff:.12g}",
                    "warning": comparison.warning,
                }
            )


def print_results(summaries: list[WeightSummary], comparisons: list[PairComparison]) -> None:
    """Print concise human-readable results."""
    print("YOLO checkpoint tensor hash summary:")
    for summary in summaries:
        print(f"- [{summary.index}] {summary.path.as_posix()}")
        print(f"  state_hash: {summary.state_hash}")
        print(f"  selected_state: {summary.selected_state}")
        print(f"  tensor_count: {summary.tensor_count}")
        print(f"  total_numel: {summary.total_numel}")

    print("Pairwise tensor comparisons:")
    for comparison in comparisons:
        print(f"- {comparison.weight_a.as_posix()}  vs  {comparison.weight_b.as_posix()}")
        print(f"  same_keys: {comparison.same_keys}")
        print(f"  checked_tensors: {comparison.checked_tensors}")
        print(f"  diff_tensors: {comparison.diff_tensors}")
        print(f"  overall_max_abs_diff: {comparison.overall_max_abs_diff:.12g}")
        if comparison.warning:
            print(
                "[WARNING] Two checkpoints have identical tensor parameters. "
                "Do not trust downstream three-model comparisons until retraining is fixed."
            )


def run(args: argparse.Namespace) -> tuple[list[WeightSummary], list[PairComparison]]:
    """Run checkpoint tensor hash checks."""
    weight_paths = [Path(item) for item in args.weights]
    if len(weight_paths) < 2:
        raise ValueError("At least two --weights are required for comparison.")

    summaries: list[WeightSummary] = []
    states: list[dict[str, Any]] = []
    for index, path in enumerate(weight_paths, start=1):
        summary, state_dict = load_weight_summary(path, index)
        summaries.append(summary)
        states.append(state_dict)

    comparisons: list[PairComparison] = []
    for index_a, index_b in itertools.combinations(range(len(summaries)), 2):
        comparisons.append(
            compare_states(
                summaries[index_a],
                states[index_a],
                summaries[index_b],
                states[index_b],
            )
        )

    if args.output:
        output_path = Path(args.output)
        write_csv(output_path, summaries, comparisons)
        print(f"[INFO] Wrote CSV: {output_path.as_posix()}")

    return summaries, comparisons


def main() -> int:
    """Run the command-line entry point."""
    args = parse_args()
    try:
        summaries, comparisons = run(args)
        print_results(summaries, comparisons)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
