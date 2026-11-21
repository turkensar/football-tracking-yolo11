"""Local/server training entrypoint for fine-tuning YOLO11 on a football dataset."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def resolve_train_device(device: str) -> str:
    """Resolve ``"auto"`` to ``"0"`` when CUDA is available, else ``"cpu"``."""
    cleaned = device.strip()
    if not cleaned:
        raise ValueError("--device must be a non-empty string.")
    if cleaned.lower() == "auto":
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    return cleaned


def resolve_base_weights(weights: str, models_dir: Path = Path("models")) -> str:
    """Reuse ``models/<weights>`` when a bare weight name is already present locally."""
    candidate = Path(weights)
    if candidate.parent == Path(".") and not weights.startswith(("./", ".\\")):
        local_weight = models_dir / candidate.name
        if local_weight.is_file():
            return str(local_weight)
    return weights


def train_model(
    data: Path,
    *,
    weights: str = "yolo11s.pt",
    epochs: int = 50,
    imgsz: int = 1280,
    batch: int = 8,
    device: str = "auto",
    project: Path = Path("runs/train"),
    name: str = "football_yolo11s",
    patience: int = 15,
    seed: int = 42,
) -> Any:
    """Fine-tune a YOLO11 model on a YOLO-format dataset and return Ultralytics results.

    Args:
        data: Path to the dataset ``data.yaml`` file.
        weights: Base pretrained weights filename or path (default: ``yolo11s.pt``).
        epochs: Number of training epochs (> 0).
        imgsz: Square training image size in pixels (> 0).
        batch: Batch size (> 0) or ``-1`` for Ultralytics AutoBatch.
        device: Target device (``"auto"``, ``"0"``, ``"cpu"``, etc.).
        project: Output directory for Ultralytics training runs.
        name: Run subdirectory name under ``project``.
        patience: Early-stopping patience in epochs (>= 0).
        seed: Random seed for reproducible training.

    Returns:
        The result object returned by ``ultralytics.YOLO.train``.

    Raises:
        FileNotFoundError: If ``data`` does not exist.
        ValueError: If any hyperparameter is out of range.
    """
    data_path = Path(data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    if epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {epochs}")
    if imgsz <= 0:
        raise ValueError(f"imgsz must be > 0, got {imgsz}")
    if batch != -1 and batch <= 0:
        raise ValueError(f"batch must be > 0 or -1 (auto), got {batch}")
    if patience < 0:
        raise ValueError(f"patience must be >= 0, got {patience}")
    if not name.strip():
        raise ValueError("name must be a non-empty string.")

    resolved_device = resolve_train_device(device)
    resolved_weights = resolve_base_weights(weights)

    from ultralytics import YOLO

    model = YOLO(resolved_weights)
    return model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=resolved_device,
        project=str(project),
        name=name,
        patience=patience,
        seed=seed,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for ``scripts/train.py``."""
    parser = argparse.ArgumentParser(
        prog="python scripts/train.py",
        description="Fine-tune a YOLO11 detector on a football dataset (ball/player/gk/referee).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the YOLO dataset data.yaml file.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11s.pt",
        help="Base pretrained YOLO weights (default: yolo11s.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Training image size in pixels (default: 1280).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size, or -1 for Ultralytics AutoBatch (default: 8).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto, 0, cpu, etc. (default: auto).",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/train"),
        help="Output project directory for runs (default: runs/train).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="football_yolo11s",
        help="Experiment run name (default: football_yolo11s).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early-stopping patience in epochs (default: 15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training (default: 42).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments, launch fine-tuning, and return a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        train_model(
            args.data,
            weights=args.weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            patience=args.patience,
            seed=args.seed,
        )
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Training complete. Outputs saved under: {args.project / args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
