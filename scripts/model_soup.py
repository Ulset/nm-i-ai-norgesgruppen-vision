"""Average weights from multiple YOLO checkpoints (model soup).

Model soups typically outperform individual checkpoints by averaging
the weights from several training snapshots (e.g., last few epochs
and best checkpoint).

Usage:
  python3 scripts/model_soup.py \
    --checkpoints epoch80.pt epoch90.pt epoch100.pt best.pt \
    --output soup.pt \
    --model yolo11x
"""

import argparse
import copy
from pathlib import Path

import torch

_torch_load = torch.load
torch.load = lambda *args, **kwargs: _torch_load(*args, **{**kwargs, "weights_only": False})


def extract_state_dict(ckpt):
    """Extract model state_dict from an ultralytics checkpoint.

    Ultralytics saves checkpoints as nested dicts with a 'model' key
    that holds the nn.Module. This function handles both raw state_dicts
    and wrapped ultralytics formats.
    """
    if isinstance(ckpt, dict):
        # Ultralytics format: {'model': <DetectionModel>, 'ema': ..., ...}
        if "model" in ckpt:
            model_obj = ckpt["model"]
            if hasattr(model_obj, "state_dict"):
                return model_obj.float().state_dict()
            elif isinstance(model_obj, dict):
                return model_obj
        # Plain state_dict
        if any(k.startswith("model.") for k in ckpt.keys()):
            return ckpt
        # EMA variant
        if "ema" in ckpt and ckpt["ema"] is not None:
            ema = ckpt["ema"]
            if hasattr(ema, "state_dict"):
                return ema.float().state_dict()
    raise ValueError("Could not extract state_dict from checkpoint")


def average_checkpoints(checkpoint_paths):
    """Average the weights of multiple checkpoints element-wise.

    Returns:
        avg_state_dict: Averaged state dict.
        full_ckpt: A copy of the first checkpoint with averaged weights
                   injected back (for ultralytics compatibility).
        num_params: Total number of parameters averaged.
    """
    print(f"Loading {len(checkpoint_paths)} checkpoints...")

    state_dicts = []
    first_ckpt = None

    for i, path in enumerate(checkpoint_paths):
        print(f"  [{i+1}/{len(checkpoint_paths)}] Loading {path}")
        ckpt = torch.load(str(path), map_location="cpu")
        if first_ckpt is None:
            first_ckpt = ckpt
        sd = extract_state_dict(ckpt)
        state_dicts.append(sd)

    # Verify all checkpoints have the same architecture
    ref_keys = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], start=2):
        current_keys = set(sd.keys())
        if current_keys != ref_keys:
            missing = ref_keys - current_keys
            extra = current_keys - ref_keys
            msg = f"Checkpoint {i} has different keys."
            if missing:
                msg += f" Missing: {list(missing)[:5]}..."
            if extra:
                msg += f" Extra: {list(extra)[:5]}..."
            raise ValueError(msg)

    print(f"All checkpoints have {len(ref_keys)} matching parameter keys.")

    # Average weights element-wise
    n = len(state_dicts)
    avg_sd = {}
    num_params = 0

    for key in state_dicts[0].keys():
        tensors = [sd[key] for sd in state_dicts]
        if tensors[0].is_floating_point():
            avg_sd[key] = torch.stack(tensors).mean(dim=0)
        else:
            # Non-float tensors (e.g., batch norm num_batches_tracked): take from first
            avg_sd[key] = tensors[0]
        num_params += avg_sd[key].numel()

    print(f"Averaged {num_params:,} parameters across {n} checkpoints.")

    # Inject averaged weights back into the first checkpoint structure
    full_ckpt = copy.deepcopy(first_ckpt)
    if isinstance(full_ckpt, dict) and "model" in full_ckpt:
        model_obj = full_ckpt["model"]
        if hasattr(model_obj, "load_state_dict"):
            model_obj.float().load_state_dict(avg_sd)
        # Clear EMA to avoid confusion
        if "ema" in full_ckpt:
            full_ckpt["ema"] = None
    else:
        full_ckpt = avg_sd

    return avg_sd, full_ckpt, num_params


def validate_soup(output_path, model_name):
    """Load the soup with ultralytics YOLO and run a quick test."""
    from ultralytics import YOLO

    print(f"\nValidating soup with YOLO('{output_path}')...")
    model = YOLO(str(output_path))

    # Quick sanity check: run a forward pass with dummy data
    import numpy as np
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = model.predict(dummy, verbose=False)
    print(f"Validation OK: inference produced {len(results)} result(s)")
    print(f"Model has {sum(p.numel() for p in model.model.parameters()):,} parameters")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Average weights from multiple YOLO checkpoints (model soup)"
    )
    parser.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="Paths to .pt checkpoint files to average"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to save the averaged checkpoint"
    )
    parser.add_argument(
        "--model", default="yolo11x",
        help="Model architecture name for validation (default: yolo11x)"
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip validation step"
    )
    args = parser.parse_args()

    # Verify all checkpoint files exist
    for path in args.checkpoints:
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Average checkpoints
    avg_sd, full_ckpt, num_params = average_checkpoints(args.checkpoints)

    # Save
    torch.save(full_ckpt, str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved soup to {output_path} ({size_mb:.1f} MB)")

    # Validate
    if not args.no_validate:
        validate_soup(output_path, args.model)

    print("\nDone!")


if __name__ == "__main__":
    main()
