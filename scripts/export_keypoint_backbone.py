"""Export encoder-only weights from a trained MTDDH keypoint checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.keypoint_detector import export_encoder_state_dict, load_keypoint_detector_from_checkpoint


def export_keypoint_backbone(checkpoint_path: str | Path, output_path: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path).resolve()
    output_path = Path(output_path).resolve()
    model, checkpoint = load_keypoint_detector_from_checkpoint(checkpoint_path, device="cpu")
    payload = {
        "architecture": "resnet50",
        "source_checkpoint": str(checkpoint_path),
        "source_experiment": checkpoint.get("training_config", {}).get("experiment"),
        "encoder_state_dict": export_encoder_state_dict(model),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export encoder-only weights from a keypoint checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = export_keypoint_backbone(args.checkpoint, args.output)
    print(json.dumps({"status": "ok", "output": str(output_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
