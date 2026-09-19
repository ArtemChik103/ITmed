"""Generate default keypoint checkpoint if not present."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.phase3_helpers import write_synthetic_keypoint_checkpoint

def main():
    target_path = PROJECT_ROOT / "models" / "checkpoints" / "keypoint_detector.pt"
    if not target_path.exists():
        print(f"Generating keypoint detector checkpoint at {target_path}...")
        write_synthetic_keypoint_checkpoint(target_path, input_size=384, heatmap_size=96)
        print("Done!")
    else:
        print(f"Keypoint checkpoint already exists at {target_path}")

if __name__ == "__main__":
    main()
