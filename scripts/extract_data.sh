#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/data}"
TRAIN_ARCHIVE="${TRAIN_ARCHIVE:-}"
TEST_ARCHIVE="${TEST_ARCHIVE:-}"
TRAINING_PASSWORD="${TRAINING_PASSWORD:-}"
TEST_PASSWORD="${TEST_PASSWORD:-}"

mkdir -p "$OUTPUT_DIR/training" "$OUTPUT_DIR/test"

extract_archive() {
  archive_path="$1"
  password="$2"
  destination="$3"

  if [ -z "$archive_path" ] || [ ! -f "$archive_path" ]; then
    return 0
  fi

  if [ -n "$password" ]; then
    unzip -o -P "$password" "$archive_path" -d "$destination"
  else
    unzip -o "$archive_path" -d "$destination"
  fi
}

extract_archive "$TRAIN_ARCHIVE" "$TRAINING_PASSWORD" "$OUTPUT_DIR/training"
extract_archive "$TEST_ARCHIVE" "$TEST_PASSWORD" "$OUTPUT_DIR/test"

echo "Extraction complete."
echo "Canonical runtime dataset paths for this repository:"
echo "  training: ../train"
echo "  test:     ../test_done"
