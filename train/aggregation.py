"""Object-level aggregation helpers for Phase 3 experiments."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

AGGREGATION_METHODS = ("max", "mean", "logit_mean", "topk_mean")
DEFAULT_TOP_K = 3
_EPSILON = 1e-6
def merge_predictions_with_manifest(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Attach manifest metadata to per-sample predictions."""
    if "sample_id" not in predictions.columns:
        raise ValueError("Predictions dataframe must include 'sample_id'.")
    if "probability" not in predictions.columns:
        raise ValueError("Predictions dataframe must include 'probability'.")
    if "group_id" not in manifest.columns:
        raise ValueError("Manifest dataframe must include 'group_id'.")

    predictions_frame = predictions.copy()
    predictions_frame["sample_id"] = predictions_frame["sample_id"].astype(str)
    predictions_frame["probability"] = predictions_frame["probability"].astype(float)

    manifest_frame = manifest.drop_duplicates(subset=["sample_id"]).copy()
    manifest_frame["sample_id"] = manifest_frame["sample_id"].astype(str)
    if "group_id" in manifest_frame.columns:
        manifest_frame["group_id"] = manifest_frame["group_id"].astype(str)
    additional_columns = [
        column
        for column in manifest_frame.columns
        if column == "sample_id" or column not in predictions_frame.columns
    ]
    manifest_frame = manifest_frame[additional_columns]

    merged = predictions_frame.merge(
        manifest_frame,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if merged["group_id"].isna().any():
        missing_count = int(merged["group_id"].isna().sum())
        raise ValueError(f"Missing manifest metadata for {missing_count} prediction rows.")

    if "target" in merged.columns:
        merged["target"] = merged["target"].astype(int)
    elif "label" in merged.columns:
        merged["target"] = merged["label"].astype(int)
    else:
        raise ValueError("Predictions dataframe must include 'target' or manifest must include 'label'.")

    if "label" in merged.columns:
        if not np.array_equal(merged["target"].to_numpy(dtype=int), merged["label"].to_numpy(dtype=int)):
            raise ValueError("Prediction targets do not match manifest labels.")

    return merged.sort_values(["group_id", "sample_id"]).reset_index(drop=True)


def aggregate_probability(
    probabilities: Iterable[float],
    *,
    method: str,
    top_k: int = DEFAULT_TOP_K,
) -> float:
    """Collapse sample-level probabilities into a single object-level probability."""
    values = np.asarray(list(probabilities), dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot aggregate an empty probability list.")

    if method == "max":
        return float(values.max())
    if method == "mean":
        return float(values.mean())
    if method == "logit_mean":
        clipped = np.clip(values, _EPSILON, 1.0 - _EPSILON)
        logits = np.log(clipped / (1.0 - clipped))
        mean_logit = float(logits.mean())
        return float(1.0 / (1.0 + np.exp(-mean_logit)))
    if method == "topk_mean":
        k = max(1, min(int(top_k), values.size))
        top_values = np.sort(values)[-k:]
        return float(top_values.mean())

    raise ValueError(f"Unsupported aggregation method: {method}")


def build_group_prediction_table(
    predictions: pd.DataFrame,
    *,
    group_key: str = "group_id",
    methods: Iterable[str] = AGGREGATION_METHODS,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Aggregate sample-level predictions into object-level rows."""
    if group_key not in predictions.columns:
        raise ValueError(f"Predictions dataframe must include '{group_key}'.")
    if "sample_id" not in predictions.columns:
        raise ValueError("Predictions dataframe must include 'sample_id'.")
    if "target" not in predictions.columns:
        raise ValueError("Predictions dataframe must include 'target'.")
    if "probability" not in predictions.columns:
        raise ValueError("Predictions dataframe must include 'probability'.")

    method_names = tuple(methods)
    records: list[dict[str, object]] = []
    for group_value, group_frame in predictions.groupby(group_key, sort=True, dropna=False):
        targets = group_frame["target"].astype(int).unique()
        if len(targets) != 1:
            raise ValueError(f"Group '{group_value}' contains multiple target labels: {targets.tolist()}")

        record: dict[str, object] = {
            group_key: str(group_value),
            "target": int(targets[0]),
            "sample_count": int(len(group_frame)),
            "sample_ids": "|".join(group_frame["sample_id"].astype(str).tolist()),
        }
        if "relative_path" in group_frame.columns:
            record["relative_paths"] = "|".join(group_frame["relative_path"].astype(str).tolist())

        for column in ("group_name", "class_name", "source", "source_code"):
            if column in group_frame.columns:
                record[column] = group_frame[column].iloc[0]

        probabilities = group_frame["probability"].to_numpy(dtype=np.float64)
        for method in method_names:
            record[f"probability_{method}"] = aggregate_probability(probabilities, method=method, top_k=top_k)

        records.append(record)

    return pd.DataFrame(records).sort_values([group_key]).reset_index(drop=True)
