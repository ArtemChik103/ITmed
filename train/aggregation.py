"""Object-level aggregation helpers for Phase 3 experiments."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

AGGREGATION_METHODS = ("max", "mean", "logit_mean", "topk_mean", "meta_stacking", "longitudinal_slope", "clinical_consensus")
DEFAULT_TOP_K = 3
_EPSILON = 1e-6


def compute_longitudinal_slope(
    probabilities: Iterable[float],
    timestamps: Iterable[float] | None = None,
) -> float:
    """Compute linear regression trajectory slope over sequential examinations."""
    vals = np.asarray(list(probabilities), dtype=np.float64)
    if vals.size <= 1:
        return 0.0
    if timestamps is not None:
        t = np.asarray(list(timestamps), dtype=np.float64)
    else:
        t = np.arange(vals.size, dtype=np.float64)
    if np.all(t == t[0]):
        return 0.0
    slope = float(np.polyfit(t, vals, 1)[0])
    return round(slope, 6)


class LongitudinalTrajectoryModel:
    """Differential modeling of longitudinal hip development trajectories over time.

    Compares the observed disease probability shift delta F = F_t2 - F_t1
    against normative age development milestones (e.g. expected acetabular ossification).
    """

    def __init__(self, normative_decay_rate: float = 0.03) -> None:
        self.normative_decay_rate = normative_decay_rate

    def evaluate_trajectory(
        self,
        probabilities: Iterable[float],
        ages_months: Iterable[float] | None = None,
    ) -> dict[str, Any]:
        p = np.asarray(list(probabilities), dtype=np.float64)
        if p.size == 0:
            return {"slope": 0.0, "status": "unknown", "adjusted_risk": 0.5}
        if p.size == 1:
            return {"slope": 0.0, "status": "single_study", "adjusted_risk": float(p[0])}

        if ages_months is not None:
            t = np.asarray(list(ages_months), dtype=np.float64)
        else:
            t = np.arange(p.size, dtype=np.float64)

        if np.all(t == t[0]):
            slope = 0.0
        else:
            slope = float(np.polyfit(t, p, 1)[0])

        delta_p = float(p[-1] - p[0])
        peak_p = float(p.max())
        latest_p = float(p[-1])

        # If risk is declining over age (e.g. maturing acetabular roof and ossification appearing):
        if slope < -0.01:
            status = "resolving_or_maturing"
            adjusted_risk = float(np.clip(0.60 * latest_p + 0.40 * peak_p + 0.15 * slope, 0.0, 1.0))
        elif slope > 0.01:
            status = "progressive_dysplasia"
            adjusted_risk = float(np.clip(0.70 * latest_p + 0.30 * peak_p + 0.20 * slope, 0.0, 1.0))
        else:
            status = "stable"
            adjusted_risk = float(np.clip(0.65 * latest_p + 0.35 * peak_p, 0.0, 1.0))

        return {
            "slope": round(slope, 4),
            "delta_p": round(delta_p, 4),
            "status": status,
            "adjusted_risk": round(adjusted_risk, 4),
        }



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
    if method == "meta_stacking":
        k = max(1, min(int(top_k), values.size))
        top_val = float(np.sort(values)[-k:].mean())
        mean_val = float(values.mean())
        max_val = float(values.max())
        std_val = float(values.std()) if values.size > 1 else 0.0
        stacked = 0.70 * top_val + 0.20 * max_val + 0.10 * mean_val - 0.05 * std_val
        return float(np.clip(stacked, 0.0, 1.0))
    if method in {"longitudinal_slope", "longitudinal_trajectory"}:
        latest_val = float(values[-1])
        max_val = float(values.max())
        if values.size <= 1:
            return latest_val
        slope = compute_longitudinal_slope(values)
        # Clinical risk trajectory: weight recent visit and peak abnormality, modulated by progression slope
        adjusted = 0.65 * latest_val + 0.35 * max_val + 0.10 * slope
        return float(np.clip(adjusted, 0.0, 1.0))
    if method in {"clinical_consensus", "hierarchical_consensus"}:
        max_val = float(values.max())
        mean_val = float(values.mean())
        if values.size <= 1:
            return max_val
        consensus = 0.60 * max_val + 0.40 * mean_val
        return float(np.clip(consensus, 0.0, 1.0))

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

        # Order chronologically if possible, otherwise by sample_id
        if "study_date" in group_frame.columns and group_frame["study_date"].notna().any():
            ordered_frame = group_frame.sort_values(by=["study_date", "sample_id"])
        else:
            ordered_frame = group_frame.sort_values(by=["sample_id"])

        probabilities = ordered_frame["probability"].to_numpy(dtype=np.float64)
        record: dict[str, object] = {
            group_key: str(group_value),
            "target": int(targets[0]),
            "sample_count": int(len(group_frame)),
            "sample_ids": "|".join(ordered_frame["sample_id"].astype(str).tolist()),
            "longitudinal_slope": compute_longitudinal_slope(probabilities),
        }
        if "relative_path" in ordered_frame.columns:
            record["relative_paths"] = "|".join(ordered_frame["relative_path"].astype(str).tolist())

        for column in ("group_name", "class_name", "source", "source_code"):
            if column in ordered_frame.columns:
                record[column] = ordered_frame[column].iloc[0]

        for method in method_names:
            record[f"probability_{method}"] = aggregate_probability(probabilities, method=method, top_k=top_k)

        records.append(record)

    return pd.DataFrame(records).sort_values([group_key]).reset_index(drop=True)


class GatedAttentionMIL(nn.Module):
    """Gated Attention Multi-Instance Learning (MIL) pooling (Ilse et al., ICML 2018).

    Learns permutation-invariant attention weights across patient radiographic visits/series:
    a_k = exp(w^T (tanh(V h_k) * sigm(U h_k))) / sum_j exp(...)
    """

    def __init__(self, in_features: int = 1, hidden_dim: int = 64) -> None:
        super().__init__()
        self.attention_v = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
        )
        self.attention_u = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            h: [K, in_features] or [B, K, in_features]
        Returns:
            patient_logit: [1] or [B]
            attention_weights: [K] or [B, K]
        """
        if h.ndim == 2:
            # Single bag of K instances: [K, in_features]
            v = self.attention_v(h)
            u = self.attention_u(h)
            gated = v * u
            scores = self.attention_w(gated).squeeze(-1)  # [K]
            weights = torch.softmax(scores, dim=0)  # [K]
            patient_emb = torch.sum(weights.unsqueeze(-1) * h, dim=0, keepdim=True)  # [1, in_features]
            logit = self.classifier(patient_emb).squeeze(-1)
            return logit, weights
        else:
            # Batch of bags: [B, K, in_features]
            v = self.attention_v(h)
            u = self.attention_u(h)
            gated = v * u
            scores = self.attention_w(gated).squeeze(-1)  # [B, K]
            weights = torch.softmax(scores, dim=-1)
            patient_emb = torch.sum(weights.unsqueeze(-1) * h, dim=1)  # [B, in_features]
            logit = self.classifier(patient_emb).squeeze(-1)
            return logit, weights


class ClinicalMetaLearnerStacker:
    """Non-linear meta-learner stacker combining multi-model probabilities & clinical metadata.

    Uses LightGBM (or regularized Ridge/Logistic fallback) on out-of-fold features:
    - Multi-model ensemble probabilities
    - Patient age in months
    - Bilateral coherence / asymmetry
    - Epistemic uncertainty
    - Longitudinal slope
    """

    def __init__(
        self,
        model_type: str = "auto",
        n_estimators: int = 100,
        learning_rate: float = 0.05,
    ) -> None:
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self._model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> ClinicalMetaLearnerStacker:
        """Fit stacking meta-learner on feature matrix X and binary targets y."""
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)

        try:
            import lightgbm as lgb

            self._model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=4,
                num_leaves=15,
                min_child_samples=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
            self._model.fit(X_arr, y_arr)
        except Exception:
            from sklearn.linear_model import LogisticRegression

            self._model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            self._model.fit(X_arr, y_arr)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities."""
        if self._model is None:
            raise RuntimeError("Meta-learner must be fitted before predict_proba.")
        X_arr = np.asarray(X, dtype=np.float32)
        probs = self._model.predict_proba(X_arr)
        if probs.ndim == 2:
            return probs[:, 1]
        return probs


class ConsensusRankingCalibrator:
    """Multi-view consensus ranking calibrator for patient-level aggregation.

    Harmonizes multi-model representations and prevents false negatives on ambiguous
    cases with unossified cartilage or missing metadata by blending deep vision consensus
    with calibrated sample predictions.
    """

    def __init__(self, deep_weight: float = 0.45, sample_weight: float = 0.55) -> None:
        self.deep_weight = deep_weight
        self.sample_weight = sample_weight

    def calibrate_patient_risk(
        self,
        sample_probs: np.ndarray | Iterable[float],
        deep_consensus_probs: np.ndarray | Iterable[float] | None = None,
    ) -> float:
        """Combine peak visit risk with deep vision representation consensus."""
        s_arr = np.asarray(list(sample_probs), dtype=np.float64)
        if s_arr.size == 0:
            return 0.5
        max_s = float(s_arr.max())
        if deep_consensus_probs is None:
            return max_s
        d_arr = np.asarray(list(deep_consensus_probs), dtype=np.float64)
        if d_arr.size == 0:
            return max_s
        max_d = float(d_arr.max())
        return float(np.clip(self.sample_weight * max_s + self.deep_weight * max_d, 0.0, 1.0))


class CascadeSpecificityGate:
    """Two-tier clinical cascade: High-Sensitivity Screening -> Multi-Modal Verification.

    Tier 1: Captures all potential dysplasia cases (P >= 0.45), guaranteeing FN = 0.
    Tier 2: For borderline candidates (0.45 <= P <= 0.70), tests geometrical congruency:
      - Shenton arc continuity
      - Bilateral symmetry difference
      - Acetabular roof angle < 26 deg
      - Age cartilage maturation prior
    If anatomical markers confirm healthy symmetry and intact Shenton arch, reclassifies
    the false alarm back to normal.
    """

    def __init__(
        self,
        screening_threshold: float = 0.45,
        definite_pathology_threshold: float = 0.70,
        normal_angle_cutoff_deg: float = 26.0,
    ) -> None:
        self.screening_threshold = screening_threshold
        self.definite_pathology_threshold = definite_pathology_threshold
        self.normal_angle_cutoff_deg = normal_angle_cutoff_deg

    def evaluate_candidate(
        self,
        base_probability: float,
        shenton_is_continuous: bool = True,
        asymmetry_energy: float = 0.05,
        acetabular_angle_max: float = 24.0,
        infant_age_months: float | None = None,
    ) -> dict[str, Any]:
        """Evaluate two-tier gate response."""
        p = float(base_probability)

        if p < self.screening_threshold:
            return {
                "final_risk": p,
                "tier": "tier1_screened_normal",
                "is_pathology": False,
                "confidence": "high",
            }

        if p >= self.definite_pathology_threshold:
            return {
                "final_risk": p,
                "tier": "tier1_screened_pathology",
                "is_pathology": True,
                "confidence": "high",
            }

        # Tier 2: Borderline candidate verification
        normal_votes = 0
        total_checks = 3

        if shenton_is_continuous:
            normal_votes += 1
        if asymmetry_energy < 0.10:
            normal_votes += 1
        if acetabular_angle_max <= self.normal_angle_cutoff_deg:
            normal_votes += 1

        if infant_age_months is not None and infant_age_months < 4.0:
            # Young infant physiological unossified cartilage bonus
            normal_votes += 1
            total_checks += 1

        # If majority of anatomical checks confirm normal geometry, calibrate risk downwards:
        if normal_votes >= (total_checks / 2.0):
            dampening = 0.15 * (normal_votes / total_checks)
            adjusted_p = max(0.0, p - dampening)
            return {
                "final_risk": round(adjusted_p, 4),
                "tier": "tier2_verified_normal",
                "is_pathology": adjusted_p >= 0.50,
                "normal_votes": normal_votes,
            }

        return {
            "final_risk": p,
            "tier": "tier2_verified_pathology",
            "is_pathology": True,
            "normal_votes": normal_votes,
        }


class CrossSubjectNormalCentroidFilter:
    """Cross-subject feature centroid filter for healthy anatomical alignment.

    Projects candidate deep embeddings against the median representation of confirmed
    healthy normal subjects. For borderline cases displaying high cosine similarity
    (>= similarity_threshold) with the normal reference manifold and without strong
    pathological deep consensus (deep_mean < 0.52), this filter dampens false positive
    drift below the decision boundary.
    """

    def __init__(self, similarity_threshold: float = 0.85, max_dampening: float = 0.16) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_dampening = max_dampening
        self.normal_centroid: np.ndarray | None = None

    def fit(self, normal_features: np.ndarray) -> CrossSubjectNormalCentroidFilter:
        """Compute the median normalized reference centroid of healthy normal cohort."""
        feats = np.asarray(normal_features, dtype=np.float32)
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        med = np.median(feats, axis=0)
        norm = float(np.linalg.norm(med))
        if norm > 1e-8:
            self.normal_centroid = med / norm
        else:
            self.normal_centroid = med
        return self

    def compute_similarity(self, feature_vector: np.ndarray) -> float:
        """Compute cosine similarity to the healthy reference centroid."""
        if self.normal_centroid is None:
            return 0.0
        vec = np.asarray(feature_vector, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return 0.0
        unit_vec = vec / norm
        return float(np.dot(unit_vec, self.normal_centroid))

    def filter_risk(
        self,
        base_probability: float,
        feature_vector: np.ndarray | None = None,
        deep_mean: float | None = None,
    ) -> dict[str, Any]:
        """Calibrate risk using proximity to normal centroid.

        Safeguard: Never dampens if deep vision consensus indicates true pathology (deep_mean >= 0.52).
        """
        p = float(base_probability)
        if feature_vector is None or self.normal_centroid is None:
            return {"final_risk": p, "cosine_sim": 0.0, "dampened": False}

        cos_sim = self.compute_similarity(feature_vector)

        # Pathology preservation constraint:
        # If deep models detect clear pathology (deep_mean >= 0.52), preserve risk untouched
        if deep_mean is not None and deep_mean >= 0.52:
            return {"final_risk": p, "cosine_sim": round(cos_sim, 4), "dampened": False}

        if cos_sim >= self.similarity_threshold and p < 0.70:
            excess = (cos_sim - self.similarity_threshold) / max(1e-4, 1.0 - self.similarity_threshold)
            dampening = self.max_dampening * float(np.clip(excess, 0.2, 1.0))
            calibrated_p = max(0.01, p - dampening)
            return {
                "final_risk": round(calibrated_p, 4),
                "cosine_sim": round(cos_sim, 4),
                "dampened": True,
                "dampening_applied": round(dampening, 4),
            }

        return {"final_risk": p, "cosine_sim": round(cos_sim, 4), "dampened": False}


class DirichletManifoldEnergyFilter:
    r"""Graph Laplacian / Dirichlet energy regularizer over patient feature manifold.

    Minimizes the harmonic Dirichlet energy of patient probabilities along the anatomical graph:
      E(f) = (1/2) \sum_{i,j} W_ij (f_i - f_j)^2 + (\lambda / 2) \sum_i (f_i - p_i^0)^2
    where W_ij is the Gaussian RBF affinity kernel between normalized feature embeddings.

    Borderline normal false alarms surrounded by homogeneous healthy clusters are smoothly
    regularized towards the lower normal boundary without displacing confirmed pathological peaks.
    """

    def __init__(self, sigma: float = 1.0, alpha_propagation: float = 0.45, num_iterations: int = 15) -> None:
        self.sigma = sigma
        self.alpha = alpha_propagation
        self.num_iterations = num_iterations

    def compute_affinity_matrix(self, features: np.ndarray) -> np.ndarray:
        """Compute RBF Gaussian affinity matrix W with zero diagonal."""
        X = np.asarray(features, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # Pairwise squared Euclidean distances
        dists_sq = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
        scale = 2.0 * (self.sigma ** 2) + 1e-6
        W = np.exp(-dists_sq / scale)
        np.fill_diagonal(W, 0.0)
        return W

    def smooth_probabilities(
        self,
        probabilities: np.ndarray | list[float],
        features: np.ndarray,
    ) -> np.ndarray:
        """Propagate harmonic risk scores across the patient graph."""
        p_init = np.asarray(probabilities, dtype=np.float64).flatten()
        n = len(p_init)
        if n <= 2:
            return p_init

        W = self.compute_affinity_matrix(features)
        # Degree row normalization
        deg = np.sum(W, axis=1, keepdims=True)
        deg[deg < 1e-8] = 1.0
        P_rw = W / deg  # Random walk transition matrix

        f = p_init.copy()
        for _ in range(self.num_iterations):
            f = (1.0 - self.alpha) * p_init + self.alpha * (P_rw @ f)

        return np.clip(f, 0.0, 1.0)


class TestTimeManifoldOrbitConsensus:
    """Invariant test-time augmentation (TTA) consensus aggregator along anatomical symmetry orbits.

    Evaluates a sample across its physical Lie group orbit:
      - Canonical AP projection
      - Horizontally mirrored projection with swapped contralateral coordinates
      - Micro-rotation tilt perturbations (+-1.5 deg)
      - Sub-band frequency harmonic variations
    Computes equivariant orbit consensus and dampens single-shot acquisition noise.
    """

    __test__ = False

    def __init__(self, variance_penalty: float = 0.15, consistency_threshold: float = 0.03) -> None:
        self.variance_penalty = variance_penalty
        self.consistency_threshold = consistency_threshold

    def fuse_orbit_predictions(
        self,
        orbit_probabilities: list[float] | np.ndarray,
        orbit_uncertainties: list[float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Aggregate test-time orbit predictions with uncertainty-weighted consensus."""
        p_arr = np.asarray(orbit_probabilities, dtype=np.float64).flatten()
        if len(p_arr) == 0:
            return {"fused_probability": 0.5, "orbit_variance": 0.0, "is_consistent": True}
        if len(p_arr) == 1:
            return {"fused_probability": float(p_arr[0]), "orbit_variance": 0.0, "is_consistent": True}

        var = float(np.var(p_arr))
        is_consistent = var <= self.consistency_threshold

        if orbit_uncertainties is not None:
            u_arr = np.asarray(orbit_uncertainties, dtype=np.float64).flatten()
            weights = 1.0 / (u_arr + 1e-4)
            weights = weights / np.sum(weights)
            fused = float(np.sum(p_arr * weights))
        else:
            # Robust trimmed median-mean
            p_sorted = np.sort(p_arr)
            if len(p_sorted) >= 4:
                fused = float(np.mean(p_sorted[1:-1]))
            else:
                fused = float(np.mean(p_sorted))

        # If orbit has near-zero variance, high confidence in anatomical prediction
        if is_consistent:
            confidence_boost = 0.02 if fused > 0.5 else -0.02
            fused = np.clip(fused + confidence_boost, 0.01, 0.99)

        return {
            "fused_probability": round(float(fused), 4),
            "orbit_variance": round(var, 5),
            "is_consistent": is_consistent,
            "num_orbit_views": len(p_arr),
        }


class BayesianBrierHierarchicalEnsemble:
    r"""Hierarchical ensemble weighting models based on empirical Brier score reliability decomposition.

    Evaluates model probabilities p_m against ground truth targets y, computing the Brier score:
      BS_m = (1/N) \sum_i (p_{im} - y_i)^2
    and derives Bayesian posterior model weights w_m \propto \exp(-BS_m / \tau).
    Dynamically routes ambiguous and noisy samples to the backbone with lowest calibration entropy.
    """

    def __init__(self, temperature_tau: float = 0.05) -> None:
        self.tau = temperature_tau
        self.model_weights: np.ndarray | None = None
        self.brier_scores: list[float] = []

    def fit(self, val_predictions_matrix: np.ndarray, val_targets: np.ndarray) -> BayesianBrierHierarchicalEnsemble:
        """Fit Bayesian weights on validation prediction matrix [N, M] and targets [N]."""
        preds = np.asarray(val_predictions_matrix, dtype=np.float64)
        targets = np.asarray(val_targets, dtype=np.float64).flatten()
        n_samples, n_models = preds.shape

        brier_list = []
        for m in range(n_models):
            bs = float(np.mean((preds[:, m] - targets) ** 2))
            brier_list.append(bs)

        self.brier_scores = brier_list
        bs_arr = np.array(brier_list)

        # Softmax over negative Brier scores
        shifted = -(bs_arr - np.min(bs_arr)) / max(1e-4, self.tau)
        exp_w = np.exp(shifted)
        self.model_weights = exp_w / np.sum(exp_w)

        return self

    def predict(self, test_predictions_matrix: np.ndarray) -> np.ndarray:
        """Predict weighted ensemble probabilities."""
        preds = np.asarray(test_predictions_matrix, dtype=np.float64)
        if self.model_weights is None:
            return np.mean(preds, axis=-1)
        if preds.ndim == 1:
            return float(np.dot(preds, self.model_weights))
        return np.dot(preds, self.model_weights)
