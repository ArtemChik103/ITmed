"""Utilities for extracting and encoding infant age for age-conditioned classification."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

AGE_MONTH_REGEX = re.compile(r"(\d+)\s*(?:мес|month|m\b)", re.IGNORECASE)
AGE_YEAR_REGEX = re.compile(r"(\d+)\s*(?:г|лет|year|y\b)", re.IGNORECASE)
DICOM_AGE_REGEX = re.compile(r"^(\d{1,3})([DWMY])$", re.IGNORECASE)


def extract_age_from_path(path_or_str: str | Path) -> float | None:
    """Extract age in months from directory or file path name."""
    normalized = str(path_or_str).replace("\\", "/")
    
    # Check for months pattern first
    match_m = AGE_MONTH_REGEX.search(normalized)
    if match_m:
        return float(match_m.group(1))
    
    # Check for years pattern
    match_y = AGE_YEAR_REGEX.search(normalized)
    if match_y:
        return float(match_y.group(1)) * 12.0
    
    return None


def extract_age_from_dicom_metadata(metadata: dict[str, Any]) -> float | None:
    """Extract age in months from DICOM metadata fields."""
    # 1. PatientAge tag (e.g., '004M', '001Y', '026W')
    raw_age = metadata.get("patient_age")
    if raw_age and isinstance(raw_age, str):
        match = DICOM_AGE_REGEX.match(raw_age.strip())
        if match:
            value = float(match.group(1))
            unit = match.group(2).upper()
            if unit == "D":
                return value / 30.0
            if unit == "W":
                return value / 4.33
            if unit == "M":
                return value
            if unit == "Y":
                return value * 12.0

    # 2. BirthDate and StudyDate difference
    bdate_str = metadata.get("patient_birth_date")
    sdate_str = metadata.get("study_date")
    if bdate_str and sdate_str and isinstance(bdate_str, str) and isinstance(sdate_str, str):
        try:
            bdate = datetime.strptime(bdate_str[:8], "%Y%m%d")
            sdate = datetime.strptime(sdate_str[:8], "%Y%m%d")
            diff_days = (sdate - bdate).days
            if diff_days >= 0:
                return float(diff_days / 30.4375)
        except Exception:
            pass

    return None


def resolve_patient_age(path: str | Path, metadata: dict[str, Any] | None = None) -> tuple[float, bool]:
    """Resolve patient age in months. Returns (age_months, is_known)."""
    # Try metadata first
    if metadata:
        age_meta = extract_age_from_dicom_metadata(metadata)
        if age_meta is not None:
            return float(np_clip_age(age_meta)), True

    # Try path
    age_path = extract_age_from_path(path)
    if age_path is not None:
        return float(np_clip_age(age_path)), True

    return 0.0, False


def np_clip_age(age: float) -> float:
    """Clip age to clinical pediatric range [0..36] months."""
    return max(0.0, min(float(age), 36.0))


class NormativeCartilageGrowthModel:
    """Biomechanical model of infant pelvic ossification and normal cartilage maturation.

    In infants under 4-6 months, the femoral head is predominantly unossified cartilage,
    and the acetabular roof has a physiologically higher cartilaginous index (up to 28-30 deg).
    This model computes:
    1. Expected ossification center probability: P_oss(age) = 1 / (1 + exp(-k * (age - tau_oss)))
    2. Dynamic specificity threshold adjustment: T_bio(age) = T_0 + delta * Sigmoid(...)
    3. Age-adjusted dysplasia prior: prevents false alarms on normal unossified cartilage.
    """

    def __init__(
        self,
        tau_ossification_months: float = 4.5,
        slope_k: float = 0.8,
        base_threshold: float = 0.54,
        max_threshold_boost: float = 0.08,
    ) -> None:
        self.tau_ossification = tau_ossification_months
        self.slope_k = slope_k
        self.base_threshold = base_threshold
        self.max_threshold_boost = max_threshold_boost

    def expected_ossification_probability(self, age_months: float) -> float:
        """Calculate physiological probability that femoral head ossification nucleus is visible."""
        import math

        z = self.slope_k * (float(age_months) - self.tau_ossification)
        return 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, z))))

    def compute_biological_decision_threshold(self, age_months: float, is_known: bool = True) -> float:
        """Compute age-adapted decision threshold protecting against false positives on young infants."""
        if not is_known:
            return self.base_threshold
        p_oss = self.expected_ossification_probability(age_months)
        boost = self.max_threshold_boost * (1.0 - p_oss)
        return float(self.base_threshold + boost)

    def modulate_risk_by_cartilage_maturity(
        self,
        raw_risk: float,
        age_months: float,
        is_known: bool = True,
        cartilage_channel_intensity: float | None = None,
    ) -> float:
        """Modulate disease probability accounting for physiological cartilage radiolucency."""
        if not is_known:
            return float(raw_risk)
        p_oss = self.expected_ossification_probability(age_months)
        if p_oss < 0.35 and 0.50 <= raw_risk <= 0.65:
            dampening = 0.05 * (1.0 - p_oss)
            return float(max(0.0, raw_risk - dampening))
        return float(raw_risk)


class TemporalMaturationKinematics:
    """Evaluates longitudinal velocity and acceleration of pediatric pelvic joint maturation.

    Normal infant development exhibits physiological improvement (decrease in acetabular angle
    at rate ~0.8 - 1.2 deg/month, dP/dt <= 0). Conversely, untreated developmental dysplasia
    displays progressive joint decompensation (lateralization, dP/dt >= 0, d alpha/dt >= 0).
    """

    def __init__(self, normal_velocity_threshold: float = -0.02) -> None:
        self.normal_velocity_threshold = normal_velocity_threshold

    def evaluate_kinematics(
        self,
        time_points_months: list[float],
        risk_or_angle_trajectory: list[float],
    ) -> dict[str, Any]:
        """Compute 1st derivative (velocity) and 2nd derivative (acceleration) across visits."""
        if len(time_points_months) < 2 or len(risk_or_angle_trajectory) < 2:
            base_val = float(risk_or_angle_trajectory[0]) if risk_or_angle_trajectory else 0.5
            return {
                "velocity": 0.0,
                "acceleration": 0.0,
                "is_maturing_normally": False,
                "is_single_visit": True,
                "calibrated_patient_risk": base_val,
            }

        # Sort by time
        sorted_pairs = sorted(zip(time_points_months, risk_or_angle_trajectory), key=lambda x: x[0])
        t_sorted = [float(p[0]) for p in sorted_pairs]
        y_sorted = [float(p[1]) for p in sorted_pairs]

        # Velocity: dy / dt
        dt = max(0.2, t_sorted[-1] - t_sorted[0])
        dy = y_sorted[-1] - y_sorted[0]
        velocity = dy / dt

        # Acceleration if >= 3 visits
        acceleration = 0.0
        if len(t_sorted) >= 3:
            dt1 = max(0.1, t_sorted[1] - t_sorted[0])
            v1 = (y_sorted[1] - y_sorted[0]) / dt1
            dt2 = max(0.1, t_sorted[-1] - t_sorted[1])
            v2 = (y_sorted[-1] - y_sorted[1]) / dt2
            acceleration = (v2 - v1) / max(0.2, t_sorted[-1] - t_sorted[0])

        is_maturing_normally = velocity <= self.normal_velocity_threshold
        latest_risk = y_sorted[-1]
        peak_risk = max(y_sorted)

        if is_maturing_normally:
            # Drop risk if trajectory shows confirmed physiological joint resolution
            calibrated_risk = max(0.01, latest_risk - 0.12 * abs(velocity))
        else:
            calibrated_risk = peak_risk

        return {
            "velocity": round(velocity, 4),
            "acceleration": round(acceleration, 4),
            "is_maturing_normally": is_maturing_normally,
            "is_single_visit": False,
            "num_visits": len(t_sorted),
            "calibrated_patient_risk": round(calibrated_risk, 4),
        }


class AgeConditionedFeatureGate:
    """Age-conditioned anatomical feature gating across 3 pediatric developmental cohorts.
    
    Pediatric hip anatomy changes drastically during infant development:
      - Cohort 0: Neonatal (0-3 months) - cartilaginous femoral head, physiological radiolucency.
      - Cohort 1: Early Infant (3-6 months) - nascent ossification center, index angle inflection.
      - Cohort 2: Late Infant (> 6 months) - mineralized femoral nucleus, Perkins quadrants, Shenton arch.
      
    This module computes physiological gating coefficients [w_cartilage, w_geometry, w_bone]
    dynamically tuning feature representation to match the infant's developmental stage.
    """

    COHORTS = ("neonatal_0_3m", "early_infant_3_6m", "late_infant_6m_plus")

    def __init__(self, tau_ossification: float = 4.5) -> None:
        self.tau = tau_ossification

    def get_cohort(self, age_months: float) -> str:
        """Determine developmental age cohort."""
        a = float(age_months)
        if a < 3.0:
            return "neonatal_0_3m"
        elif a <= 6.0:
            return "early_infant_3_6m"
        else:
            return "late_infant_6m_plus"

    def compute_cohort_weights(self, age_months: float) -> dict[str, float]:
        """Calculate continuous anatomical gating weights summing to 1.0."""
        a = float(max(0.0, age_months))
        import math

        # Cartilage weight decreases monotonically with age
        w_cartilage = 1.0 / (1.0 + math.exp(0.8 * (a - 3.0)))
        # Bone ossification increases monotonically with age
        w_bone = 1.0 / (1.0 + math.exp(-0.8 * (a - self.tau)))
        # Geometric structure peaks in the transition window (3-6 months)
        w_geom = math.exp(-0.5 * (((a - 4.5) / 2.0) ** 2))

        total = w_cartilage + w_geom + w_bone
        return {
            "w_cartilage": round(float(w_cartilage / total), 4),
            "w_geometry": round(float(w_geom / total), 4),
            "w_bone": round(float(w_bone / total), 4),
        }

    def gate_features(
        self,
        features: Any,
        age_months: float,
    ) -> dict[str, Any]:
        """Apply age-dependent soft gating on feature representations."""
        import numpy as np

        feat_arr = np.asarray(features, dtype=np.float32).copy()
        weights = self.compute_cohort_weights(age_months)
        cohort = self.get_cohort(age_months)

        dim = len(feat_arr)
        # Partition feature vector into 3 equal or proportional sectors
        d3 = dim // 3
        if d3 > 0:
            feat_arr[:d3] *= float(weights["w_cartilage"] * 3.0)
            feat_arr[d3 : 2 * d3] *= float(weights["w_geometry"] * 3.0)
            feat_arr[2 * d3 :] *= float(weights["w_bone"] * 3.0)

        return {
            "gated_features": feat_arr,
            "cohort": cohort,
            "weights": weights,
            "age_months": round(float(age_months), 2),
        }


class GrafUltrasoundPriorDistillation:
    """Virtual cross-modal knowledge distillation from Graf pediatric ultrasound classification.
    
    In infants < 6 months, Graf ultrasound is the international gold standard:
      - Alpha angle (bony roof angle): >= 60 deg is mature normal; < 50 deg is severe dysplasia.
      - Beta angle (cartilage roof angle): < 55 deg is normal; > 77 deg indicates everted labrum.
      - Graf Types:
        * Type I: alpha >= 60 deg (Mature normal)
        * Type IIa: 50 <= alpha < 60 deg, age < 3 months (Physiological maturation delay)
        * Type IIb: 50 <= alpha < 60 deg, age >= 3 months (Delayed ossification / mild dysplasia)
        * Type IIc/D: 43 <= alpha < 50 deg (Severe dysplasia / decentering)
        * Type III/IV: alpha < 43 deg (Eccentric subluxation or complete dislocation)
    """

    GRAF_DESCRIPTIONS = {
        "Type_I": "Mature normal hip joint",
        "Type_IIa": "Physiological delay in maturation (infant < 3 months)",
        "Type_IIb": "Definite dysplasia requiring abduction orthosis (infant >= 3 months)",
        "Type_IIc_D": "Critical / decentering hip joint",
        "Type_III_IV": "Dislocated hip joint requiring orthopedic reduction",
    }

    def estimate_graf_type(
        self,
        acetabular_angle_xray: float,
        age_months: float,
        is_dislocated: bool = False,
    ) -> dict[str, Any]:
        """Estimate equivalent Graf ultrasound alpha/beta angles and diagnostic type."""
        th = float(acetabular_angle_xray)
        age = float(max(0.0, age_months))

        # Physiological geometric conversion: alpha_graf ~ 90 - theta_xray
        est_alpha = float(np.clip(90.0 - th + (0.5 if age < 3.0 else 0.0), 30.0, 75.0))
        # Beta angle inversely correlates with alpha
        est_beta = float(np.clip(50.0 + (th - 22.0) * 1.2, 40.0, 85.0))

        if is_dislocated or est_alpha < 43.0:
            graf_type = "Type_III_IV"
            is_pathology = True
        elif est_alpha < 50.0:
            graf_type = "Type_IIc_D"
            is_pathology = True
        elif est_alpha < 60.0:
            if age < 3.0:
                graf_type = "Type_IIa"
                is_pathology = False  # Physiological, monitor
            else:
                graf_type = "Type_IIb"
                is_pathology = True
        else:
            graf_type = "Type_I"
            is_pathology = False

        return {
            "graf_type": graf_type,
            "description": self.GRAF_DESCRIPTIONS[graf_type],
            "estimated_alpha_deg": round(est_alpha, 1),
            "estimated_beta_deg": round(est_beta, 1),
            "is_ultrasound_pathology": is_pathology,
            "requires_ultrasound_confirmation": (graf_type in ("Type_IIa", "Type_IIb", "Type_IIc_D")),
        }


class PediatricHipPrognosticEngine:
    """Longitudinal prognostic engine evaluating conservative therapy failure risk & outcome trajectory.
    
    Predicts the risk of persistent residual dysplasia or Pavlik harness failure at 12-24 months
    based on baseline clinical parameters:
      - Patient age at diagnosis (success is >92% when treated < 4 months, drops > 6 months).
      - Baseline Reimers extrusion index.
      - Acetabular roof steepness.
      - Tönnis dislocation grade.
      
    Computes Harrell's C-index concordance and risk stratifications.
    """

    def __init__(self, high_risk_threshold: float = 0.60) -> None:
        self.high_risk_threshold = high_risk_threshold

    def predict_treatment_failure_risk(
        self,
        age_months: float,
        reimers_index_pct: float,
        acetabular_angle_deg: float,
        tonnis_grade: int = 0,
    ) -> dict[str, Any]:
        """Compute personalized failure probability score R in [0, 1]."""
        age = float(max(0.0, age_months))
        r_pct = float(max(0.0, reimers_index_pct))
        ang = float(max(10.0, acetabular_angle_deg))
        tg = max(0, min(4, int(tonnis_grade)))

        # Age penalty: treatment initiated after 6 months increases failure odds
        age_factor = 0.05 * max(0.0, age - 4.0)

        # Reimers migration factor: migration above 33% increases risk
        reimers_factor = 0.015 * max(0.0, r_pct - 30.0)

        # Acetabular slope factor: angles above 28 deg add risk
        angle_factor = 0.02 * max(0.0, ang - 26.0)

        # Tönnis grade baseline risk
        tonnis_base = [0.02, 0.08, 0.35, 0.65, 0.85][tg]

        raw_score = tonnis_base + age_factor + reimers_factor + angle_factor
        prob_failure = float(np.clip(raw_score, 0.01, 0.98))

        if prob_failure < 0.25:
            trajectory = "spontaneous_resolution_or_conservative_success"
            action = "Routine Pavlik harness or observation with 6-week ultrasound control."
        elif prob_failure < self.high_risk_threshold:
            trajectory = "abduction_orthosis_monitoring_required"
            action = "Rigid abduction splint (Tubingen/Frejka) with bi-weekly clinical monitoring."
        else:
            trajectory = "high_risk_surgical_reduction_probable"
            action = "Pediatric orthopedic consultation for traction / closed or open surgical reduction."

        return {
            "treatment_failure_risk": round(prob_failure, 3),
            "prognostic_trajectory": trajectory,
            "action_recommendation": action,
            "success_probability": round(1.0 - prob_failure, 3),
            "is_high_risk": prob_failure >= self.high_risk_threshold,
        }

    @staticmethod
    def compute_c_index(risk_scores: np.ndarray | list[float], event_occurred: np.ndarray | list[int]) -> float:
        """Compute Harrell's Concordance Index (C-Index)."""
        r = np.asarray(risk_scores, dtype=np.float64).flatten()
        e = np.asarray(event_occurred, dtype=np.int64).flatten()
        n = len(r)
        if n < 2:
            return 1.0

        concordant = 0
        total_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                if e[i] != e[j]:
                    total_pairs += 1
                    # Pair where one failed (e=1) and other succeeded (e=0)
                    if (e[i] == 1 and r[i] > r[j]) or (e[j] == 1 and r[j] > r[i]):
                        concordant += 1
                    elif r[i] == r[j]:
                        concordant += 0.5

        if total_pairs == 0:
            return 1.0
        return float(round(concordant / total_pairs, 4))




