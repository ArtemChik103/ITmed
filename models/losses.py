"""Loss functions for Phase 3 classifier training."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss on top of logits with optional label smoothing."""

    def __init__(
        self,
        *,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(dtype=logits.dtype)
        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probabilities, 1.0 - probabilities)
        alpha_t = torch.where(targets > 0.5, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * torch.pow(1.0 - pt, self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def build_loss(
    name: str,
    *,
    pos_weight: torch.Tensor | None = None,
    alpha: float = 0.75,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Factory for supported binary classification losses."""
    normalized_name = name.lower()
    if normalized_name == "bce":
        if label_smoothing > 0.0:
            class BCEWithSmoothing(nn.Module):
                def __init__(self, pos_w: torch.Tensor | None, ls: float) -> None:
                    super().__init__()
                    self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
                    self.ls = ls

                def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                    targets = targets.to(dtype=logits.dtype)
                    smoothed = targets * (1.0 - self.ls) + 0.5 * self.ls
                    return self.bce(logits, smoothed)

            return BCEWithSmoothing(pos_weight, label_smoothing)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if normalized_name == "focal":
        return FocalLoss(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)
    raise ValueError(f"Unsupported loss '{name}'. Expected 'bce' or 'focal'.")


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss for clustering embeddings by patient/class."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute SupCon loss across batch features and labels."""
        device = features.device
        if len(features.shape) < 2:
            raise ValueError("`features` needs to be [bsz, n_features] or [bsz, n_views, n_features]")
        if len(features.shape) == 2:
            features = features.unsqueeze(1)

        batch_size = features.shape[0]
        n_views = features.shape[1]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = F.normalize(contrast_feature, dim=1)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature,
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        loss = -mean_log_prob_pos.mean()
        return loss


class OHEMLoss(nn.Module):
    """Online Hard Example Mining (OHEM) wrapper.

    Focuses backpropagation on the top fraction of most ambiguous / misclassified samples.
    """

    def __init__(self, base_loss: nn.Module, keep_ratio: float = 0.70) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.keep_ratio = float(keep_ratio)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute element-wise loss
        if hasattr(self.base_loss, "reduction"):
            orig_reduction = self.base_loss.reduction
            self.base_loss.reduction = "none"
            per_sample_loss = self.base_loss(logits, targets)
            self.base_loss.reduction = orig_reduction
        else:
            per_sample_loss = F.binary_cross_entropy_with_logits(
                logits, targets.to(dtype=logits.dtype), reduction="none"
            )

        if per_sample_loss.ndim > 1:
            per_sample_loss = per_sample_loss.view(per_sample_loss.size(0), -1).mean(dim=1)

        batch_size = per_sample_loss.size(0)
        keep_num = max(1, int(round(self.keep_ratio * batch_size)))

        topk_losses, _ = torch.topk(per_sample_loss, k=keep_num, sorted=False)
        return topk_losses.mean()


class BarlowTwinsLoss(nn.Module):
    """Barlow Twins Self-Supervised Invariance & Cross-Correlation Loss.

    Forces cross-correlation between two views to be identity matrix:
    decorrelates representation dimensions and eliminates feature collapse without negatives.
    """

    def __init__(self, lambd: float = 0.0051) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # Normalize representations along batch dimension
        z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-6)
        z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-6)

        batch_size = z1.size(0)
        c = torch.mm(z1_norm.T, z2_norm) / float(batch_size)

        diag = torch.diagonal(c)
        on_diag = torch.sum(torch.pow(diag - 1.0, 2))

        # Off-diagonal elements
        off_diag = torch.sum(torch.pow(c.flatten()[:-1].view(c.size(0) - 1, c.size(0) + 1)[:, 1:], 2))
        return on_diag + self.lambd * off_diag


class SimCLRLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) Self-Supervised Loss."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        representations = torch.cat([z1, z2], dim=0)
        batch_size = z1.size(0)

        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)

        mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=z1.device)).float()
        denominator = mask * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        return torch.mean(all_losses)


class DifferentiableGeometryConsistencyLoss(nn.Module):
    """Enforces consistency between deep visual logits and pediatric orthopedic geometry.

    Penalizes model predictions whenever visual probability disagrees with the
    geometric acetabular index (alpha angle) and Shenton line continuity.
    """

    def __init__(self, angle_threshold_deg: float = 30.0, weight: float = 0.2) -> None:
        super().__init__()
        self.angle_threshold_deg = float(angle_threshold_deg)
        self.weight = float(weight)

    def forward(
        self,
        logits: torch.Tensor,
        keypoints: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            logits: [B] or [B, 1] classifier logits
            keypoints: [B, 8, 2] normalized coordinates in [0, 1]
        """
        if keypoints.ndim != 3 or keypoints.size(1) < 4:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Hilgenreiner vector connecting triradiate cartilages (0 -> 1)
        v_hilg = keypoints[:, 1] - keypoints[:, 0]  # [B, 2]
        norm_hilg = torch.norm(v_hilg, dim=1, keepdim=True) + 1e-6

        # Roof vectors
        v_roof_l = keypoints[:, 2] - keypoints[:, 0]
        v_roof_r = keypoints[:, 3] - keypoints[:, 1]
        norm_roof_l = torch.norm(v_roof_l, dim=1, keepdim=True) + 1e-6
        norm_roof_r = torch.norm(v_roof_r, dim=1, keepdim=True) + 1e-6

        cos_l = torch.sum(v_hilg * v_roof_l, dim=1) / (norm_hilg.squeeze(1) * norm_roof_l.squeeze(1))
        cos_r = torch.sum(-v_hilg * v_roof_r, dim=1) / (norm_hilg.squeeze(1) * norm_roof_r.squeeze(1))
        cos_l = torch.clamp(cos_l, -0.9999, 0.9999)
        cos_r = torch.clamp(cos_r, -0.9999, 0.9999)

        angle_l_rad = torch.acos(cos_l)
        angle_r_rad = torch.acos(cos_r)
        rad_to_deg = 180.0 / 3.141592653589793
        angle_l_deg = angle_l_rad * rad_to_deg
        angle_r_deg = angle_r_rad * rad_to_deg

        max_angle = torch.maximum(angle_l_deg, angle_r_deg)
        # Continuous geometric risk proxy: sigmoid over angle margin
        geometric_risk = torch.sigmoid((max_angle - self.angle_threshold_deg) / 3.0)

        visual_prob = torch.sigmoid(logits.view(-1))
        consistency_loss = F.binary_cross_entropy(visual_prob, geometric_risk.detach())
        return self.weight * consistency_loss


class MaskedPatchReconstructionLoss(nn.Module):
    """Loss for Masked Image Modeling (MIM / Masked Autoencoder).

    Combines MSE reconstruction loss on masked patches with cosine similarity.
    """

    def __init__(self, cosine_weight: float = 0.2) -> None:
        super().__init__()
        self.cosine_weight = float(cosine_weight)

    def forward(
        self,
        predicted_patches: torch.Tensor,
        target_patches: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
            predicted_patches: [B, N, D]
            target_patches: [B, N, D]
            mask: [B, N] boolean or float mask (1 for masked, 0 for unmasked)
        """
        if mask is not None:
            mask_bool = mask.bool()
            pred = predicted_patches[mask_bool]
            targ = target_patches[mask_bool]
        else:
            pred = predicted_patches
            targ = target_patches

        if pred.numel() == 0:
            return torch.tensor(0.0, device=predicted_patches.device, requires_grad=True)

        mse = F.mse_loss(pred, targ)
        cos_sim = F.cosine_similarity(pred, targ, dim=-1).mean()
        return mse + self.cosine_weight * (1.0 - cos_sim)


class FoundationModelDistillationLoss(nn.Module):
    """Knowledge Distillation Loss from Foundation Models (BioMedCLIP / DINOv2).

    Aligns intermediate clinical backbone features with foundation representations:
    L = alpha * (1 - cosine_similarity(f_student, f_teacher)) + (1 - alpha) * KL(p_student || p_teacher).
    """

    def __init__(self, temperature: float = 3.0, feature_weight: float = 0.5) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.feature_weight = float(feature_weight)

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        student_logits: torch.Tensor | None = None,
        teacher_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. Feature cosine alignment
        f_s = F.normalize(student_features, dim=-1)
        f_t = F.normalize(teacher_features, dim=-1)
        cos_loss = (1.0 - F.cosine_similarity(f_s, f_t, dim=-1)).mean()

        if student_logits is not None and teacher_logits is not None:
            # 2. Soft logit matching via KL divergence
            p_s = F.log_softmax(student_logits / self.temperature, dim=-1)
            p_t = F.softmax(teacher_logits / self.temperature, dim=-1)
            kl_loss = F.kl_div(p_s, p_t, reduction="batchmean") * (self.temperature**2)
            return self.feature_weight * cos_loss + (1.0 - self.feature_weight) * kl_loss

        return cos_loss


class BilateralArcFaceLoss(nn.Module):
    """Bilateral Additive Angular Margin Loss (ArcFace) on hip pair representations.

    Forces symmetric healthy hips into a tight hyperspherical cone (cos(theta) ~ 1),
    and enforces an angular margin m when dysplasia introduces asymmetry.
    """

    def __init__(self, scale: float = 16.0, margin: float = 0.25) -> None:
        super().__init__()
        self.scale = float(scale)
        self.margin = float(margin)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor, is_dysplastic: torch.Tensor) -> torch.Tensor:
        """Args:
            f_left: [B, D] left hip embedding
            f_right: [B, D] horizontally-flipped right hip embedding
            is_dysplastic: [B] binary label (1 if either hip is dysplastic, 0 if healthy symmetric)
        """
        f_l = F.normalize(f_left, dim=-1)
        f_r = F.normalize(f_right, dim=-1)
        # Cosine similarity between bilateral hips
        cos_theta = (f_l * f_r).sum(dim=-1)
        cos_theta = torch.clamp(cos_theta, -0.9999, 0.9999)

        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        target_cos = torch.where(is_dysplastic.bool(), cos_theta_m, cos_theta)
        loss = -torch.log(torch.sigmoid(self.scale * target_cos) + 1e-6).mean()
        return loss


class ClinicalConceptAlignmentLoss(nn.Module):
    """Aligns predicted visual features with explicit clinical symptoms.

    Enforces that:
    1. Healthy controls (y=0) have all pathological symptoms suppressed towards 0.
    2. Dysplastic cases (y=1) show strong activation in at least one or more primary signs
       (Shenton disruption, steep acetabular roof, or Perkins displacement).
    3. Prevents mode collapse across clinical concept tokens.
    """

    def __init__(self, suppression_weight: float = 0.5, coverage_weight: float = 0.5) -> None:
        super().__init__()
        self.suppression_weight = float(suppression_weight)
        self.coverage_weight = float(coverage_weight)

    def forward(
        self,
        symptom_logits: torch.Tensor,
        dysplasia_targets: torch.Tensor,
        dysplasia_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
            symptom_logits: [B, num_symptoms]
            dysplasia_targets: [B] binary {0, 1}
            dysplasia_logits: [B] (optional main task logit)
        """
        targets = dysplasia_targets.float()
        s_probs = torch.sigmoid(symptom_logits)  # [B, num_symptoms]

        # 1. Healthy suppression loss: healthy patients should not exhibit dysplastic signs
        healthy_mask = (targets == 0).unsqueeze(-1)  # [B, 1]
        loss_suppress = torch.mean(s_probs * healthy_mask)

        # 2. Dysplastic coverage loss: pathology must trigger at least one prominent sign
        path_mask = targets == 1
        if path_mask.any():
            max_symptom_prob = torch.max(s_probs[path_mask], dim=-1)[0]
            loss_coverage = torch.mean(1.0 - max_symptom_prob)
        else:
            loss_coverage = torch.tensor(0.0, device=symptom_logits.device)

        total_loss = self.suppression_weight * loss_suppress + self.coverage_weight * loss_coverage

        if dysplasia_logits is not None:
            bce_task = F.binary_cross_entropy_with_logits(dysplasia_logits, targets)
            total_loss = total_loss + bce_task

        return total_loss


class EvidentialDirichletLoss(nn.Module):
    """Evidential Deep Learning (EDL) loss quantifying epistemic uncertainty via Dirichlet concentration.

    Given evidence logits z in R^2:
      e = softplus(z)
      alpha = e + 1
      S = alpha_0 + alpha_1
      expected_p = alpha_1 / S
      uncertainty = 2 / S in (0, 1]
    The loss minimizes expected square error under the Dirichlet distribution plus a KL divergence
    regularizer penalizing misleading evidence.
    """

    def __init__(self, kl_weight: float = 0.1) -> None:
        super().__init__()
        self.kl_weight = kl_weight

    def forward(self, evidence_logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        """Args:
            evidence_logits: [B, 2] raw model outputs
            targets: [B] binary targets in {0, 1}
        """
        B = evidence_logits.shape[0]
        evidence = F.softplus(evidence_logits)  # [B, 2] >= 0
        alpha = evidence + 1.0  # Dirichlet concentration >= 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)  # [B, 1]

        p_hat = alpha / S  # [B, 2]
        uncertainty = 2.0 / S.squeeze(-1)  # [B]

        # One-hot targets
        y_one_hot = F.one_hot(targets.long(), num_classes=2).float()  # [B, 2]

        # Expected MSE loss: E_{p ~ Dir(alpha)} [ ||y - p||^2 ]
        err = (y_one_hot - p_hat) ** 2
        var = p_hat * (1.0 - p_hat) / (S + 1.0)
        loss_mse = torch.mean(torch.sum(err + var, dim=-1))

        # KL divergence regularizer with flat Dirichlet(1, 1) prior
        alpha_tilde = y_one_hot + (1.0 - y_one_hot) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=-1, keepdim=True)
        kl = (
            torch.lgamma(S_tilde)
            - torch.sum(torch.lgamma(alpha_tilde), dim=-1, keepdim=True)
            - torch.lgamma(torch.tensor(2.0, device=evidence_logits.device))
            + torch.sum((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=-1, keepdim=True)
        )
        loss_kl = torch.mean(kl)

        total_loss = loss_mse + self.kl_weight * loss_kl

        return {
            "total_loss": total_loss,
            "loss_mse": loss_mse,
            "loss_kl": loss_kl,
            "prob_pathology": p_hat[:, 1],
            "uncertainty": uncertainty,
        }
