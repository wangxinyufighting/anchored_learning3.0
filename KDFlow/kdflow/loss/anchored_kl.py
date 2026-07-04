import torch

from kdflow.loss import register_loss


@register_loss("anchored_kl")
def compute_anchored_kl_div(
    student_logits,
    teacher_logits,
    temperature=1.0,
    anchor_alpha=0.5,
    anchor_interpolation="logit",
    reduction="none",
    **kwargs,
):
    """KL(q_anchor || p_student) for Anchored Learning.

    The anchor is detached from the graph and is built from the current student
    distribution and a fixed SFT/reference distribution.
    """
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"Unsupported reduction: {reduction}")

    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    current_logits = student_logits.detach()
    reference_logits = teacher_logits.detach()

    student_log_probs = torch.log_softmax(student_logits, dim=-1, dtype=torch.float32)

    if anchor_interpolation == "logit":
        anchor_logits = (1.0 - anchor_alpha) * current_logits + anchor_alpha * reference_logits
        # Optimized: compute softmax directly instead of log_softmax().exp()
        anchor_probs = torch.softmax(anchor_logits, dim=-1, dtype=torch.float32)
        anchor_log_probs = torch.log(anchor_probs.clamp_min(1e-20))
    elif anchor_interpolation in ("prob", "probability"):
        current_probs = torch.softmax(current_logits, dim=-1, dtype=torch.float32)
        reference_probs = torch.softmax(reference_logits, dim=-1, dtype=torch.float32)
        anchor_probs = (1.0 - anchor_alpha) * current_probs + anchor_alpha * reference_probs
        anchor_log_probs = torch.log(anchor_probs.clamp_min(1e-20))
    else:
        raise ValueError(
            "anchor_interpolation must be one of {'logit', 'prob', 'probability'}, "
            f"got {anchor_interpolation!r}."
        )

    loss = (anchor_probs * (anchor_log_probs - student_log_probs)).sum(dim=-1)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
