import torch


def compute_topk_token_overlap_ratios(student_logits, teacher_logits, topks=(4, 16, 64), **kwargs):
    """Compute top-k token overlap ratios between student and teacher logits."""
    overlap_ratios = {}
    with torch.no_grad():
        for topk in topks:
            k = min(topk, student_logits.shape[-1])
            student_topk = student_logits.topk(k=k, dim=-1).indices
            teacher_topk = teacher_logits.topk(k=k, dim=-1).indices
            token_overlap_ratio = (
                (student_topk.unsqueeze(-1) == teacher_topk.unsqueeze(-2))
                .any(dim=-1)
                .float()
                .sum(dim=-1)
                / k
            )
            overlap_ratios[f"token_overlap_ratio/top{topk}"] = token_overlap_ratio.mean()
    return overlap_ratios
