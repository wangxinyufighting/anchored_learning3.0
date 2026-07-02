import torch

from kdflow.loss import build_loss_fn
from kdflow.algorithms import register_algorithm
from kdflow.loss.chunked_loss import chunked_loss
from kdflow.loss.cross_entropy import compute_cross_entropy
from kdflow.metrics.topk_token_overlap import compute_topk_token_overlap_ratios
from kdflow.metrics.entropy import compute_entropy


@register_algorithm("anchored_learning")
class AnchoredLearning:
    """Anchored Learning for stable SFT via a moving distributional anchor.

    The fixed teacher should be an SFT reference model. At each training step,
    the loss builds a detached anchor by interpolating current student logits
    with reference logits, then distills the student toward that anchor.
    """

    def __init__(self, strategy, student_model, teacher_lm_head, **kwargs):
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher_lm_head = teacher_lm_head
        if self.args.kd.kd_loss_fn != "anchored_kl":
            raise ValueError("AnchoredLearning requires `--kd_loss_fn anchored_kl`.")
        self.loss_fn = build_loss_fn(self.args.kd.kd_loss_fn, self.args)
        self.metric_fns = [compute_topk_token_overlap_ratios]
        if self.args.scenario == "on_policy_kd":
            self.metric_fns.append(compute_entropy)

        if isinstance(self.teacher_lm_head, dict):
            raise ValueError("AnchoredLearning currently expects one fixed SFT reference model, not multi-teacher KD.")

    def training_step(self, micro_batch):
        student_input_ids = micro_batch["stu_input_ids"]
        student_attn_mask = micro_batch["stu_attn_mask"]
        student_loss_mask = micro_batch["stu_loss_mask"].bool()
        teacher_hiddens = micro_batch.get("teacher_hiddens", None)
        avg_token_num = micro_batch["avg_micro_batch_token_num"]

        assert teacher_hiddens is not None, "micro_batch must contain `teacher_hiddens` for Anchored Learning"

        mm_kwargs = micro_batch.get("stu_multi_modal_inputs") or {}

        output = self.student(
            student_input_ids,
            attention_mask=student_attn_mask,
            allgather_logits=True,
            ring_attn_group=self.strategy.ring_attn_group,
            **mm_kwargs,
        )
        student_hiddens = output["hidden_states"][-1][student_loss_mask]
        del output

        chunk_size = self.args.train.chunked_loss_size or student_hiddens.shape[0]
        teacher_hiddens = teacher_hiddens.to(self.teacher_lm_head.weight)

        anchor_loss, metric_sums = chunked_loss(
            student_hiddens,
            self.student.model.lm_head,
            self.loss_fn,
            teacher_hidden=teacher_hiddens,
            teacher_head=self.teacher_lm_head,
            chunk_size=chunk_size,
            reduction="sum",
            metric_fns=self.metric_fns,
            return_metrics=True,
        )
        anchor_loss = anchor_loss / avg_token_num

        loss_info = {
            "loss": anchor_loss,
            "anchor_loss": anchor_loss,
        }
        loss_info.update({key: value / avg_token_num for key, value in metric_sums.items()})

        if self.args.kd.kd_ratio < 1:
            student_label_ids = student_input_ids.roll(shifts=-1, dims=1)[student_loss_mask]
            ce_loss = chunked_loss(
                student_hiddens,
                self.student.model.lm_head,
                compute_cross_entropy,
                label=student_label_ids,
                chunk_size=chunk_size,
                reduction="sum",
            ) / avg_token_num
            loss = (1 - self.args.kd.kd_ratio) * ce_loss + self.args.kd.kd_ratio * anchor_loss
            loss_info["loss"] = loss
            loss_info["ce_loss"] = ce_loss

        return loss_info
