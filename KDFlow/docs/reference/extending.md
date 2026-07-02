# Extending KDFlow

KDFlow exposes two simple registries:

- `kdflow.algorithms.register_algorithm(name)` populates `ALGO_DICT`.
- `kdflow.loss.register_loss(name)` populates `LOSS_DICT`.

Both `kdflow/algorithms/__init__.py` and `kdflow/loss/__init__.py`
**auto-import every sibling Python module**, so dropping a new file under the
right directory is enough — no further wiring needed.

The chosen algorithm and loss are picked at runtime from the CLI:

```bash
--kd_algorithm <name>   # must exist in ALGO_DICT
--kd_loss_fn   <name>   # must exist in LOSS_DICT
```

---

## Adding a custom KD algorithm

Create a file under `kdflow/algorithms/`, e.g. `my_custom_kd.py`:

```python
import torch
from kdflow.loss import LOSS_DICT
from kdflow.algorithms import register_algorithm


@register_algorithm("my_custom_kd")
class MyCustomKD:
    def __init__(self, strategy, student_model, teacher_lm_head, **kwargs):
        self.strategy        = strategy
        self.student         = student_model
        self.teacher_lm_head = teacher_lm_head
        self.loss_fn         = LOSS_DICT[strategy.args.kd.loss_fn]

    def training_step(self, micro_batch):
        # 1. unpack a micro batch produced by KDFlow's data pipeline
        student_input_ids = micro_batch["stu_input_ids"]
        student_attn_mask = micro_batch["stu_attn_mask"]
        student_loss_mask = micro_batch["stu_loss_mask"].bool()
        teacher_hiddens   = micro_batch["teacher_hiddens"]
        avg_token_num     = micro_batch["avg_micro_batch_token_num"]

        # 2. student forward
        output = self.student(
            student_input_ids,
            attention_mask=student_attn_mask,
            return_output=True,
        )
        student_logits = output["logits"][student_loss_mask]

        # 3. teacher logits via teacher lm_head (kept on the student side)
        teacher_logits = self.teacher_lm_head(
            teacher_hiddens.to(self.teacher_lm_head.weight)
        )

        # 4. compute KD loss with a registered divergence
        kd_loss = self.loss_fn(student_logits, teacher_logits, temperature=1.0)
        kd_loss = kd_loss.sum() / avg_token_num

        return {"loss": kd_loss, "kd_loss": kd_loss}
```

Then run it with:

```bash
--kd_algorithm my_custom_kd --kd_loss_fn kl
```

### What the algorithm receives

The trainer calls `training_step(micro_batch)` once per gradient-accumulation
step. The dictionary `micro_batch` contains:

| Key                              | Description                                                         |
|----------------------------------|---------------------------------------------------------------------|
| `stu_input_ids`                  | Student-side `input_ids` (post-packing if enabled).                 |
| `stu_attn_mask`                  | Attention mask for the student.                                     |
| `stu_loss_mask`                  | Mask of *training tokens* (where the loss is computed).             |
| `teacher_hiddens`                | Teacher last-layer hidden states (already on the student GPUs).     |
| `avg_micro_batch_token_num`      | Avg number of training tokens — divide your loss by this for
                                     gradient-accumulation invariance.                                  |

For cross-tokenizer algorithms additional teacher-side tensors (e.g.
`teacher_input_ids`, `teacher_attn_mask`, alignment indices) are also present;
see `simple_ctkd.py` and `dskd.py` for working examples.

---

## Adding a custom KD loss

Create a file under `kdflow/loss/`, e.g. `my_custom_loss.py`:

```python
import torch
import torch.nn.functional as F

from kdflow.loss import register_loss


@register_loss("my_custom_loss")
@torch.compile()
def compute_my_loss(
    student_logits,
    teacher_logits,
    temperature=1.0,
    reduction="none",
    **kwargs,
):
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    log_probs    = torch.log_softmax(student_logits, -1, dtype=torch.float32)
    target_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
    kl_div = F.kl_div(log_probs, target_probs, reduction=reduction).sum(-1)
    return kl_div
```

Then run it with:

```bash
--kd_loss_fn my_custom_loss
```

### Loss signature

A loss function should accept at least:

- `student_logits` — per-token logits, shape `(N, V)`.
- `teacher_logits` — per-token logits, shape `(N, V)`.
- `temperature` — softmax temperature (`--kd_temperature`).
- `reduction` — typically `"none"`, the algorithm reduces the result.
- `**kwargs` — accept extra hyperparameters (e.g. `jsd_beta`, `skew_lambda`,
  `adaptive_alpha`, `hrl_topk`) so you can call any loss uniformly.

It should return a per-token loss tensor.

---

## Where to look for inspiration

- `kdflow/algorithms/vanilla_kd.py` — the simplest KD algorithm.
- `kdflow/algorithms/simple_ctkd.py` — cross-tokenizer with vocab masking.
- `kdflow/algorithms/dskd.py` — algorithm with extra learnable parameters
  (DSKD projectors).
- `kdflow/loss/kl_div.py`, `kdflow/loss/reverse_kl_div.py`,
  `kdflow/loss/js_div.py` — straightforward divergence implementations.
- `kdflow/loss/hierarchical_ranking_loss.py` — a non-divergence example
  using top-k logic.
