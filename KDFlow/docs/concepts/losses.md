# KD Loss Functions

A KD algorithm is responsible for *producing* student logits and teacher
logits; the **divergence** between them is selected by `--kd_loss_fn`.
KDFlow registers ten losses in `kdflow/loss/`, all populated through
`@register_loss(name)`:

```python
from kdflow.loss import LOSS_DICT
print(list(LOSS_DICT))
# ['kl', 'rkl', 'jsd', 'akl', 'skl', 'srkl', 'tvd', 'hrl', 'top1_ce', 'anchored_kl']
```

| `--kd_loss_fn` | Name                          | File                                       | Notes                                                    |
|----------------|-------------------------------|--------------------------------------------|----------------------------------------------------------|
| `kl`           | KL divergence                 | `kdflow/loss/kl_div.py`                    | `torch.compile`d; the canonical KD loss.                 |
| `rkl`          | Reverse KL                    | `kdflow/loss/reverse_kl_div.py`            | Mode-seeking; popular on-policy choice.                  |
| `jsd`          | Jensen-Shannon divergence     | `kdflow/loss/js_div.py`                    | Symmetric; tunable mix via `--jsd_beta`.                 |
| `akl`          | Adaptive KL divergence        | `kdflow/loss/adaptive_kl_div.py`           | Mixes forward/reverse, controlled by `--adaptive_alpha`. |
| `skl`          | Skewed KL                     | `kdflow/loss/skewed_kl_div.py`             | Skewed forward KL, controlled by `--skew_lambda`.        |
| `srkl`         | Skewed reverse KL             | `kdflow/loss/skewed_rkl_div.py`            | Skewed reverse KL, controlled by `--skew_lambda`.        |
| `tvd`          | Total variation distance      | `kdflow/loss/tvd.py`                       | Bounded distance between distributions.                  |
| `hrl`          | Hierarchical Ranking Loss     | `kdflow/loss/hierarchical_ranking_loss.py` | Top-`--hrl_topk` ranking based loss.                     |
| `top1_ce`      | Top-1 cross-entropy           | `kdflow/loss/top1_ce.py`                   | CE against the teacher's argmax (cheap, hard-label KD).  |
| `anchored_kl`  | Anchored KL                   | `kdflow/loss/anchored_kl.py`               | Moving-anchor KL for `--kd_algorithm anchored_learning`. |

## Quick guide for picking a loss

- **`kl`** — the safe default for off-policy KD.
- **`rkl`** — strong choice for on-policy / self-distillation (mode-seeking).
- **`akl`** — when you want adaptive mixing of forward / reverse KL.
- **`jsd`** — symmetric, robust to teacher errors.
- **`skl` / `srkl`** — skewed variants when the teacher is much stronger.
- **`tvd`** — when you need a bounded distance.
- **`hrl`** — focus on relative top-k ranking rather than full distribution.
- **`top1_ce`** — cheapest, falls back to hard-label KD.
- **`anchored_kl`** — use with `anchored_learning` and a fixed SFT reference.

## Hyperparameters

A few losses have their own dials, settable on the CLI:

```bash
# Jensen-Shannon
--jsd_beta 0.5                 # mixture weight

# Skewed KL / RKL
--skew_lambda 0.1

# Adaptive KL
--adaptive_alpha 0.5

# Hierarchical Ranking Loss
--hrl_topk 5

# Anchored Learning
--anchor_alpha 0.5
--anchor_interpolation logit        # logit, prob, or probability
```

## Chunked Loss (Memory-Efficient Computation)

When distilling from large teacher models, materializing the full logits
tensor (shape `[num_tokens, vocab_size]`) can easily exhaust GPU memory —
especially with vocabularies of 100k+ tokens and long sequences.

KDFlow provides a **chunked loss** mechanism (`kdflow/loss/chunked_loss.py`)
that computes the loss in small token-level chunks without ever materializing
the full logits tensor. This is a **pure memory optimization** — the computed
loss is mathematically equivalent to the non-chunked version.

### How it works

```
student_hidden: [N, hidden_size]
                    │
                    ▼  (chunk_size tokens at a time)
            ┌───────────────┐
            │ student_head() │──► student_logits [chunk_size, vocab]
            └───────────────┘
            ┌───────────────┐
            │ teacher_head() │──► teacher_logits [chunk_size, vocab]
            └───────────────┘
                    │
                    ▼
              loss_fn(student_logits, teacher_logits)
                    │
                    ▼
            accumulate / concatenate
```

At each step, only one chunk of logits lives on GPU. The `lm_head` forward
pass is computed per-chunk, and the loss is accumulated (for `sum` / `mean`
reduction) or concatenated (for `none` reduction).

### Enabling chunked loss

Set `--chunked_loss_size <N>` in your training command:

```bash
# Process 2048 tokens at a time (recommended starting point)
--chunked_loss_size 2048
```

When `--chunked_loss_size` is `None` (default), chunked loss is disabled and
the full logits are materialized as before.

### Supported reductions

| `reduction` | Behavior                                                                 |
|-------------|--------------------------------------------------------------------------|
| `none`      | Returns per-token loss tensor `[N]` (chunks are concatenated).           |
| `sum`       | Returns scalar sum of all token losses.                                  |
| `mean`      | Returns scalar mean (sum divided by total token count).                  |

### When to use it

- **Large vocabularies** (e.g., 150k+ tokens) where logits consume significant memory.
- **Long sequences** where the total token count per micro-batch is high.
- **Limited GPU memory** — chunked loss trades compute time for memory savings.

### Performance notes

- Chunked loss adds a small overhead due to multiple `lm_head` forward passes
  instead of one batched call. In practice, the overhead is negligible compared
  to the main model forward/backward pass.
- The `lm_head` is marked as `skip=True` during the main student forward pass
  (avoiding redundant computation), and only invoked inside `chunked_loss`.

---

## Implementing your own loss

Add a file under `kdflow/loss/` and register it; see
[Extending KDFlow](../reference/extending.md). The `__init__.py` of
`kdflow.loss` auto-imports every sibling module, so your file is enough.

## See also

- [KD Algorithms](algorithms.md) — algorithms decide which logits to compare.
- [Arguments → Distillation Arguments](../reference/arguments.md#distillation-arguments) —
  the full list of distillation knobs.
