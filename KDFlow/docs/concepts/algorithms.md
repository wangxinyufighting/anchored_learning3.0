# KD Algorithms

KDFlow ships with five built-in algorithms registered via
`@register_algorithm(name)` in `kdflow/algorithms/`. Pick one with
`--kd_algorithm <name>`. Each algorithm decides **how** the student forward
and the teacher signal are combined into a loss; the **what** (the divergence)
is independently picked by `--kd_loss_fn` (see [KD Losses](losses.md)).

## Built-in algorithms

| `--kd_algorithm`  | Tokenizer        | Description                                                                                       | File                                |
|-------------------|------------------|---------------------------------------------------------------------------------------------------|-------------------------------------|
| `sft`             | n/a              | Pure cross-entropy training (used by `SFTTrainer`, no teacher).                                    | `kdflow/algorithms/sft.py`          |
| `vanilla_kd`      | **same**         | Standard same-tokenizer KD: student forward, teacher hidden states → teacher logits via lm_head, divergence loss. | `kdflow/algorithms/vanilla_kd.py`   |
| `anchored_learning` | **same**      | Anchored Learning: interpolate current student logits with a fixed SFT reference, then distill toward the moving anchor. | `kdflow/algorithms/anchored_learning.py` |
| `simple_ctkd`     | **different**    | Restrict the KD loss to the **overlapping vocabulary** between student and teacher tokenizers.    | `kdflow/algorithms/simple_ctkd.py`  |
| `dskd`            | **different**    | **Dual-Space KD** with learned projectors and `eta` / `cma` token alignment.                       | `kdflow/algorithms/dskd.py`         |

```python
from kdflow.algorithms import ALGO_DICT
print(list(ALGO_DICT))   # ['sft', 'vanilla_kd', 'anchored_learning', 'simple_ctkd', 'dskd']
```

## `vanilla_kd` — same-tokenizer KD

This is the default. The training step does:

1. Student forward on the batch's **student-side** input ids.
2. Multiply teacher hidden states (sent over shared memory) by the
   teacher `lm_head` to recover teacher logits.
3. Apply the chosen divergence (`--kd_loss_fn`) at temperature `--kd_temperature`.
4. Combine with token-level CE:
   \[
   \mathcal{L} = (1-\lambda)\,\mathcal{L}_{\text{CE}} + \lambda\,\mathcal{L}_{\text{KD}}, \quad
   \lambda = \texttt{kd\_ratio}.
   \]

Typical command:

```bash
--kd_algorithm vanilla_kd \
--kd_loss_fn   kl \
--kd_temperature 1.0 \
--kd_ratio 0.5
```

## `anchored_learning` — Anchored Learning

Anchored Learning implements stable SFT via explicit distributional control.
Use a fixed SFT model as `--teacher_name_or_path` and initialize the student
from the base model. At every training step, KDFlow builds

\[
q^{(t)} = I_\alpha(p_{\theta(t)}, p_{\text{sft}})
\]

either in logit space or probability space, detaches the anchor, and minimizes
`KL(q_anchor || p_student)`.

Typical command:

```bash
--kd_algorithm anchored_learning \
--kd_loss_fn anchored_kl \
--kd_ratio 1.0 \
--anchor_alpha 0.5 \
--anchor_interpolation logit
```

The paper uses logit-space interpolation, `alpha=0.5`, `K=5` inner epochs,
and `T=5` outer iterations. In KDFlow, each optimizer step constructs the
moving anchor from the current student and the fixed SFT reference; increase
`--num_epochs` to allocate a comparable optimization budget.

## `simple_ctkd` — Simple Cross-Tokenizer KD

For different-tokenizer pairs (e.g. Qwen3 → Llama3.2), `simple_ctkd` aligns
the two vocabularies once and computes the KD loss only on **overlapping
tokens**. It is fast, robust, and recommended as the **first thing to try**
for cross-tokenizer KD.

```bash
--kd_algorithm simple_ctkd \
--kd_loss_fn   kl \
--kd_ratio 0.5
```

## `dskd` — Dual-Space Knowledge Distillation

DSKD trains lightweight **projectors** that map student and teacher
representations into a shared space, optionally restricting alignment to the
top-k vocabulary tokens at initialisation.

Relevant flags:

```bash
--kd_algorithm dskd \
--dskd_token_align eta      # 'eta' (default) or 'cma'
--dskd_topk_vocab 1000      # -1 = all vocab tokens
--dskd_projector_lr 1e-4    # LR for the projectors
```

DSKD is a good fit when student and teacher tokenizers diverge significantly
(little vocabulary overlap), at the cost of a few additional hyperparameters.

## `sft` — black-box / supervised fine-tuning

Pure cross-entropy training. No teacher actor group is spawned; this algorithm
is what `SFTTrainer` uses. See the [SFT user guide](../user_guide/sft.md).

## See also

- [Cross-Tokenizer KD](../user_guide/cross_tokenizer_kd.md) — when to choose
  `simple_ctkd` vs. `dskd`.
- [Extending KDFlow](../reference/extending.md) — how to register your own
  algorithm.
