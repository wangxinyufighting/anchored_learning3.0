# Cross-Tokenizer Knowledge Distillation

KDFlow natively supports distilling between models with **different tokenizers**
(e.g. Qwen3 → Llama3.2, or vice versa). Two algorithms are provided:

| `--kd_algorithm`  | Idea                                                                                                  | When to use                                |
|-------------------|--------------------------------------------------------------------------------------------------------|--------------------------------------------|
| `simple_ctkd`     | Restrict the KD loss to the **overlapping vocabulary** between the two tokenizers (recommended)        | Default choice — simple & strong baseline |
| `dskd`            | **Dual-Space KD**: project teacher and student into a shared space using token-alignment (`eta`/`cma`) | Suggested for off-policy distillation    |

Both work in **off-policy** and **on-policy** modes — pick the matching CLI
entry point.

## Entry points

| Mode       | Command                                          |
|------------|--------------------------------------------------|
| Off-policy | `python -m kdflow.cli.train_kd_off_policy ...`   |
| On-policy  | `python -m kdflow.cli.train_kd_on_policy ...`    |

The cross-tokenizer behaviour is selected purely via `--kd_algorithm`.

## Example scripts

```bash
# Off-policy — simple cross-tokenizer KD (recommended)
bash examples/cross_tokenizer_kd/run_qwen3_30b_a3b_to_llama3_2_3b_offpolicy_simple_ctkd.sh

# Off-policy — DSKD
bash examples/cross_tokenizer_kd/run_qwen3_30b_a3b_to_llama3_2_3b_offpolicy.sh

# On-policy — simple cross-tokenizer KD
bash examples/cross_tokenizer_kd/run_qwen3_30b_a3b_to_llama3_2_3b_onpolicy_simple_ctkd.sh

# On-policy — DSKD
bash examples/cross_tokenizer_kd/run_qwen3_30b_a3b_to_llama3_2_3b_onpolicy.sh
```

## SimpleCrossTokenizerKD

```bash
--kd_algorithm simple_ctkd
--kd_loss_fn   kl              # any registered loss works
--kd_ratio     0.5
```

`SimpleCrossTokenizerKD` computes the KD loss only on tokens that exist in
**both** vocabularies (after a one-time alignment pass), keeping it cheap and
stable.

## DSKD (Dual-Space Knowledge Distillation)

```bash
--kd_algorithm dskd
--dskd_token_align eta         # 'eta' or 'cma' alignment strategy
--dskd_topk_vocab 1000         # top-k vocab tokens for the projector init (-1 = all)
--dskd_projector_lr 1e-4       # learning rate for DSKD projectors
```

DSKD adds small projector heads on top of student and teacher representations,
trained jointly with the student. Two token-alignment strategies are supported:

- `eta` — empirical token alignment (default).
- `cma` — cross-model alignment.

See `kdflow/algorithms/dskd.py` for the full implementation.

## When to choose which?

| Situation                                                  | Suggested algorithm  |
|-------------------------------------------------------------|----------------------|
| First experiment, want a strong baseline fast               | `simple_ctkd`        |
| Vocabularies overlap heavily (same family)                  | `simple_ctkd`        |
| Vocabularies are very different and you want full coverage  | `dskd`               |
| You need extra hyperparameter knobs to chase last-mile gains | `dskd`              |

## See also

- [KD algorithms](../concepts/algorithms.md) — what `simple_ctkd` and `dskd` do.
- [Off-Policy KD](off_policy_kd.md) and [On-Policy KD](on_policy_kd.md) — the
  surrounding training loops.
