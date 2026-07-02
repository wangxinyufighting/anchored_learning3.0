# Multimodal (VLM) Distillation

Since **v0.1.1**, KDFlow supports distillation with **vision-language models**
(VLMs) such as Qwen3-VL. The teacher and / or student can be a VLM, and the
data pipeline handles image keys end-to-end.

## Example scripts

```bash
# Off-policy VLM KD
bash examples/off_policy_kd/run_qwen3_vl_30b_a3b_to_4b.sh

# On-policy VLM KD
bash examples/on_policy_kd/run_qwen3_vl_30b_a3b_to_4b.sh
```

## What changes vs. text-only

The training command is identical to its text-only counterpart, with three
practical differences:

1. The model paths point to VLM checkpoints (e.g. `Qwen3-VL-30B-A3B`).
2. The dataset has an **image column**, exposed via:
   ```bash
   --image_key images
   ```
3. The chat template applies the VLM-specific multimodal placeholders (kept
   automatically by `--apply_chat_template True`).

Everything else — FSDP2, packing, dynamic batch size, LoRA, KD algorithm /
loss, on-policy rollouts — works the same.

## Engine compatibility

To benefit from the latest VLM features (Qwen3.5 etc.):

- Install **`sglang>=0.5.10`** (avoid 0.5.9 due to a known critical VLM bug —
  [sglang#19335](https://github.com/sgl-project/sglang/issues/19335)).
- Use a `transformers` version that matches your VLM family.

## Tips

- VLM teachers are usually large; consider quantisation
  (`--teacher_quantization awq` / `fp8`) and TP/PP sharding
  (`--teacher_tp_size`, `--teacher_pp_size`) to fit them on your hardware.
- Image-heavy batches benefit a lot from `--use_dynamic_bsz True` together
  with `--max_token_len_per_gpu <N>`.

## See also

- [Off-Policy KD](off_policy_kd.md)
- [On-Policy KD](on_policy_kd.md)
- [Arguments → Data Arguments](../reference/arguments.md#data-arguments)
