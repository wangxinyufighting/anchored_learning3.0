# KDFlow

> **A user-friendly and efficient framework for LLM knowledge distillation.**

KDFlow is a scalable knowledge-distillation (KD) framework for large language models.
It decouples the teacher and student backends — running the teacher on
[SGLang](https://github.com/sgl-project/sglang) for high-throughput inference,
and the student on [FSDP2](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
for memory-efficient training — and colocate them **on the same GPUs** through a
sleep / wakeup mechanism. The result is a fast, flexible KD pipeline that scales
from a single 8-GPU node to multi-node TP/PP setups for 200B+ teachers.

<p align="center">
  <img src="https://raw.githubusercontent.com/songmzhang/KDFlow/main/figures/architecture.png"
       alt="KDFlow Architecture" width="80%">
</p>

---

## Why KDFlow?

- **Decoupled infrastructure** — SGLang for teacher inference, FSDP2 for student training.
- **Off-policy KD** — distill from pre-collected teacher hidden states on static datasets.
- **On-policy KD** — student rollouts feed the teacher in a closed loop.
- **Cross-tokenizer KD** — distill across different tokenizers (e.g. Llama → Qwen).
- **Multi-teacher KD** — route samples to domain-specific teachers with `--multi_teacher_config` and `--teacher_routing_key`.
- **SFT (black-box KD)** — single-controller supervised fine-tuning on collected data.
- **Multimodal support** — distill vision-language models such as Qwen3-VL.
- **GPU co-location** — teacher and student share GPUs via the sleep/wakeup mechanism.
- **Hidden-state transfer** — only teacher hidden states (not full logits) cross the
  process boundary, via shared memory.
- **Pluggable algorithms / losses** — built-in `vanilla_kd`, `simple_ctkd`, `dskd`, `sft`
  plus 9 registered loss functions; new ones can be added with a single decorator.
- **LoRA + Liger Kernel + dynamic batching** for efficient student training.
- **W&B integration** out of the box.
- **1.4× – 6×** faster than mainstream KD frameworks.

---

## Where to go next

**Get up and running**

Install KDFlow and run your first distillation job in minutes.

- [Installation](getting_started/installation.md)
- [Quickstart](getting_started/quickstart.md)

**User guide**

Walk through every supported training scenario with annotated example scripts.

- [Off-Policy KD](user_guide/off_policy_kd.md)
- [On-Policy KD](user_guide/on_policy_kd.md)
- [Multi-Teacher KD](user_guide/multi_teacher_kd.md)
- [Cross-Tokenizer KD](user_guide/cross_tokenizer_kd.md)
- [Supervised Fine-Tuning](user_guide/sft.md)
- [Multimodal (VLM) Distillation](user_guide/multimodal.md)

**Concepts**

Understand the architecture, supported KD algorithms, and loss functions.

- [Architecture](concepts/architecture.md)
- [KD Algorithms](concepts/algorithms.md)
- [KD Loss Functions](concepts/losses.md)

**Reference**

Full CLI surface, argument tables, and extension hooks.

- [CLI Entry Points](reference/cli.md)
- [Arguments](reference/arguments.md)
- [Extending KDFlow](reference/extending.md)

---

## Citation

If KDFlow helps your research, please cite:

```bibtex
@article{zhang2026kdflow,
  title={KDFlow: A User-Friendly and Efficient Knowledge Distillation Framework for Large Language Models},
  author={Zhang, Songming and Zhang, Xue and Zhang, Tong and Hu, Bojie and Chen, Yufeng and Xu, Jinan},
  journal={arXiv preprint arXiv:2603.01875},
  year={2026}
}
```

KDFlow is released under the [MIT License](about/license.md).
