# Citation

If you use KDFlow in your research or work, please cite the project paper:

```bibtex
@article{zhang2026kdflow,
  title={KDFlow: A User-Friendly and Efficient Knowledge Distillation Framework for Large Language Models},
  author={Zhang, Songming and Zhang, Xue and Zhang, Tong and Hu, Bojie and Chen, Yufeng and Xu, Jinan},
  journal={arXiv preprint arXiv:2603.01875},
  year={2026}
}
```

## Acknowledgements

KDFlow stands on the shoulders of several outstanding open-source projects:

- [**SGLang**](https://github.com/sgl-project/sglang) — for its support of
  hidden-state extraction and exceptional inference efficiency, which power
  KDFlow's teacher inference and rollout pipelines.
- [**OpenRLHF**](https://github.com/OpenRLHF/OpenRLHF) — its model wrapping
  and distributed-training abstractions form the foundation of KDFlow's
  training infrastructure.
- [**slime**](https://github.com/THUDM/slime) — Ray placement-group
  initialisation and SGLang weight-update mechanism, which inspired KDFlow's
  on-policy distillation design.
