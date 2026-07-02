import os


RUNTIME_ENV_INHERIT_VARS = [
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CONDA_PREFIX",
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "TORCH_EXTENSIONS_DIR",
    "TRITON_CACHE_DIR",
    "CUDA_MODULE_LOADING",
]


def get_runtime_env_vars(extra=None):
    """Return environment variables that Ray workers should inherit.

    Ray runtime_env overlays environment variables for actors and workers. CUDA
    library paths are especially important because SGLang starts scheduler
    subprocesses inside Ray actors.
    """
    env_vars = {
        name: os.environ[name]
        for name in RUNTIME_ENV_INHERIT_VARS
        if os.environ.get(name)
    }
    if extra:
        env_vars.update(extra)
    return env_vars


def normalize_gpu_id_for_visible_devices(gpu_id, env_vars=os.environ):
    """Map a Ray physical GPU id to the process-local CUDA ordinal.

    When users launch on shared machines with CUDA_VISIBLE_DEVICES=2,3, Ray may
    report physical ids such as 2 and 3. Inside that process, CUDA sees them as
    ordinal 0 and 1, so SGLang must receive the local ordinal.
    """
    gpu_id_str = str(gpu_id)
    if gpu_id_str.endswith(".0"):
        gpu_id_str = gpu_id_str[:-2]

    visible_devices = env_vars.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return int(float(gpu_id))

    visible = [item.strip() for item in visible_devices.split(",") if item.strip()]
    if gpu_id_str in visible:
        return visible.index(gpu_id_str)

    try:
        gpu_id_int = int(float(gpu_id))
    except (TypeError, ValueError):
        return gpu_id

    if 0 <= gpu_id_int < len(visible):
        return gpu_id_int
    return gpu_id_int


# Address https://github.com/ray-project/ray/issues/51117
# This function is used to get the bundle indices of a placement group
# and ensure that the bundles placed on the same node are grouped together.
def get_bundle_indices(placement_group, index, length):
    import ray

    pg_infos = ray.util.placement_group_table(placement_group)

    node_id_to_bundles = {}
    for bundle, node_id in pg_infos["bundles_to_node_id"].items():
        node_id_to_bundles.setdefault(node_id, []).append(bundle)

    sorted_bundle_indices = sum(node_id_to_bundles.values(), [])
    return sorted_bundle_indices[index * length : (index + 1) * length]


def ray_noset_visible_devices(env_vars=os.environ):
    # Refer to
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
    # https://github.com/ray-project/ray/blob/3b9e729f6a669ffd85190f901f5e262af79771b0/python/ray/_private/accelerators/amd_gpu.py#L114-L115
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
    ]
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def get_physical_gpu_id():
    import torch

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)
