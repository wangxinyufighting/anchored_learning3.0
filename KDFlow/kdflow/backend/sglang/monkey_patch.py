"""
Monkey patch for SGLang scheduler's process_batch_result_prefill method.
This allows using numpy() instead of tolist() for hidden_states, which is much faster.
"""

from __future__ import annotations

import logging
import os
import time
import inspect
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from kdflow.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import (
        EmbeddingBatchResult,
        GenerationBatchResult,
        ScheduleBatch,
        Scheduler,
    )

logger = init_logger(__name__)

# Flag to prevent multiple patch applications
_PATCH_APPLIED = False


def _get_req_is_chunked(req) -> int:
    if hasattr(req, "is_chunked"):
        return getattr(req, "is_chunked", 0)
    return getattr(req, "inflight_middle_chunks", 0)


def _decrement_req_is_chunked(req) -> None:
    if hasattr(req, "is_chunked"):
        req.is_chunked -= 1
    elif hasattr(req, "inflight_middle_chunks"):
        req.inflight_middle_chunks -= 1


def _safe_getattr(obj, name, default=None):
    return getattr(obj, name, default)


def _call_first_available(obj, names, *args, **kwargs):
    for name in names:
        fn = getattr(obj, name, None)
        if fn is not None:
            return fn(*args, **kwargs)
    return None


def _update_req_finish_state(req) -> None:
    if hasattr(req, "check_finished"):
        req.check_finished()
        return
    if hasattr(req, "update_finish_state"):
        req.update_finish_state()


def _maybe_set_prefill_finished_time(req) -> None:
    time_stats = getattr(req, "time_stats", None)
    if time_stats is None:
        return

    if hasattr(time_stats, "prefill_finished_ts"):
        if time_stats.prefill_finished_ts == 0.0:
            time_stats.prefill_finished_ts = time.time()
    elif hasattr(time_stats, "set_prefill_finished_time"):
        time_stats.set_prefill_finished_time()


def _maybe_set_completion_time(req) -> None:
    time_stats = getattr(req, "time_stats", None)
    if time_stats is not None and hasattr(time_stats, "completion_time"):
        time_stats.completion_time = time.perf_counter()


def _maybe_cache_unfinished_req(processor, batch, req) -> None:
    decoding_reqs = _safe_getattr(batch, "decoding_reqs", None)
    if decoding_reqs and req in decoding_reqs:
        return

    tree_cache = getattr(processor, "tree_cache", None)
    if tree_cache is not None and hasattr(tree_cache, "cache_unfinished_req"):
        tree_cache.cache_unfinished_req(req)


def _maybe_collect_routed_experts(processor, req) -> None:
    _call_first_available(
        processor,
        ("maybe_collect_routed_experts", "_maybe_collect_routed_experts"),
        req,
    )


def _maybe_collect_customized_info(processor, i, req, logits_output) -> None:
    _call_first_available(
        processor,
        ("maybe_collect_customized_info", "_maybe_collect_customized_info"),
        i,
        req,
        logits_output,
    )


def _stream_output(processor, batch, skip_stream_req) -> None:
    _call_first_available(
        processor,
        ("stream_output", "_stream_output"),
        batch.reqs,
        batch.return_logprob,
        skip_stream_req,
    )


def _maybe_log_prefill_stats(processor, batch, result) -> None:
    if not getattr(processor, "current_scheduler_metrics_enabled", False):
        return
    log_prefill_stats = getattr(processor, "log_prefill_stats", None)
    if log_prefill_stats is None:
        return
    can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
    log_prefill_stats(
        prefill_stats=batch.prefill_stats,
        can_run_cuda_graph=can_run_cuda_graph,
        dp_cooperation_info=batch.dp_cooperation_info,
    )


def process_batch_result_prefill_patched(
    self: "Scheduler",
    batch: "ScheduleBatch",
    result: Union["GenerationBatchResult", "EmbeddingBatchResult"],
):
    """
    Patched version of process_batch_result_prefill.
    Key change: Use .numpy() instead of .tolist() for hidden_states (much faster).
    """
    from sglang.srt.environ import envs
    from sglang.srt.managers.io_struct import AbortReq
    # from sglang.srt.managers.schedule_batch import RequestStage
    from sglang.srt.mem_cache.common import release_kv_cache
    # from sglang.srt.tracing.trace import trace_slice

    skip_stream_req = None

    if self.is_generation:
        if result.copy_done is not None:
            result.copy_done.synchronize()

        (
            logits_output,
            next_token_ids,
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
        )

        # Move next_token_ids and logprobs to cpu
        next_token_ids = next_token_ids.tolist()
        if batch.return_logprob:
            if logits_output.next_token_logprobs is not None:
                logits_output.next_token_logprobs = (
                    logits_output.next_token_logprobs.tolist()
                )
            if logits_output.input_token_logprobs is not None:
                logits_output.input_token_logprobs = tuple(
                    logits_output.input_token_logprobs.tolist()
                )

        hidden_state_offset = 0

        # Check finish conditions
        logprob_pt = 0

        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            if req.finished() or _safe_getattr(req, "is_retracted", False):
                # decode req in mixed batch or retracted req
                continue

            if _get_req_is_chunked(req) <= 0:
                _maybe_set_prefill_finished_time(req)

                # req output_ids are set here
                req.output_ids.append(next_token_id)
                _update_req_finish_state(req)

                if req.finished():
                    _maybe_collect_routed_experts(self, req)
                    release_kv_cache(req, self.tree_cache)
                    _maybe_set_completion_time(req)
                else:
                    # This updates radix so others can match
                    _maybe_cache_unfinished_req(self, batch, req)

                _maybe_collect_customized_info(self, i, req, logits_output)

                if batch.return_logprob:
                    assert extend_logprob_start_len_per_req is not None
                    assert extend_input_len_per_req is not None
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]

                    num_input_logprobs = self._calculate_num_input_logprobs(
                        req, extend_input_len, extend_logprob_start_len
                    )

                    if req.return_logprob:
                        self.add_logprob_return_values(
                            i,
                            req,
                            logprob_pt,
                            next_token_ids,
                            num_input_logprobs,
                            logits_output,
                        )
                    logprob_pt += num_input_logprobs

                # === KEY CHANGE: Use .numpy() instead of .tolist() ===
                if (
                    req.return_hidden_states
                    and logits_output.hidden_states is not None
                ):
                    req.hidden_states.append(
                        logits_output.hidden_states[
                            hidden_state_offset : (
                                hidden_state_offset := hidden_state_offset
                                + len(req.origin_input_ids)
                            )
                        ]
                        .half()
                        .cpu()
                        .numpy()
                    )

                if _safe_getattr(req, "grammar", None) is not None:
                    # FIXME: this try-except block is for handling unexpected xgrammar issue.
                    try:
                        req.grammar.accept_token(next_token_id)
                    except ValueError as e:
                        # Grammar accept_token can raise ValueError if the token is not in the grammar.
                        # This can happen if the grammar is not set correctly or the token is invalid.
                        logger.error(
                            f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                        )
                        self.abort_request(AbortReq(rid=req.rid))
                    req.grammar.finished = req.finished()

                # trace_slice(
                #     RequestStage.PREFILL_FORWARD,
                #     req.rid,
                #     auto_next_anon=not req.finished(),
                #     thread_finish_flag=req.finished(),
                # )

            else:
                # being chunked reqs' prefill is not finished
                _decrement_req_is_chunked(req)
                # There is only at most one request being currently chunked.
                # Because this request does not finish prefill,
                # we don't want to stream the request currently being chunked.
                skip_stream_req = req

                # Incrementally update input logprobs.
                if batch.return_logprob:
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    if extend_logprob_start_len < extend_input_len:
                        # Update input logprobs.
                        num_input_logprobs = self._calculate_num_input_logprobs(
                            req, extend_input_len, extend_logprob_start_len
                        )
                        if req.return_logprob:
                            self.add_input_logprob_return_values(
                                i,
                                req,
                                logits_output,
                                logprob_pt,
                                num_input_logprobs,
                                last_prefill_chunk=False,
                            )
                        logprob_pt += num_input_logprobs

                # trace_slice(
                #     RequestStage.PREFILL_CHUNKED_FORWARD,
                #     req.rid,
                #     auto_next_anon=True,
                # )

    else:  # embedding or reward model
        if result.copy_done is not None:
            result.copy_done.synchronize()

        is_sparse = envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set()

        embeddings = result.embeddings

        if is_sparse:
            batch_ids, token_ids = embeddings.indices()
            values = embeddings.values()

            embeddings = [{} for _ in range(embeddings.size(0))]
            for i in range(batch_ids.shape[0]):
                embeddings[batch_ids[i].item()][token_ids[i].item()] = values[
                    i
                ].item()
        else:
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.tolist()
            else:
                embeddings = [tensor.tolist() for tensor in embeddings]

        # Check finish conditions
        for i, req in enumerate(batch.reqs):
            if _safe_getattr(req, "is_retracted", False):
                continue

            req.embedding = embeddings[i]
            if _get_req_is_chunked(req) <= 0:
                # Dummy output token for embedding models
                req.output_ids.append(0)
                _update_req_finish_state(req)

                if req.finished():
                    release_kv_cache(req, self.tree_cache)
                else:
                    _maybe_cache_unfinished_req(self, batch, req)
            else:
                # being chunked reqs' prefill is not finished
                _decrement_req_is_chunked(req)

            # trace_slice(
            #     RequestStage.PREFILL_FORWARD,
            #     req.rid,
            #     auto_next_anon=not req.finished(),
            #     thread_finish_flag=req.finished(),
            # )

    _stream_output(self, batch, skip_stream_req)
    _maybe_log_prefill_stats(self, batch, result)


def apply_patch():
    """
    Apply the monkey patch to SGLang's SchedulerOutputProcessorMixin.
    
    This function is idempotent - calling it multiple times is safe.
    Returns True if patch was applied (or already applied), False otherwise.
    """
    global _PATCH_APPLIED
    
    if _PATCH_APPLIED:
        return True
    
    patch_targets = []

    try:
        from sglang.srt.managers.scheduler_output_processor_mixin import (
            SchedulerOutputProcessorMixin,
        )
        patch_targets.append(SchedulerOutputProcessorMixin)
    except ImportError as e:
        print(f"[monkey_patch] Cannot import legacy SchedulerOutputProcessorMixin: {e}", flush=True)

    try:
        import sglang.srt.managers.scheduler as scheduler_module
        for _, obj in inspect.getmembers(scheduler_module, inspect.isclass):
            if hasattr(obj, "process_batch_result_prefill"):
                patch_targets.append(obj)
    except ImportError as e:
        print(f"[monkey_patch] Cannot import scheduler module: {e}", flush=True)

    if not patch_targets:
        return False

    try:
        for target_cls in patch_targets:
            current_method = getattr(target_cls, "process_batch_result_prefill", None)
            if current_method is None:
                continue

            if getattr(current_method, "_kdflow_patched", False):
                _PATCH_APPLIED = True
                print(
                    f"[monkey_patch] Patch already applied on {target_cls.__name__}, PID={os.getpid()}",
                    flush=True,
                )
                return True

            process_batch_result_prefill_patched._kdflow_patched = True
            target_cls.process_batch_result_prefill = process_batch_result_prefill_patched

            _PATCH_APPLIED = True
            print(
                f"[monkey_patch] SUCCESS: process_batch_result_prefill patched on "
                f"{target_cls.__module__}.{target_cls.__name__}! PID={os.getpid()}",
                flush=True,
            )

        if _PATCH_APPLIED:
            return True
        
        print("[monkey_patch] No patchable process_batch_result_prefill method found.", flush=True)
        return False
    except Exception as e:
        print(f"[monkey_patch] Error applying patch: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False


def is_patch_applied():
    """Check if the patch has been applied in this process."""
    return _PATCH_APPLIED
