#!/usr/bin/env python3
"""Measure the CPU autoregressive stage without changing a deployable artifact.

The synthetic input isolates GPT token generation. Android end-to-end traces remain the
source of truth for user-facing latency; this script identifies costly operators.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path(".."))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--profile-step", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads < 1 or args.steps < 1 or not 1 <= args.profile_step <= args.steps:
        parser.error("threads and steps must be positive; profile-step must be within steps")

    checkpoint = args.checkpoint.resolve()
    upstream = args.upstream.resolve()
    os.chdir(upstream)
    sys.path.insert(0, str(upstream / "GPT_SoVITS"))
    from export_torch_script import T2SModel, get_raw_t2s_model
    from stream_v2pro import StreamT2SModel

    torch.set_num_threads(args.threads)
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = StreamT2SModel(T2SModel(get_raw_t2s_model(source).float().eval())).eval()
    prompt = torch.ones(1, 100, dtype=torch.long)
    reference_phones = torch.ones(1, 30, dtype=torch.long)
    text_phones = torch.ones(1, 20, dtype=torch.long)
    reference_bert = torch.zeros(30, 1024)
    text_bert = torch.zeros(20, 1024)
    with torch.inference_mode():
        started = time.perf_counter()
        y_len, y, xy_pos, k_cache, v_cache = model.pre_infer(
            prompt, reference_phones, text_phones, reference_bert, text_bert, 10,
        )
        prefill_ms = (time.perf_counter() - started) * 1000
        step_ms = []
        operator_table = ""
        for index in range(1, args.steps + 1):
            started = time.perf_counter()
            if index == args.profile_step:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                    y, xy_pos, _, k_cache, v_cache = model(
                        index, 10, y_len, y, xy_pos, k_cache, v_cache,
                    )
                operator_table = trace.key_averages().table(sort_by="self_cpu_time_total", row_limit=15)
            else:
                y, xy_pos, _, k_cache, v_cache = model(
                    index, 10, y_len, y, xy_pos, k_cache, v_cache,
                )
            step_ms.append((time.perf_counter() - started) * 1000)

    report = {
        "threads": args.threads,
        "prefill_ms": prefill_ms,
        "steps": args.steps,
        "first_20_mean_ms": sum(step_ms[:20]) / min(20, len(step_ms)),
        "last_20_mean_ms": sum(step_ms[-20:]) / min(20, len(step_ms)),
        "all_steps_ms": sum(step_ms),
        "profiled_step": args.profile_step,
        "operator_table": operator_table,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "operator_table"}, indent=2))
    print(operator_table)


if __name__ == "__main__":
    main()
