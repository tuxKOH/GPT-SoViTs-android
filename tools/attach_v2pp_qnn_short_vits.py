#!/usr/bin/env python3
"""Add an independently compiled short VITS capacity to an existing SM8750 voice attachment.

The old 512-capacity contexts remain untouched and are the fallback for long segments and
temporary references. This is a conversion-time operation; Android only reads the descriptor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

from assemble_v2pp_qnn_attachments import (
    partition_graph_specs, runtime_partition_contract, wrap_component,
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--partitions", required=True, type=Path)
    parser.add_argument("--component", required=True, action="append", type=Path)
    parser.add_argument("--capacity", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.base.resolve() == args.output.resolve() or args.output.exists():
        raise SystemExit("output must be a new file distinct from the base attachment")
    prefix = "vits_short"
    stages = partition_graph_specs(args.partitions, args.component, prefix)
    with zipfile.ZipFile(args.base) as source, tempfile.TemporaryDirectory(
        prefix="gsv-short-vits-", dir=args.output.parent
    ) as temporary:
        original = {item.filename for item in source.infolist()}
        manifest = json.loads(source.read("manifest.json"))
        executor = json.loads(source.read("runtime/qnn/executor.json"))
        attachment = json.loads(source.read("runtime/qnn/attachment.json"))
        if manifest["target_soc"] != "snapdragon_8_elite":
            raise ValueError("this build is restricted to SM8750")
        if int(executor["shapes"]["semantic_capacity"]) <= args.capacity:
            raise ValueError("short capacity must be below the existing fallback")
        if "vits_short" in executor["graphs"]:
            raise ValueError("base attachment already has a short VITS bucket")
        wrapped = {}
        for stage in stages:
            name = stage["name"]
            component = stage["component"]
            info = json.loads((component / "manifest.json").read_text())
            for key in ("target_soc", "target_soc_family", "htp_arch", "qairt_version"):
                if info[key] != manifest[key]:
                    raise ValueError(f"{name} {key} differs from the base attachment")
            onnx, binary = wrap_component(
                name=name, source=stage["source"], component=component,
                output=Path(temporary),
            )
            wrapped[f"runtime/qnn/{name}.onnx"] = onnx.read_bytes()
            wrapped[f"runtime/qnn/{name}.bin"] = binary.read_bytes()
            attachment["components"][name] = {
                "source_onnx_sha256": info["source_onnx_sha256"],
                "static_inputs": info["static_inputs"],
            }
        executor["graphs"][prefix] = [
            runtime_partition_contract(stage, Path(temporary) / f"{stage['name']}.onnx")
            for stage in stages
        ]
        executor["shapes"]["short_semantic_capacity"] = args.capacity
        replacements = dict(wrapped)
        replacements["runtime/qnn/executor.json"] = (
            json.dumps(executor, ensure_ascii=False, indent=2) + "\n"
        ).encode()
        replacements["runtime/qnn/attachment.json"] = (
            json.dumps(attachment, ensure_ascii=False, indent=2) + "\n"
        ).encode()
        if original.intersection(wrapped):
            raise ValueError("short graph paths already exist in the base package")
        files = {item["path"]: item for item in manifest["files"]}
        for path, payload in replacements.items():
            files[path] = {"path": path, "size": len(payload), "sha256": digest(payload)}
        manifest["files"] = list(files.values())
        replacements["manifest.json"] = (
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        ).encode()
        pending = args.output.with_name(args.output.name + ".pending")
        if pending.exists():
            raise ValueError(f"stale pending output exists: {pending}")
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as output:
            for item in source.infolist():
                output.writestr(item, replacements.pop(item.filename, source.read(item)))
            for path, payload in replacements.items():
                output.writestr(path, payload)
        pending.replace(args.output)
    print(f"Wrote {args.output} with {len(stages)} short VITS partitions")


if __name__ == "__main__":
    main()
