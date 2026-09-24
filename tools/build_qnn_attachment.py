#!/usr/bin/env python3
"""Build one SoC-specific QNN pipeline or voice attachment package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from build_qnn_htp_context import ANDROID_QNN_RUNTIME_VERSION, TARGETS, TARGET_SOC_FAMILY
from model_profiles import PROFILES


QNN_SUFFIX = ".qnn.gsvm"


@dataclass(frozen=True)
class Payload:
    source: Path
    destination: str
    size: int
    sha256: str


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_destination(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"invalid package destination: {value}")
    return str(path)


def parse_mapping(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise ValueError(f"expected NAME=PATH, received {value!r}")
    return name, Path(raw_path).resolve()


def load_component(name: str, root: Path, target_soc: str) -> dict:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"QNN component {name} has no manifest.json: {root}")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = TARGETS[target_soc]
    expected = {
        "format": "gsv-qnn-compiled-component",
        "executor": "qnn-htp",
        "target_soc": target_soc,
        "target_soc_family": TARGET_SOC_FAMILY,
        "target_asic": target.asic,
        "target_soc_model": target.soc_model,
        "htp_arch": target.htp_arch,
        "qnn_runtime_version": ANDROID_QNN_RUNTIME_VERSION,
        "precision": "fp16",
        "quantization": "none",
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise ValueError(
                f"QNN component {name} has {key}={document.get(key)!r}, expected {value!r}"
            )
    if document.get("deployable") is not False:
        raise ValueError(f"QNN component {name} must remain a non-deployable build component")
    for item in document.get("files", []):
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["size"] or digest(path) != item["sha256"]:
            raise ValueError(f"QNN component {name} payload verification failed: {item['path']}")
    return document


def make_payload(source: Path, destination: str) -> Payload:
    source = source.resolve()
    if not source.is_file():
        raise ValueError(f"payload does not exist: {source}")
    destination = safe_destination(destination)
    return Payload(source, destination, source.stat().st_size, digest(source))


def validate_executor(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "format": "gsv-qnn-executor",
        "format_version": 1,
        "operation": "synthesize_utf8_to_pcm16",
        "runtime_abi_version": 1,
        "complete": True,
        "utf8_text_input": True,
        "pcm16_output": True,
        "cpu_neural_fallback": False,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise ValueError(f"executor descriptor has {key}={document.get(key)!r}, expected {value!r}")
    return document


def build_attachment(
    *,
    output: Path,
    role: str,
    name: str,
    version: str,
    frontend_profile: str,
    target_soc: str,
    components: dict[str, Path],
    payload_mappings: list[tuple[Path, str]],
    executor_descriptor: Path | None,
    base_model: Path | None,
    deployable: bool,
    product_version: str = "v3.2.0+0",
) -> dict:
    output = output.resolve()
    if not output.name.endswith(QNN_SUFFIX):
        raise ValueError(f"QNN attachment filename must end with {QNN_SUFFIX}")
    if role not in ("pipeline", "model"):
        raise ValueError("role must be pipeline or model")
    if version not in PROFILES:
        raise ValueError(f"unsupported model version: {version}")
    if target_soc not in TARGETS:
        raise ValueError(f"unsupported QNN target: {target_soc}")
    if not components:
        raise ValueError("at least one verified QNN component is required")

    component_documents = {
        component_name: load_component(component_name, root.resolve(), target_soc)
        for component_name, root in sorted(components.items())
    }
    qairt_versions = {document.get("qairt_version") for document in component_documents.values()}
    if len(qairt_versions) != 1 or None in qairt_versions:
        raise ValueError(f"QNN components use inconsistent QAIRT versions: {sorted(qairt_versions)}")
    qairt_version = next(iter(qairt_versions))

    payloads = [make_payload(source, destination) for source, destination in payload_mappings]
    attachment_descriptor = {
        "format": "gsv-qnn-attachment",
        "format_version": 1,
        "role": role,
        "components": {
            component_name: {
                "source_onnx_sha256": document.get("source_onnx_sha256", ""),
                "static_inputs": document.get("static_inputs", {}),
            }
            for component_name, document in component_documents.items()
        },
    }
    generated: list[tuple[str, bytes]] = [
        (
            "runtime/qnn/attachment.json",
            (json.dumps(attachment_descriptor, ensure_ascii=False, indent=2) + "\n").encode(),
        )
    ]
    backend_artifact = "runtime/qnn/attachment.json"
    executor_document: dict | None = None
    if executor_descriptor is not None:
        executor_descriptor = executor_descriptor.resolve()
        executor_document = validate_executor(executor_descriptor)
        payloads.append(make_payload(executor_descriptor, "runtime/qnn/executor.json"))
        backend_artifact = "runtime/qnn/executor.json"
    if deployable and role == "model" and executor_descriptor is None:
        raise ValueError("deployable QNN model attachments require a complete high-level executor descriptor")
    if role == "pipeline" and executor_descriptor is not None:
        raise ValueError("the voice/model attachment owns the high-level executor")
    base_model_sha256 = ""
    if base_model is not None:
        base_model = base_model.resolve()
        if role != "model":
            raise ValueError("--base-model is only valid for model attachments")
        if not base_model.is_file():
            raise ValueError(f"base model package does not exist: {base_model}")
        base_model_sha256 = digest(base_model)
    if deployable and role == "model" and not base_model_sha256:
        raise ValueError("deployable QNN model attachments require --base-model binding")

    destinations = [item.destination for item in payloads] + [item[0] for item in generated]
    if len(destinations) != len(set(destinations)):
        raise ValueError("duplicate QNN attachment payload destination")

    target = TARGETS[target_soc]
    profile = PROFILES[version]
    base_bundle = f"gsvm:{profile.id}:{frontend_profile}:api1"
    backend_bundle = f"{base_bundle}:qnn-htp:{target.asic}:qairt-{qairt_version}"
    files = [
        {"path": item.destination, "size": item.size, "sha256": item.sha256}
        for item in payloads
    ]
    for destination, data in generated:
        files.append(
            {
                "path": destination,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    files.sort(key=lambda item: item["path"])
    artifact_role = f"qnn-{role}-attachment"
    manifest = {
        "format": "gsvm-deploy",
        "format_version": 1,
        "name": name,
        "model_version": profile.id,
        "product_version": product_version,
        "sample_rate": profile.sample_rate,
        "executor": "qnn-htp",
        "entrypoint": "synthesize_utf8_to_pcm16",
        "api_version": 1,
        "deployable": deployable,
        "artifact_role": artifact_role,
        "requires_role": "qnn-model-attachment" if role == "pipeline" else "qnn-pipeline-attachment",
        "bundle_id": backend_bundle,
        "attachment_for": base_bundle,
        "frontend_profile": frontend_profile,
        "target_soc": target_soc,
        "target_soc_family": TARGET_SOC_FAMILY,
        "target_asic": target.asic,
        "target_soc_model": target.soc_model,
        "supported_target_socs": [target_soc],
        "htp_arch": target.htp_arch,
        "qairt_version": qairt_version,
        "qnn_runtime_version": ANDROID_QNN_RUNTIME_VERSION,
        "backend_artifact": backend_artifact,
        "precision": "fp16",
        "quantization": "none",
        "cpu_neural_fallback": False,
        "files": files,
    }
    if executor_document is not None:
        manifest["runtime_options_version"] = int(
            executor_document.get("runtime_options_version", 0)
        )
        manifest["reference_input_version"] = int(
            executor_document.get("reference_input_version", 0)
        )
        reference = executor_document.get("reference")
        if manifest["reference_input_version"] >= 1:
            if not isinstance(reference, dict):
                raise ValueError("reference-enabled QNN executor is missing its input policy")
            manifest["reference_input"] = {
                "preset_when_omitted": True,
                "duration_policy": reference.get("duration_policy"),
                "pcm": [
                    {
                        "sample_rate": 16000,
                        "channels": 1,
                        "dtype": "float32",
                        "samples": int(reference.get("pcm_16k_samples", 0)),
                    },
                    {
                        "sample_rate": 32000,
                        "channels": 1,
                        "dtype": "float32",
                        "samples": int(reference.get("pcm_32k_samples", 0)),
                    },
                ],
                "prompt": "utf8_text",
                "conditioning_stage": "converted_artifact",
            }
            if manifest["reference_input"]["duration_policy"] != "exact_samples":
                raise ValueError("QNN runtime reference input must declare exact_samples")
            if any(item["samples"] <= 0 for item in manifest["reference_input"]["pcm"]):
                raise ValueError("QNN runtime reference input has invalid PCM capacities")
    if base_model_sha256:
        manifest["base_model_sha256"] = base_model_sha256
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(output.suffix + ".partial")
    with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for item in payloads:
            archive.write(item.source, item.destination)
        for destination, data in generated:
            archive.writestr(destination, data)
    os.replace(pending, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("pipeline", "model"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True, choices=sorted(PROFILES))
    parser.add_argument("--product-version", default="v3.2.0+0")
    parser.add_argument("--frontend-profile", default="portable-char-v1")
    parser.add_argument("--target-soc", required=True, choices=sorted(TARGETS))
    parser.add_argument("--component", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--payload", action="append", default=[], metavar="SOURCE=DESTINATION")
    parser.add_argument("--executor-descriptor", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--deployable", action="store_true")
    args = parser.parse_args()
    components = dict(parse_mapping(value) for value in args.component)
    payloads = []
    for value in args.payload:
        source, separator, destination = value.partition("=")
        if not separator:
            raise SystemExit(f"expected SOURCE=DESTINATION, received {value!r}")
        payloads.append((Path(source), destination))
    manifest = build_attachment(
        output=args.output,
        role=args.role,
        name=args.name,
        version=args.version,
        frontend_profile=args.frontend_profile,
        target_soc=args.target_soc,
        components=components,
        payload_mappings=payloads,
        executor_descriptor=args.executor_descriptor,
        base_model=args.base_model,
        deployable=args.deployable,
        product_version=args.product_version,
    )
    print(
        f"Created {args.output.resolve()} role={manifest['artifact_role']} "
        f"target={manifest['target_asic']} deployable={manifest['deployable']}"
    )


if __name__ == "__main__":
    main()
