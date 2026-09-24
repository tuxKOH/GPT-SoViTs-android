# GSV Mobile v3.2.0

<p align="center">
  <img src="docs/gsv-mobile-icon.svg" alt="GSV Mobile icon" width="132" />
</p>

<p align="center"><strong>The best-optimized mobile version of GPT-SoVITS.</strong><br/>
GPT-SoVITS V2 Pro Plus and V4, converted once and synthesized locally on Android.</p>

<p align="center">
  <a href="#features">Features</a> ·
  <a href="#android-usage">Quick start</a> ·
  <a href="#v100-reference-speed">Benchmarks</a>
</p>

[中文文档](README.zh-CN.md)

GSV Mobile converts and runs [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
V2 Pro Plus and V4 voices locally on Android. Exact FP32 CPU execution remains the correctness
reference. QNN/NPU work is included as an engineering preview only: it has not completed matching
device acceptance and is not a supported v3 feature.

This project is an independent Android deployment implementation. It is not an official
GPT-SoVITS application and is not affiliated with the upstream maintainers.

## Features

- GPT-SoVITS V2 Pro Plus and V4 model conversion
- FP32 weights without quantization or dtype conversion
- English and Chinese Android UI, with system-language selection and an in-app override
- Request-scoped reference audio while retaining the converted reference as the preset
- Shared, persistent V2PP/V4 pipeline components
- Automatic model discovery in `/sdcard/models/gs`, plus the Android system file picker
- Persistent model list backed by Android Storage Access Framework URIs
- Compose interface for synthesis, sampling controls and audio playback
- OpenAI-compatible local `POST /v1/audio/speech` endpoint
- Model-only packages by default; optional self-contained packages with pipeline assets

Ordinary UTF-8 `.gsvm` packages remain the FP32 CPU correctness artifacts. The repository also
contains experimental QNN pipeline and voice attachments ending in `.qnn.gsvm`; do not rely on them
until matching-device acceptance is published.

## Requirements

### Android

- Android 8.0 (API 26) or newer
- 64-bit Android device
- V2 Pro Plus: minimum 8 GB physical RAM (verified)
- V4 large workloads: allow about 8--12 GB of free process memory

### Conversion host

- The original GPT-SoVITS repository checked out next to this directory
- The upstream Python environment and pretrained models
- Enough RAM and disk space for FP32 graph export

The expected source layout is:

```text
GPT-SoVITS/
├── android/       # this project
├── GPT_SoVITS/
├── GPT_weights_v2ProPlus/
└── SoVITS_weights_v2ProPlus/
```

## Build Android

Set the Android SDK path in `local.properties`:

```properties
sdk.dir=/path/to/Android/Sdk
```

Build the debug APK:

```bash
./gradlew :app:assembleDebug
```

The APK is written to `app/build/outputs/apk/debug/app-debug.apk`.

Build the unsigned arm64 engineering APK for device acceptance:

```bash
./gradlew -PacceptanceAbi=arm64-v8a :app:assembleRelease
```

The generated `app-release-unsigned.apk` is intentionally unsigned. Sign it with the product's
long-lived Android signing key before distribution; the debug APK is signed only with the local
Android debug certificate and is suitable for development or acceptance testing, not release.

## Convert From The Command Line

Create a model-only package:

```bash
../gpt/bin/python tools/build_android_cpu_pipeline.py \
  --gpt ../GPT_weights_v2ProPlus/voice.ckpt \
  --sovits ../SoVITS_weights_v2ProPlus/voice.pth \
  --reference ../reference.wav \
  --prompt '参考音频对应文本' \
  --name voice \
  --work build/voice \
  --model-output converted_models/voice-model.gsvm \
  --upstream ..
```

Use `--output converted_models/voice.gsvm` instead of `--model-output` to create a self-contained
package.

## Android Usage

1. Install one or both pipeline components on the first run.
2. Put model-only or self-contained `.gsvm` packages in `/sdcard/models/gs`.
3. Select the saved model from the collapsible model list.
4. Enter text, adjust synthesis options if needed, synthesize and play the WAV result.

Legacy `.gsvm` models are not supported. Regenerate them with the web converter before loading;
the app intentionally does not bypass frontend ABI checks for old packages.

For QNN engineering validation only, keep these four files together in `/sdcard/models/gs`, replacing `<target>` with
`sm8650`, `sm8750`, or `sm8850` for the device:

```text
v2pp-cpu-pipeline.gsvm
firefly-v2pp-cpu.gsvm
v2pp-<target>-pipeline.qnn.gsvm
firefly-v2pp-<target>.qnn.gsvm
```

This workflow is not a supported v3 feature. Install the CPU V2 Pro Plus pipeline first, add the CPU voice model, then import the QNN pipeline
attachment from **Settings**. In **Models**, expand the CPU voice, choose its matching QNN voice
attachment, and select **Load on NPU**. The app verifies the exact base-model hash, SoC, HTP
architecture, QAIRT version and paired backend bundle before loading. There is no CPU/GPU selector
for a QNN attachment: a compatible attachment loads HTP/NPU, and an incompatible device is rejected.

The QNN V2 Pro Plus temporary-reference graphs are specialized for exactly five seconds of decoded audio.
The UI and local API reject another duration instead of silently cropping or padding it. Omitting a
temporary reference always uses the reference embedded in the converted voice.

For development devices:

```bash
adb shell mkdir -p /sdcard/models/gs
adb push converted_models/voice-model.gsvm /sdcard/models/gs/
```

Open **Models**, choose **Scan folder**, and grant all-files access when Android requests it. The
permission is needed only for direct discovery of this engineering directory. **Add voice model**
continues to use the Storage Access Framework and works without broad storage access.

Downloaded pipelines are retained in app-private persistent storage. Model list entries retain
persistable document URIs rather than copying every source package. Missing or moved model files are
removed from the list when loading fails.

CPU timing and the emulator thread sweep are documented in [CPU performance](docs/CPU_PERFORMANCE.md).
The Snapdragon 8 Elite QNN trace and repeated short-phrase PCM cache are documented in [NPU performance](docs/NPU_PERFORMANCE.md). Settings can check GitHub Releases for updates; startup checks in the background and falls back to a mirror when GitHub is unreachable.

## Local OpenAI API

Enable the local API from the Android **Settings** tab. It listens on device loopback port `9880` by
default. Forward the port to a connected development host:

```bash
adb forward tcp:9880 tcp:9880
```

Synthesize speech:

```bash
curl http://127.0.0.1:9880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-sovits-local",
    "voice": "loaded-artifact",
    "input": "这是一段本地语音合成测试。",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output tts.wav
```

Supported extension fields are `temperature`, `top_p`, `top_k`, `repetition_penalty`,
`sample_steps`, `seed` and `language`. Only WAV output is currently supported. With no reference
fields, synthesis uses the reference baked into the converted voice. A request can temporarily
override it with:

- `reference_audio`: Base64-encoded WAV, FLAC, MP3, OGG or M4A bytes, up to 50 MiB
- `reference_text`: required transcript matching the temporary audio
- `reference_language`: `auto`, `zh` or `en`

The override is request-scoped and does not modify the model package or later API calls.

## Package Contract

Android is a thin executor. Conversion handles checkpoint versions, graph construction, frontend
assets, compatibility metadata and weight verification.

Every deployable package exposes one operation:

```text
UTF-8 text + synthesis options -> mono PCM16 audio
```

Model-only packages declare `artifact_role=model`. Shared pipelines declare
`artifact_role=pipeline`. Android verifies package manifests and SHA-256 hashes before combining
compatible artifacts.

Packages produced by the current converter declare `reference_input_version=1`. Older packages
still use their preset reference but are rejected explicitly when a temporary override is supplied.
Newly converted packages carry `product_version=v3.2.0+0`; increment only the `+N` revision when
conversion output or conversion logic changes.

### V100 reference speed

On a Tesla V100-SXM2 16 GB, the upstream V2 Pro Plus Firefly checkpoint (CUDA FP16, seed 1234,
text `你好今天我们一起测试语音合成。`, 32 kHz output) took **4.84 s** for synthesis after
**4.66 s** model load and produced 105,600 samples. This is a desktop-GPU reference, not an
Android performance promise; the run used `ex146.wav` and omitted prompt text to use the audio preset.

### Same-text device comparison

All rows use the exact text `你好今天我们一起测试语音合成。` and seed 1234. Android timings are
from the ASUS AI2501C (Snapdragon 8 Elite / SM8750) acceptance run, with the app otherwise idle.
“First request” includes module/session work that was still lazy at that time; QNN neural CPU
fallback was disabled.

| Backend | Device | Model/session load | Synthesis / first request | Output |
| --- | --- | ---: | ---: | ---: |
| CUDA FP16 | Tesla V100-SXM2 16 GB | 4.66 s | 4.84 s after load | 105,600 samples |
| CPU FP32 staged | ASUS SM8750 | BERT 4.66 s + acoustic 5.41 s (inside first request) | 26.98 s first request | 163,840 samples |
| QNN HTP FP16 | ASUS SM8750 / V79 | common graphs warmed during load; reference graphs lazy | 21.34 s first request | 193,280 samples |

The QNN row is an engineering measurement, not a release compatibility claim. Its preset path
passed with `cpu_fallback=disabled`; temporary-reference coverage remains a separate acceptance gate.

## Qualcomm QNN / HTP (Engineering Preview)

QNN/HTP is present for converter and hardware-validation work, not as a released Android capability.
The CPU packages above are the supported v3 path. Do not publish compatibility claims for QNN/NPU
until a matching physical device has completed the acceptance procedure below.

Every QNN pipeline or voice attachment must use the `*.qnn.gsvm` suffix. Ordinary CPU packages
continue to use `*.gsvm`. The `/sdcard/models/gs` CPU-model scan ignores QNN attachments; import
them through the dedicated QNN pipeline or voice attachment picker so the app can enforce target
SoC and base-model pairing.

The engineering validation allowlist is intentionally limited to:

| Target | SoC | HTP |
| --- | --- | --- |
| `snapdragon_8_gen_3` | SM8650 | V75 |
| `snapdragon_8_elite` | SM8750 | V79 |
| `snapdragon_8_elite_gen_5` | SM8850 | V81 |

The Android dependency is pinned to QNN Runtime `2.48.0`, matching the downloaded QAIRT SDK
`2.48.0.260626`. QAIRT's documented Python versions for this release include ONNX `1.19.1`, ONNX
Runtime `1.23.2`, and ONNX Simplifier `0.6.2`; use a separate conversion environment rather than
changing the GPT-SoVITS environment.

Compile one already-static ONNX graph for one target:

```bash
/path/to/qairt-venv/bin/python tools/build_qnn_htp_context.py \
  --onnx build/static-component.onnx \
  --output build/qnn/sm8750/component \
  --qairt-sdk /path/to/qairt/2.48.0.260626 \
  --target-soc snapdragon_8_elite
```

`clang++` is required for the host model library. Use repeated `--input-dim NAME=...` arguments to
resolve dynamic ONNX inputs. The tool runs `qnn-onnx-converter --float_bitwidth 16`, builds a host
model library, then creates a target-specific offline context using explicit `soc_model` and
`dsp_arch` backend-extension settings. It inspects the generated context and rejects a mismatched
SoC or HTP architecture before writing exact `target_soc`, `target_soc_model`,
`target_soc_family`, `htp_arch`, `qairt_version` and `backend_artifact` metadata.

FP16 is a floating-point precision conversion, not integer quantization. The generated graph has
FP16 graph I/O and cannot replace the FP32 CPU TorchScript artifact. CPU and HTP are separate
backend artifacts. `build_qnn_htp_context.py` emits individual non-deployable components;
`build_v2pp_qnn_product.py` validates and compiles all required contexts, then assembles the paired
pipeline and voice attachments. Long VITS convolutions are split and partitioned during conversion
so each HTP op fits the selected V75, V79, or V81 target. Partition boundary layouts are preserved explicitly and checked before
packaging. The product builder also detects the QAIRT FP16 static-tensor alignment requirement in
the G2PW descriptor table, creates a hash-bound zero-padded conversion input when needed, and reuses
it across target-specific builds. General QNN conversion is not yet exposed in the conversion web UI.

The arm64 QAIRT/QNN and ONNX Runtime libraries in the APK use 16 KiB-compatible load alignment,
and the APK passes `zipalign -P 16`. The legacy PyTorch Android 2.1 CPU runtime (`libpytorch_jni`,
`libfbjni` and its `libc++_shared`) still uses 4 KiB ELF load alignment. A 16 KiB-page device must
therefore provide Android's page-size compatibility mode until PyTorch is rebuilt or replaced; this
APK must not be described as fully native 16 KiB-page compatible.

The APK packages only the QNN HTP backend and the V75, V79 and V81 target libraries. QNN GPU,
generic DSP and older HTP architecture libraries are intentionally excluded; CPU remains a separate
TorchScript correctness backend rather than a fallback inside a QNN session.

Before publishing a target product, reopen and audit the final packages rather than trusting the
intermediate conversion directories. This example audits SM8750; use `sm8650` or `sm8850` for the
other target-specific outputs:

```bash
../gpt/bin/python tools/audit_v2pp_qnn_product.py \
  --pipeline build/huggingface/v2pp-sm8750-pipeline.qnn.gsvm \
  --model build/huggingface/firefly-v2pp-sm8750.qnn.gsvm \
  --base-model build/huggingface/firefly-v2pp-cpu.gsvm \
  --output build/huggingface/firefly-v2pp-sm8750.audit.json
(cd build/huggingface && sha256sum -c SHA256SUMS)
```

Upload the two CPU `.gsvm` files plus the pipeline and voice `.qnn.gsvm` pair for every published
target, together with `build/huggingface/SHA256SUMS`. The audit JSON files are local release evidence
and are optional; they contain machine-local paths, so they are not application inputs.

The complete product acceptance path installs the same four packages used by the UI and invokes the
same backend API. Copy them under the required model root and launch it explicitly:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell mkdir -p /sdcard/models/gs
adb push build/huggingface/v2pp-cpu-pipeline.gsvm /sdcard/models/gs/
adb push build/huggingface/firefly-v2pp-cpu.gsvm /sdcard/models/gs/
adb push build/huggingface/v2pp-sm8750-pipeline.qnn.gsvm /sdcard/models/gs/
adb push build/huggingface/firefly-v2pp-sm8750.qnn.gsvm /sdcard/models/gs/
adb shell appops set ai.gsv.mobile MANAGE_EXTERNAL_STORAGE allow
adb logcat -c
adb shell am start -n ai.gsv.mobile/.MainActivity \
  --es qnn_product_cpu_pipeline /sdcard/models/gs/v2pp-cpu-pipeline.gsvm \
  --es qnn_product_cpu_model /sdcard/models/gs/firefly-v2pp-cpu.gsvm \
  --es qnn_product_pipeline /sdcard/models/gs/v2pp-sm8750-pipeline.qnn.gsvm \
  --es qnn_product_model /sdcard/models/gs/firefly-v2pp-sm8750.qnn.gsvm
adb logcat -s GSV_QNN_PRODUCT GSV_QNN_TTS GSV_QNN ORT
```

The runner sets `session.disable_cpu_ep_fallback=1` for every neural session and rejects silent or
clipped PCM. It also starts the real loopback `LocalOpenAiServer`, submits an omitted-reference
request and verifies the streamed WAV plus `X-GSV-Backend` response. Without a temporary reference
the runner reports `PASS_PARTIAL_REFERENCE_NOT_RUN`, not complete product coverage. Retrieve its
machine-readable result and preset outputs with:

```bash
adb shell run-as ai.gsv.mobile cat files/qnn-product-acceptance/result.json
adb exec-out run-as ai.gsv.mobile \
  cat files/qnn-product-acceptance/preset.wav \
  > build/qnn-v2pp-sm8750-output.wav
adb exec-out run-as ai.gsv.mobile \
  cat files/qnn-product-acceptance/openai-preset.wav \
  > build/qnn-v2pp-sm8750-openai-output.wav
```

Debug builds emit one ONNX Runtime JSON profile for every executed QNN graph. Pull only those
profiles from the installed pipeline/model roots and reject any CPU provider node:

```bash
profile_root=$(mktemp -d build/qnn-device-profiles.XXXXXX)
adb exec-out run-as ai.gsv.mobile \
  tar -C files/models -cf - qnn-pipelines qnn-models \
  | tar -C "$profile_root" -xf -
mapfile -d '' qnn_profiles < <(
  find "$profile_root" -type f \
    \( -path '*/profiles/*.json' -o -name 'qnn-profile_*.json' \) -print0
)
python3 tools/audit_android_qnn_profiles.py "${qnn_profiles[@]}" \
  --output build/qnn-v2pp-sm8750-profile-audit.json
```

Acceptance requires a `QNN HTP` backend, `cpu_neural_fallback=false`, QNN-only ORT profiles for every
neural graph, non-silent peak/RMS values and an audible inspection of the pulled WAV. Repeat the
runner with `qnn_product_reference`, `qnn_product_reference_text` and
`qnn_product_reference_language` extras to validate both direct-engine and Base64 OpenAI requests
with the five-second temporary reference. The resulting JSON must have
`reference_coverage_complete=true`. The ADB runner deliberately records
`passed=false`, `product_acceptance_complete=false`, `ui_workflow_validated=false` and
`qnn_profile_audit_embedded=false`; `PASS_ENGINE_API_ONLY` means only its direct engine and local API
checks passed. Complete acceptance additionally requires the same preset/reference operations through the visible UI and a successful
`audit_android_qnn_profiles.py` report. A successful APK build or context load alone is not a
complete NPU TTS result.

## Quality Policy

- CPU correctness packages use exact FP32 with no quantization, FP16 conversion, pruning or weight rewriting
- QNN FP16 contexts are separate acceleration artifacts and must be validated against the CPU result
- No reduction of V4 CFM steps for memory or speed
- No operator substitution that changes PCM output
- Memory optimization may change component residency and scheduling only
- CPU inference remains the deployment correctness baseline

Model quality is still determined by the source checkpoints and reference conditioning. Conversion
cannot correct pronunciation or breath artifacts learned by a poorly trained voice model.

## Upstream And Third-Party Work

This project builds on [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS), which is
distributed under the MIT License. GPT-SoVITS architecture, upstream inference code and pretrained
asset handling remain attributable to its authors and contributors.

Model checkpoints, pretrained weights, Android libraries and Python dependencies may have their
own licenses or usage restrictions. Users are responsible for reviewing those terms before
redistributing an APK, converted model or hosted conversion service.

## Star graph :3
<a href="https://www.star-history.com/?repos=tuxKOH%2FGPT-SoViTs-android&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=tuxKOH/GPT-SoViTs-android&type=date&theme=dark&legend=top-left&sealed_token=eYUeD8gLl3qezgYMEj4z-DAot92KN-1IJ522V_u8s2-duF-y5ZwiNvEYncKF2GbAI0VPGd6dM30YYLBsP2FgUCJ1xNBlo61lAFJ4f9nhF_gw6t0J3SrWrw" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=tuxKOH/GPT-SoViTs-android&type=date&legend=top-left&sealed_token=eYUeD8gLl3qezgYMEj4z-DAot92KN-1IJ522V_u8s2-duF-y5ZwiNvEYncKF2GbAI0VPGd6dM30YYLBsP2FgUCJ1xNBlo61lAFJ4f9nhF_gw6t0J3SrWrw" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=tuxKOH/GPT-SoViTs-android&type=date&legend=top-left&sealed_token=eYUeD8gLl3qezgYMEj4z-DAot92KN-1IJ522V_u8s2-duF-y5ZwiNvEYncKF2GbAI0VPGd6dM30YYLBsP2FgUCJ1xNBlo61lAFJ4f9nhF_gw6t0J3SrWrw" />
 </picture>
</a>
## License

GSV Mobile source code in this directory is licensed under the
[GNU General Public License v3.0](LICENSE).

The GPL-3.0 license applies to this project's source code. It does not relicense GPT-SoVITS,
third-party dependencies, pretrained models or user-provided checkpoints.
