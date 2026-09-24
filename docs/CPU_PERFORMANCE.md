# CPU 性能基线（工程测试）

测试环境：Android 30 x86_64 模拟器，12 GB RAM，6 个虚拟核心；已发布的 Firefly V2 Pro Plus FP32 CPU 包；文本「你好，这是CPU性能测试。」、seed 1234。模拟器可用于定位瓶颈，不能代替 Snapdragon 真机的速度或功耗数据。

| PyTorch 线程 | 声学图推理 | 完整合成 | 声学段进程 CPU 时间 | 音频校验 |
| ---: | ---: | ---: | ---: | --- |
| 4 | 42.51 s | 63.83 s | 124.56 s | 通过 |
| 6 | 32.51–38.38 s | 49.21–57.06 s | 134.24–155.23 s | 两次通过 |
| 8 | 139.27 s | 181.15 s | 729.89 s | 通过 |

4 和 8 线程各跑一次，6 线程跑两次；数值是诊断样本，不是产品速度承诺。进程 CPU 时间会累计多个工作线程，因此可以大于墙钟时间。6 线程在这台 6 核模拟器上比 4 线程的声学段更快；8 线程虽然增加 CPU 工作量，却大幅延长等待。当前 CPU 后端把线程数设为 `min(可用核心数, 8)`，调试验收入口可用 `cpu_product_threads` 临时指定线程数。

合成阶段的主要开销是声学图、模型装载和文本前端。原版 WAV 逐样本写入约 3.1–3.9 秒；64 KB 分块写入后约 19–57 毫秒。FP32 图、权重、采样参数和 PCM 校验规则没有改变。

在主机上用原始 checkpoint 做的算子诊断显示：GPT 单步生成中 `aten::addmm` 约占 55% 的自 CPU 时间，K/V 拼接的 `aten::cat` 约占 10%；SoVITS 输出 184,320 个样本时，卷积约占 83% 的自 CPU 时间。主机使用的卷积内核与 Android 不同，这些比例只能决定下一步研究方向，不能直接外推成手机上的占比。复现 GPT 单步诊断：

```bash
../gpt/bin/python tools/profile_v2pp_t2s.py \
  --checkpoint ../GPT_weights_v2ProPlus/firefly.ckpt \
  --upstream .. --threads 4 --steps 100 \
  --output build/cpu-profile/t2s.json
```

调试 APK 的模拟器验收命令：

```bash
adb shell am start -n ai.gsv.mobile/.MainActivity \
  --es cpu_product_pipeline @installed \
  --es cpu_product_model @installed \
  --es cpu_product_text '你好，这是CPU性能测试。' \
  --ei cpu_product_threads 6 \
  --ei cpu_product_repeat 2
adb logcat -d -s GSV_CPU_PRODUCT GSV_TIMING GSV_CPU_THREADS
```

每次合成的 JSON 计时文件位于应用私有目录 `files/cpu-product-acceptance/`。`process_cpu_ms` 表示进程内所有线程累计用时。调试版还记录 phone/BERT 输入指纹，便于排查重复请求。

本轮曾试验让前端和神经模块跨请求驻留，后续请求的前端准备显著缩短，但出现过全幅削波，因此该驻留改动没有保留。v3.1 原版 APK 在相同 seed 下的两次 WAV 哈希也不同；字节级重复性不能用作这套 Android CPU 路径的唯一正确性判据。CPU 质量基线仍要求非静音、非削波和可听检查。设置页的速度百分比应在 CPU 与真机 NPU 的最高可用速度都测定之后实现，百分比相对于各后端最快稳定配置，而不是设备占用率。

## Snapdragon 8 Elite 真机补测

2026-09-24 在 ASUS AI2501C（SM8750，Android arm64）上使用已安装的 Firefly V2PP FP32 CPU 包，以同一句「你好，这是性能测试。」进行单次合成。测试音频和 sidecar 均写入 `/sdcard/models/gs/gs-acceptance`，没有在 `/sdcard` 根目录放置文件。

| 算子线程 | 完整合成 | 声学推理 | BERT/声学模块装载合计 | WAV 写出 | 结果 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 40.92 s | 22.10 s | 15.21 s | 0.17 s | 通过 |
| 6 | 42.67 s | 22.67 s | 15.63 s | 0.16 s | 通过 |
| 8（默认） | 33.32 s | 16.68 s | 12.83 s | 0.095 s | 通过 |

这只是每档一次的快速对比，受温度、系统负载和模型装载缓存影响，不是稳定速度结论。与 6 核模拟器不同，这台 8 Elite 的 8 线程在本组样本里最快。主要瓶颈仍是声学图和冷装载；“让占用接近 100%”不能直接作为优化目标。
