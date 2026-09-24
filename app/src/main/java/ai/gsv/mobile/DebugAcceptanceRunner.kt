package ai.gsv.mobile

import android.content.Intent
import android.net.Uri
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File

/** Debug-only ADB hooks kept outside the production UI path. */
internal object DebugAcceptanceRunner {
    private fun resultRoot(activity: ComponentActivity, intent: Intent, extra: String, fallback: String): File {
        val requested = intent.getStringExtra(extra)
        if (requested == null) return File(activity.filesDir, fallback).also { it.mkdirs() }
        val root = File(requested).canonicalFile
        val allowed = listOf("/data/local/tmp", "/sdcard/models/gs").map { File(it).canonicalFile }
        require(allowed.any { root != it && root.toPath().startsWith(it.toPath()) }) {
            "debug result directory must be below /data/local/tmp or /sdcard/models/gs"
        }
        require(root.isDirectory) { "debug result directory does not exist: $root" }
        return root
    }

    fun launch(activity: ComponentActivity, intent: Intent) {
        if (!BuildConfig.DEBUG) return
        intent.getStringExtra("cpu_product_model")?.let { modelPath ->
            Log.i(
                "GSV_CPU_PRODUCT",
                "START model=$modelPath pipeline=${intent.getStringExtra("cpu_product_pipeline")}",
            )
            activity.lifecycleScope.launch(Dispatchers.IO) {
                val resultRoot = resultRoot(
                    activity, intent, "cpu_product_output_root", "cpu-product-acceptance",
                )
                val result = File(resultRoot, "result.txt")
                result.writeText("RUNNING model=$modelPath\n")
                runCatching {
                    val model = if (modelPath == "@installed") {
                        ModelPackage.openInstalledCpuPair(activity)
                    } else {
                        val pipelinePath = requireNotNull(intent.getStringExtra("cpu_product_pipeline"))
                        if (pipelinePath == "@installed") {
                            ModelPackage.importModelWithInstalledPipeline(
                                activity,
                                Uri.fromFile(File(modelPath)),
                            )
                        } else {
                            ModelPackage.importPair(
                                activity,
                                Uri.fromFile(File(pipelinePath)),
                                Uri.fromFile(File(modelPath)),
                            )
                        }
                    }
                    synchronized(GsvRuntime.engineLock) {
                        GsvRuntime.engine.load(model)
                        val requestedThreads = intent.getIntExtra("cpu_product_threads", 0)
                        if (requestedThreads > 0) TorchCpuThreads.set(requestedThreads)
                        Log.i("GSV_CPU_PRODUCT", "torch_threads=${TorchCpuThreads.current()}")
                        val request = SynthesisRequest(
                            text = intent.getStringExtra("cpu_product_text") ?: "你好，这是共用 Pipeline 验收。",
                            language = intent.getStringExtra("cpu_product_language") ?: "auto",
                            seed = 1234,
                        )
                        val repeats = intent.getIntExtra("cpu_product_repeat", 1).coerceIn(1, 5)
                        repeat(repeats) { index ->
                            val wav = File(resultRoot, if (index == 0) "output.wav" else "output-$index.wav")
                            GsvRuntime.engine.synthesize(request, wav)
                            require(wav.isFile && wav.length() > 44) { "CPU synthesis did not produce PCM WAV" }
                            Log.i("GSV_CPU_PRODUCT", "run=${index + 1} bytes=${wav.length()}")
                        }
                    }
                    "PASS model=${model.name} bundle=${model.bundleId} backend=${GsvRuntime.engine.backendName}"
                }.onSuccess {
                    result.writeText(it)
                    Log.i("GSV_CPU_PRODUCT", it)
                }.onFailure {
                    result.writeText("FAIL ${it.stackTraceToString()}")
                    Log.e("GSV_CPU_PRODUCT", "FAIL", it)
                }
            }
        }
        intent.getStringExtra("qnn_epcontext_probe")?.let { modelPath ->
            activity.lifecycleScope.launch(Dispatchers.IO) {
                val result = File(activity.filesDir, "qnn-probe/result.txt")
                runCatching { QnnEpContextProbe.run(File(modelPath), result) }
                    .onSuccess { Log.i("GSV_QNN_PROBE", "PASS $it") }
                    .onFailure {
                        result.parentFile?.mkdirs()
                        result.writeText("FAIL ${it.stackTraceToString()}")
                        Log.e("GSV_QNN_PROBE", "FAIL", it)
                    }
            }
        }
        intent.getStringExtra("qnn_v2pp_acceptance")?.let { packagePath ->
            activity.lifecycleScope.launch(Dispatchers.IO) {
                val resultRoot = File(activity.filesDir, "qnn-v2pp-acceptance").also { it.mkdirs() }
                val result = File(resultRoot, "result.txt")
                val wav = File(resultRoot, "output.wav")
                runCatching { QnnV2ppRuntime.run(File(packagePath), wav, result) }
                    .onSuccess { Log.i("GSV_QNN_TTS", it) }
                    .onFailure {
                        result.writeText("FAIL ${it.stackTraceToString()}")
                        Log.e("GSV_QNN_TTS", "FAIL", it)
                    }
            }
        }
        intent.getStringExtra("qnn_product_model")?.let { qnnModelPath ->
            val keepAlive = intent.getBooleanExtra("qnn_product_keep_alive", false)
            if (keepAlive) {
                activity.startForegroundService(
                    Intent(activity, LocalOpenAiService::class.java)
                        .setAction(LocalOpenAiService.ACTION_START)
                        .putExtra(LocalOpenAiService.EXTRA_PORT, 15888)
                )
            }
            activity.lifecycleScope.launch(Dispatchers.IO) {
                val resultRoot = resultRoot(
                    activity, intent, "qnn_product_output_root", "qnn-product-acceptance",
                )
                val profileRoot = File(resultRoot, "profiles").also { it.mkdirs() }
                DebugQnnProfiles.directory =
                    if (intent.getBooleanExtra("qnn_product_profile", true)) profileRoot else null
                try {
                    runCatching {
                        QnnProductAcceptance.run(
                            activity,
                            QnnProductAcceptance.Inputs(
                                cpuPipeline = File(
                                    requireNotNull(intent.getStringExtra("qnn_product_cpu_pipeline")),
                                ),
                                cpuModel = File(
                                    requireNotNull(intent.getStringExtra("qnn_product_cpu_model")),
                                ),
                                qnnPipeline = File(
                                    requireNotNull(intent.getStringExtra("qnn_product_pipeline")),
                                ),
                                qnnModel = File(qnnModelPath),
                                text = intent.getStringExtra("qnn_product_text")
                                    ?: "重庆银行行长正在重新检查音乐记录，这是正式 QNN 产品验收。",
                                referenceAudio = intent.getStringExtra("qnn_product_reference")?.let(::File),
                                referenceText = intent.getStringExtra("qnn_product_reference_text").orEmpty(),
                                referenceLanguage = intent.getStringExtra("qnn_product_reference_language")
                                    ?: "auto",
                            ),
                            resultRoot,
                        )
                    }.onSuccess { Log.i("GSV_QNN_PRODUCT", it) }
                        .onFailure { Log.e("GSV_QNN_PRODUCT", "FAIL", it) }
                } finally {
                    DebugQnnProfiles.directory = null
                    if (keepAlive) {
                        activity.stopService(Intent(activity, LocalOpenAiService::class.java))
                    }
                }
            }
        }
    }
}
