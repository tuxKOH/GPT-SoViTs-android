package ai.gsv.mobile

import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import android.util.Log
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtProvider
import org.json.JSONObject
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor

data class SynthesisOptions(
    val temperature: Float = 1.0f,
    val topP: Float = 1.0f,
    val topK: Int = 10,
    val repetitionPenalty: Float = 1.35f,
    val speedFactor: Float = 1.0f,
    val sampleSteps: Int = 32,
) {
    init {
        require(temperature >= 0.0f && temperature <= 2.0f) { "temperature must be between 0 and 2" }
        require(topP >= 0.0f && topP <= 1.0f) { "top_p must be between 0 and 1" }
        require(topK in 1..100) { "top_k must be between 1 and 100" }
        require(repetitionPenalty > 0.0f && repetitionPenalty <= 3.0f) { "repetition penalty must be between 0 and 3" }
        require(speedFactor in 0.25f..4.0f) { "speed must be between 0.25 and 4" }
        require(sampleSteps in 1..128) { "sample steps must be between 1 and 128" }
    }

    fun isDefault() = this == SynthesisOptions()
}

data class SynthesisRequest(
    val text: String,
    val language: String = "auto",
    val seed: Long = -1,
    val options: SynthesisOptions = SynthesisOptions(),
    val reference: ReferenceInput? = null,
)

interface ExecutionSession : Closeable {
    val displayName: String
    fun synthesize(request: SynthesisRequest, output: File): File
}

interface ExecutionBackend {
    fun supports(model: ModelPackage): Boolean
    fun open(model: ModelPackage): ExecutionSession
}

/** Boundary for a complete text-to-PCM QNN executor. */
interface QnnRuntime {
    val available: Boolean
    fun open(model: ModelPackage, target: QualcommTargetSoc): ExecutionSession
}

private object OrtQnnRuntime : QnnRuntime {
    override val available: Boolean
        get() = runCatching { OrtProvider.QNN in OrtEnvironment.getAvailableProviders() }.getOrDefault(false)

    override fun open(model: ModelPackage, target: QualcommTargetSoc): ExecutionSession {
        require(available) { "ONNX Runtime QNN EP is not loaded" }
        val descriptorFile = model.runtimeFile(model.backendArtifact)
        require(descriptorFile.isFile) { "QNN package backend artifact is missing" }
        val descriptor = JSONObject(descriptorFile.readText())
        require(descriptor.optString("format") == "gsv-qnn-executor") {
            "QNN attachment does not expose the high-level executor ABI"
        }
        require(descriptor.optInt("format_version") == 1)
        require(descriptor.optInt("runtime_abi_version") == 1) {
            "QNN executor runtime ABI is unsupported"
        }
        require(descriptor.optString("operation") == "synthesize_utf8_to_pcm16")
        require(descriptor.optBoolean("complete")) { "QNN executor is incomplete" }
        require(descriptor.optBoolean("utf8_text_input")) { "QNN executor does not accept UTF-8 text" }
        require(descriptor.optBoolean("pcm16_output")) { "QNN executor does not return PCM16" }
        require(!descriptor.optBoolean("cpu_neural_fallback", true)) {
            "QNN executor enables CPU neural fallback"
        }
        return when (val engine = descriptor.optString("engine")) {
            "gpt-sovits-v2pp-qnn-buckets" ->
                QnnV2ppRuntime.openArtifactSession(model, target, descriptor)
            else -> error("QNN executor engine is not implemented: $engine")
        }
    }
}

class QnnHtpBackend(private val runtime: QnnRuntime = OrtQnnRuntime) : ExecutionBackend {
    override fun supports(model: ModelPackage): Boolean {
        if (model.executor != "qnn-htp" || !model.deployable || !model.strictCpuFallback) return false
        val status = QualcommTargetPolicy.current()
        if (!status.isProductTarget) return false
        return QualcommTargetPolicy.isCompatible(model, status)
    }

    override fun open(model: ModelPackage): ExecutionSession {
        val status = QualcommTargetPolicy.current()
        require(status.isProductTarget) {
            "this device is not a supported QNN product target (Snapdragon 8 Gen 3, 8 Elite, or 8 Elite Gen 5)"
        }
        if (model.executor == "qnn-htp") {
            require(QualcommTargetPolicy.isCompatible(model, status)) {
                "QNN HTP artifact is unavailable: ${QualcommTargetPolicy.explain(model, status)}"
            }
        }
        return runtime.open(model, requireNotNull(status.target))
    }
}

/**
 * Thin host contract required by AGENTS.md. Topology, conditioning and preprocessing belong to
 * the converted artifact/native runtime, never to this Kotlin layer.
 */
class CpuBackend : ExecutionBackend {
    override fun supports(model: ModelPackage) =
        model.executor in setOf("torchscript-cpu-single", "torchscript-cpu-staged")
    override fun open(model: ModelPackage): ExecutionSession {
        require(model.entrypoint == "synthesize_utf8_to_pcm16") { "unknown entrypoint ${model.entrypoint}" }
        require(model.deployable) { "CPU graphs exist, but the text frontend is not fused and the package is not deployable" }
        val threads = Runtime.getRuntime().availableProcessors().coerceIn(1, 8)
        runCatching {
            if (TorchCpuThreads.current() != threads) TorchCpuThreads.set(threads)
        }.onFailure { Log.w("GSV_CPU_THREADS", "thread configuration unavailable", it) }
        return if(model.executor=="torchscript-cpu-staged") StagedCpuSession(model) else NativeCpuSession(model)
    }
}

/** Keeps only one large FP32 neural stage resident at a time. */
private class StagedCpuSession(private val model: ModelPackage) : ExecutionSession {
    override val displayName = "Android CPU + G2PW (FP32 staged)"
    override fun synthesize(request: SynthesisRequest, output: File): File {
        val trace = TimingTrace("synthesis.cpu.staged", output)
        return try {
            val result = TimingContext.with(trace) {
                TimingContext.measure("request.validate") {
                    require(request.text.isNotBlank()) { "text must not be empty" }
                    requireOptions(model, request.options, request.seed)
                    requireReferenceInput(model, request.reference)
                }
                val prepared = TimingContext.measure("frontend.prepare") {
                    FullTextFrontend(model.runtimeFile("runtime/frontend")).use { frontend ->
                        frontend.prepare(request.text, request.language) to request.reference?.let {
                            frontend.prepare(it.text, it.language)
                        }
                    }
                }
                val features = TimingContext.measure("bert.module_load") {
                    Module.load(model.runtimeFile("runtime/bert.pt").path)
                }.useModule { bert ->
                    fun infer(value: FullTextFrontend.Prepared): Tensor {
                        val features = FloatArray(value.phoneIds.size * 1024)
                        value.bertSpans.forEach { span ->
                            val tokenShape = longArrayOf(1, span.tokenIds.size.toLong())
                            val output = bert.forward(
                                IValue.from(Tensor.fromBlob(span.tokenIds, tokenShape)),
                                IValue.from(Tensor.fromBlob(LongArray(span.tokenIds.size) { 1L }, tokenShape)),
                                IValue.from(Tensor.fromBlob(LongArray(span.tokenIds.size) { 0L }, tokenShape)),
                                IValue.from(Tensor.fromBlob(span.word2ph,longArrayOf(span.word2ph.size.toLong()))),
                            ).toTensor().dataAsFloatArray
                            require(output.size == span.phoneCount * 1024) {
                                "BERT output does not match the frontend phone span"
                            }
                            output.copyInto(features, span.phoneOffset * 1024)
                        }
                        return Tensor.fromBlob(features, longArrayOf(value.phoneIds.size.toLong(), 1024))
                    }
                    TimingContext.measure("bert.inference") { infer(prepared.first) to prepared.second?.let(::infer) }
                }
                if (BuildConfig.DEBUG) TimingContext.mark(
                    "cpu.input_fingerprint",
                    "phones=${prepared.first.phoneIds.contentHashCode()} " +
                        "bert=${features.first.dataAsFloatArray.contentHashCode()} seed=${request.seed}",
                )
                val acoustic = TimingContext.measure("acoustic.module_load") {
                    Module.load(model.runtimeFile("runtime/acoustic.pt").path)
                }
                val acousticResult = acoustic.useModule { module ->
                    TimingContext.measure("acoustic.inference") {
                        val phone = IValue.from(Tensor.fromBlob(prepared.first.phoneIds,longArrayOf(1,prepared.first.phoneIds.size.toLong())))
                        val bert = IValue.from(features.first)
                        val reference = request.reference
                        if (reference != null) {
                            val prompt = requireNotNull(prepared.second)
                            module.runMethod(
                                "synthesize_reference_options",
                                phone, bert,
                                IValue.from(Tensor.fromBlob(reference.pcm16k, longArrayOf(1, reference.pcm16k.size.toLong()))),
                                IValue.from(Tensor.fromBlob(reference.pcm32k, longArrayOf(1, reference.pcm32k.size.toLong()))),
                                IValue.from(Tensor.fromBlob(prompt.phoneIds, longArrayOf(1, prompt.phoneIds.size.toLong()))),
                                IValue.from(requireNotNull(features.second)),
                                IValue.from(request.options.temperature.toDouble()),
                                IValue.from(request.options.topK.toLong()),
                                IValue.from(request.options.topP.toDouble()),
                                IValue.from(request.options.repetitionPenalty.toDouble()),
                                IValue.from(request.options.speedFactor.toDouble()),
                                IValue.from(request.options.sampleSteps.toLong()),
                                IValue.from(request.seed),
                            )
                        } else if (model.runtimeOptionsVersion >= 1) {
                            val common = arrayOf(
                                phone, bert,
                                IValue.from(request.options.temperature.toDouble()),
                                IValue.from(request.options.topK.toLong()),
                                IValue.from(request.options.topP.toDouble()),
                                IValue.from(request.options.repetitionPenalty.toDouble()),
                                IValue.from(request.options.speedFactor.toDouble()),
                            )
                            if ("sample_steps" in model.runtimeOptions) {
                                module.forward(
                                    *common,
                                    IValue.from(request.options.sampleSteps.toLong()),
                                    IValue.from(request.seed),
                                )
                            } else {
                                module.forward(*common, IValue.from(request.seed))
                            }
                        } else module.forward(phone, bert, IValue.from(10L))
                    }
                }
                TimingContext.measure("pcm.convert_validate_wav_write") { writeResult(acousticResult,output) }
            }
            trace.finish(true)
            result
        } catch (error: Throwable) {
            trace.finish(false, error)
            throw error
        }
    }
    override fun close() = Unit
}

private inline fun <T> Module.useModule(block:(Module)->T):T=try{block(this)}finally{destroy()}

private fun writeResult(result:IValue,output:File):File {
    val tuple=result.toTuple();val sampleRate=tuple[0].toLong().toInt();val pcm32=tuple[1].toTensor().dataAsIntArray
    require(pcm32.isNotEmpty()){ "model returned empty PCM" };var peak=0;var clipped=0;var sumSquares=0.0
    pcm32.forEach{value->val sample=value.coerceIn(-32768,32767);val magnitude=kotlin.math.abs(sample);peak=maxOf(peak,magnitude);if(magnitude>=32767)clipped++;sumSquares+=sample.toDouble()*sample}
    val rms=kotlin.math.sqrt(sumSquares/pcm32.size);val clippedRatio=clipped.toDouble()/pcm32.size
    require(peak>100&&rms>20.0){"model returned silent PCM: peak=$peak rms=$rms"}
    require(rms<20000.0&&clippedRatio<0.01){"model returned severely clipped PCM: peak=$peak rms=$rms clippedRatio=$clippedRatio"}
    WavWriter.write(output,pcm32.map{it.coerceIn(-32768,32767).toShort()}.toShortArray(),sampleRate);return output
}

private class NativeCpuSession(
    private val model: ModelPackage,
    private val qnnFrontendTarget: QualcommTargetSoc? = null,
    private val strictQnn: Boolean = false,
) : ExecutionSession {
    override val displayName: String
        get() {
            if (qnnFrontendTarget == null) return "Android CPU${if (frontend != null) " + G2PW" else ""}"
            val stats = frontend?.qnnExecutionStats
            return when {
                stats == null -> "QNN HTP pending (${qnnFrontendTarget.displayName}) + CPU acoustic"
                stats.usedQnnCompute -> "QNN HTP G2PW (${qnnFrontendTarget.displayName}) + CPU acoustic"
                stats.usedQnn -> "QNN HTP frontend utility (${stats.qnnNodes} nodes) + CPU G2PW/acoustic"
                strictQnn -> "QNN HTP failed (${qnnFrontendTarget.displayName})"
                else -> "Android CPU fallback (QNN assigned 0 nodes)"
            }
        }
    private val module = TimingContext.measure("acoustic.module_load") {
        Module.load(model.runtimeFile("runtime/pipeline.pt").path)
    }
    private val frontend = model.runtimeFile("runtime/frontend/g2pW.onnx").parentFile.takeIf { File(it,"g2pW.onnx").isFile }
        ?.let { FullZhFrontend(it, qnnFrontendTarget) }
    override fun synthesize(request: SynthesisRequest, output: File): File {
        val trace = TimingTrace(
            if (qnnFrontendTarget == null) "synthesis.cpu" else "synthesis.qnn_mixed",
            output,
        )
        return try {
            val result = TimingContext.with(trace) {
                TimingContext.measure("request.validate") {
                    require(request.text.isNotBlank()) { "text must not be empty" }
                    requireOptions(model, request.options, request.seed)
                    requireReferenceInput(model, request.reference)
                }
                val result = frontend?.let { frontend ->
                    val prepared = TimingContext.measure("frontend.prepare") { frontend.prepare(request.text) }
                    val inputs = TimingContext.measure("acoustic.input_tensor_build") {
                        arrayOf(
                            IValue.from(Tensor.fromBlob(prepared.phoneIds,longArrayOf(prepared.phoneIds.size.toLong()))),
                            IValue.from(Tensor.fromBlob(prepared.tokenIds,longArrayOf(prepared.tokenIds.size.toLong()))),
                            IValue.from(Tensor.fromBlob(prepared.word2ph,longArrayOf(prepared.word2ph.size.toLong()))),
                            IValue.from(Tensor.fromBlob(prepared.chineseMask,longArrayOf(prepared.chineseMask.size.toLong()))),
                        )
                    }
                    TimingContext.measure("acoustic.inference") {
                        val phone = inputs[0]
                        val tokens = inputs[1]
                        val word2ph = inputs[2]
                        val chinese = inputs[3]
                        val reference = request.reference
                        if (reference != null) {
                            val prompt = TimingContext.measure("frontend.reference_prepare") { frontend.prepare(reference.text) }
                            module.runMethod(
                                "synthesize_reference_preprocessed_options",
                                phone, tokens, word2ph, chinese,
                                IValue.from(Tensor.fromBlob(prompt.phoneIds,longArrayOf(prompt.phoneIds.size.toLong()))),
                                IValue.from(Tensor.fromBlob(prompt.tokenIds,longArrayOf(prompt.tokenIds.size.toLong()))),
                                IValue.from(Tensor.fromBlob(prompt.word2ph,longArrayOf(prompt.word2ph.size.toLong()))),
                                IValue.from(Tensor.fromBlob(prompt.chineseMask,longArrayOf(prompt.chineseMask.size.toLong()))),
                                IValue.from(Tensor.fromBlob(reference.pcm16k,longArrayOf(1,reference.pcm16k.size.toLong()))),
                                IValue.from(Tensor.fromBlob(reference.pcm32k,longArrayOf(1,reference.pcm32k.size.toLong()))),
                                IValue.from(request.seed),
                                IValue.from(request.options.temperature.toDouble()), IValue.from(request.options.topP.toDouble()),
                                IValue.from(request.options.topK.toLong()), IValue.from(request.options.repetitionPenalty.toDouble()),
                                IValue.from(request.options.speedFactor.toDouble()), IValue.from(request.options.sampleSteps.toLong()),
                            )
                        } else if (model.runtimeOptionsVersion >= 1) module.runMethod("synthesize_preprocessed_options",
                            phone, tokens, word2ph, chinese, IValue.from(request.seed),
                            IValue.from(request.options.temperature.toDouble()), IValue.from(request.options.topP.toDouble()),
                            IValue.from(request.options.topK.toLong()), IValue.from(request.options.repetitionPenalty.toDouble()),
                            IValue.from(request.options.speedFactor.toDouble()), IValue.from(request.options.sampleSteps.toLong()))
                        else module.runMethod("synthesize_preprocessed", phone, tokens, word2ph, chinese, IValue.from(request.seed))
                    }
                } ?: run {
                    val utf8 = request.text.toByteArray(Charsets.UTF_8)
                    val input = Tensor.fromBlobUnsigned(utf8, longArrayOf(utf8.size.toLong()))
                    TimingContext.measure("acoustic.inference_raw_utf8") {
                        module.forward(IValue.from(input), IValue.from(request.language), IValue.from(request.seed))
                    }
                }
                TimingContext.measure("qnn.profile_finalize") { frontend?.finalizeQnnProfiling() }
                if (qnnFrontendTarget != null) {
                    val stats = frontend?.qnnExecutionStats
                    TimingContext.mark(
                        "qnn.profile_summary",
                        "providers=${stats?.providers?.joinToString(",") ?: "none"} " +
                            "qnnNodes=${stats?.qnnNodes ?: 0} qnnComputeNodes=${stats?.qnnComputeNodes ?: 0} " +
                            "cpuNodes=${stats?.cpuNodes ?: 0} profileNodeUs=${stats?.profileNodeDurationUs ?: 0}",
                    )
                    Log.i("GSV_QNN", "G2PW backend=${displayName} stats=$stats")
                }
                TimingContext.measure("pcm.convert_validate_wav_write") {
                    val tuple = result.toTuple()
                    val sampleRate = tuple[0].toLong().toInt()
                    // PyTorch Android has no Int16 tensor bridge; pipeline returns PCM16 values in Int32.
                    val pcm32 = tuple[1].toTensor().dataAsIntArray
                    require(pcm32.isNotEmpty()){ "model returned empty PCM" }
                    var peak=0;var clipped=0;var sumSquares=0.0
                    pcm32.forEach { value ->
                        val sample=value.coerceIn(-32768,32767);val magnitude=kotlin.math.abs(sample)
                        peak=maxOf(peak,magnitude);if(magnitude>=32767)clipped++;sumSquares+=sample.toDouble()*sample
                    }
                    val rms=kotlin.math.sqrt(sumSquares/pcm32.size);val clippedRatio=clipped.toDouble()/pcm32.size
                    require(peak>100&&rms>20.0){"model returned silent PCM: peak=$peak rms=$rms"}
                    require(rms<20000.0&&clippedRatio<0.01){"model returned severely clipped PCM: peak=$peak rms=$rms clippedRatio=$clippedRatio"}
                    val pcm = pcm32.map { it.coerceIn(-32768, 32767).toShort() }.toShortArray()
                    WavWriter.write(output, pcm, sampleRate)
                    output
                }
            }
            trace.finish(true)
            result
        } catch (error: Throwable) {
            trace.finish(false, error)
            throw error
        }
    }
    override fun close() { frontend?.close(); module.destroy() }
}

internal fun requireOptions(model: ModelPackage, options: SynthesisOptions, seed: Long = -1L) {
    fun requireOption(name: String, changed: Boolean) {
        if (changed) {
            require(model.runtimeOptionsVersion >= 1 && name in model.runtimeOptions) {
                "this package does not declare runtime option $name; reconvert it before changing that value"
            }
        }
    }
    requireOption("temperature", options.temperature != 1.0f)
    requireOption("top_p", options.topP != 1.0f)
    requireOption("top_k", options.topK != 10)
    requireOption("repetition_penalty", options.repetitionPenalty != 1.35f)
    requireOption("speed_factor", options.speedFactor != 1.0f)
    requireOption("sample_steps", options.sampleSteps != 32)
    requireOption("seed", seed != -1L)
}

internal fun requireReferenceInput(model: ModelPackage, reference: ReferenceInput?) {
    if (reference == null) return
    require(model.referenceInputVersion >= 1) {
        "this model package does not support runtime reference input; reconvert it with the current tool"
    }
    require(reference.text.isNotBlank()) { "reference transcript is required" }
    require(reference.pcm16k.isNotEmpty() && reference.pcm32k.isNotEmpty()) { "reference PCM is empty" }
    requireSupportedLanguage(reference.language, "reference_language")
}

internal fun requireSupportedLanguage(language: String, field: String = "language") {
    require(language.lowercase() in setOf("auto", "zh", "en")) {
        "$field=$language is unsupported; use auto, zh, or en"
    }
}

private object WavWriter {
    fun write(file: File, pcm: ShortArray, sampleRate: Int) {
        RandomAccessFile(file, "rw").use { out ->
            out.setLength(0)
            val dataSize = pcm.size * 2
            out.writeBytes("RIFF"); writeIntLE(out, 36 + dataSize); out.writeBytes("WAVE")
            out.writeBytes("fmt "); writeIntLE(out, 16); writeShortLE(out, 1); writeShortLE(out, 1)
            writeIntLE(out, sampleRate); writeIntLE(out, sampleRate * 2)
            writeShortLE(out, 2); writeShortLE(out, 16); out.writeBytes("data"); writeIntLE(out, dataSize)
            val buffer = ByteArray(64 * 1024)
            var position = 0
            for (sample in pcm) {
                val value = sample.toInt()
                buffer[position++] = value.toByte()
                buffer[position++] = (value ushr 8).toByte()
                if (position == buffer.size) {
                    out.write(buffer)
                    position = 0
                }
            }
            if (position > 0) out.write(buffer, 0, position)
        }
    }
    private fun writeIntLE(out: RandomAccessFile, value: Int) {
        out.write(value); out.write(value ushr 8); out.write(value ushr 16); out.write(value ushr 24)
    }
    private fun writeShortLE(out: RandomAccessFile, value: Int) {
        out.write(value); out.write(value ushr 8)
    }
}

class TtsEngine(private val backends: List<ExecutionBackend>) : Closeable {
    @Volatile
    private var session: ExecutionSession? = null
    @Volatile
    private var loadedModel: ModelPackage? = null
    var lastLoadDiagnostic: String = ""
        private set
    val isLoaded get() = session != null
    val backendName get() = session?.displayName ?: "not loaded"
    val loadedPackage: ModelPackage? get() = loadedModel
    val referenceExactPcm16kSamples: Int?
        get() = loadedModel?.takeIf { it.referenceDurationPolicy == "exact_samples" }
            ?.referencePcm16kSamples?.takeIf { it > 0 }
    fun load(model: ModelPackage) {
        val trace = TimingTrace("engine.load")
        try {
            TimingContext.with(trace) {
                val errors = mutableListOf<String>()
                val candidates = backends.filter { backend ->
                    val supported = TimingContext.measure("backend.supports.${backend.javaClass.simpleName}") {
                        runCatching { backend.supports(model) }
                            .onFailure { error ->
                                errors += "${backend.javaClass.simpleName} supports: " +
                                    (error.message ?: error::class.simpleName)
                            }
                            .getOrDefault(false)
                    }
                    supported
                }
                if (candidates.isEmpty()) {
                    lastLoadDiagnostic = errors.joinToString(" | ")
                    val suffix = if (errors.isEmpty()) "" else "; checks: ${errors.joinToString(" | ")}"
                    error("unsupported executor=${model.executor}$suffix")
                }
                TimingContext.measure("previous_session.close") { session?.close() }
                session = null
                loadedModel = null
                for (backend in candidates) {
                    try {
                        session = TimingContext.measure("backend.open.${backend.javaClass.simpleName}") { backend.open(model) }
                        loadedModel = model
                        lastLoadDiagnostic = if (errors.isEmpty()) "" else errors.joinToString(" | ")
                        TimingContext.mark("backend.selected", session?.displayName)
                        return@with
                    } catch (error: Throwable) {
                        errors += "${backend.javaClass.simpleName}: ${error.message ?: error::class.simpleName}"
                    }
                }
                lastLoadDiagnostic = errors.joinToString(" | ")
                val suffix = if (errors.isEmpty()) "" else "; attempts: ${errors.joinToString(" | ")}"
                error("unsupported executor=${model.executor}$suffix")
            }
            trace.finish(true)
        } catch (error: Throwable) {
            trace.finish(false, error)
            throw error
        }
    }
    fun synthesize(request: SynthesisRequest, output: File): File {
        requireSupportedLanguage(request.language)
        request.reference?.let { requireSupportedLanguage(it.language, "reference_language") }
        return requireNotNull(session) { "no model is loaded" }.synthesize(request, output)
    }
    override fun close() { session?.close(); session=null; loadedModel=null }
}
