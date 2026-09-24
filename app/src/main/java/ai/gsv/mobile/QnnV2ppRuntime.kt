package ai.gsv.mobile

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.DoubleBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.nio.ShortBuffer
import java.security.MessageDigest
import java.util.ArrayDeque
import java.util.Random
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt

/** Production V2PP QNN executor plus the retained ADB-only static acceptance entrypoint. */
object QnnV2ppRuntime {
    private const val tag = "GSV_QNN_TTS"
    private const val acceptanceEos = 1024
    private const val acceptanceCacheCapacity = 512
    private const val acceptanceLayers = 24
    private const val acceptanceHidden = 512
    private const val minimumSemanticIterations = 11
    private const val maximumQualityAttempts = 3

    private data class TensorContract(
        val name: String,
        val logicalName: String,
        val type: OnnxJavaType,
        val shape: LongArray,
    )

    private data class GraphStage(
        val name: String,
        val path: String,
        val inputs: List<TensorContract>,
        val outputs: List<TensorContract>,
    )

    internal fun openArtifactSession(
        model: ModelPackage,
        target: QualcommTargetSoc,
        descriptor: JSONObject,
    ): ExecutionSession = ArtifactSession(model, target, descriptor)

    private class ArtifactSession(
        private val model: ModelPackage,
        target: QualcommTargetSoc,
        descriptor: JSONObject,
    ) : ExecutionSession {
        private val frontendConfig = descriptor.getJSONObject("frontend")
        private val graphs = descriptor.getJSONObject("graphs")
        private val vitsStages = graphStages(graphs.getJSONArray("vits"), "vits")
        private val referenceVitsStages = if (descriptor.getInt("reference_input_version") >= 1) {
            graphStages(graphs.getJSONArray("vits_reference"), "vits_reference")
        } else {
            emptyList()
        }
        private val shortCapacity = descriptor.getJSONObject("shapes").optInt("short_semantic_capacity", 0)
        private val shortVitsStages = graphs.optJSONArray("vits_short")?.let {
            graphStages(it, "vits_short")
        }
        private val shortReferenceVitsStages = graphs.optJSONArray("vits_reference_short")?.let {
            graphStages(it, "vits_reference_short")
        }
        private val shapes = descriptor.getJSONObject("shapes")
        private val preset = descriptor.getJSONObject("preset")
        private val referenceConfig = descriptor.optJSONObject("reference")
        private val tokenCapacity = shapes.getInt("token_capacity")
        private val phoneCapacity = shapes.getInt("phone_capacity")
        private val semanticCapacity = if (shapes.has("semantic_capacity")) {
            shapes.getInt("semantic_capacity")
        } else {
            shapes.getInt("semantic_length")
        }
        private val compactLength = shapes.getInt("prefill_cache_length")
        private val presetPromptPhoneLength = shapes.getInt("preset_prompt_phone_length")
        private val cacheCapacity = shapes.getInt("cache_capacity")
        private val layers = shapes.getInt("layers")
        private val hidden = shapes.getInt("hidden_size")
        private val samplesPerSemantic = shapes.optInt("samples_per_semantic", 1280)
        private val eosToken = shapes.optInt("eos_token", acceptanceEos)
        private val paddedInputs = shapes.optBoolean("padding_mask_inputs", false)
        private val maxTextCodePoints = descriptor.optInt("max_text_codepoints", 4000)
        private val interSegmentSilenceMs = descriptor.optInt("inter_segment_silence_ms", 150)
        private val promptSemantic = preset.intArray("prompt_semantic")
        private val promptLength = promptSemantic.size
        private val sampleRate = descriptor.getInt("sample_rate")
        private val runtimeOptionsVersion = descriptor.getInt("runtime_options_version")
        private val referenceInputVersion = descriptor.getInt("reference_input_version")
        private val environment = OrtEnvironment.getEnvironment()
        private val frontend: FullTextFrontend
        // VITS contexts are expensive to open. Keep only the active preset OR reference set,
        // scoped to this loaded artifact; switching models closes every retained context.
        private val vitsSessions = GraphSessionCache(environment, model)

        private data class RuntimeReference(
            val promptPhoneIds: IntArray,
            val promptBert: ShortArray,
            val promptSemantic: IntArray,
            val referenceSpectrogram: ShortArray,
            val speakerEmbedding: ShortArray,
            val compactLength: Int,
        )

        init {
            require(descriptor.getString("engine") == "gpt-sovits-v2pp-qnn-buckets")
            require(descriptor.getInt("engine_version") == 2)
            require(runtimeOptionsVersion in 0..1)
            require(referenceInputVersion in 0..1)
            require(runtimeOptionsVersion == model.runtimeOptionsVersion) {
                "QNN executor and package runtime option ABIs do not match"
            }
            require(referenceInputVersion == model.referenceInputVersion) {
                "QNN executor and package reference input ABIs do not match"
            }
            require(cacheCapacity > compactLength)
            require(layers > 0 && hidden > 0)
            require(tokenCapacity in 3..512)
            require(phoneCapacity > 0 && semanticCapacity > 0 && samplesPerSemantic > 0 && eosToken > 0)
            require(shortCapacity == 0 || shortCapacity in 1 until semanticCapacity)
            require((shortCapacity > 0) == (shortVitsStages != null))
            require(shortReferenceVitsStages == null || shortCapacity > 0)
            require(maxTextCodePoints > 0 && interSegmentSilenceMs in 0..2000)
            require(compactLength + semanticCapacity <= cacheCapacity) {
                "QNN T2S cache capacity is too small for the prepared semantic length"
            }
            require(presetPromptPhoneLength + phoneCapacity + promptLength == compactLength) {
                "QNN preset prefill layout does not match its declared cache length"
            }
            require(sampleRate == model.sampleRate)
            val requiredGraphs = mutableListOf("bert", "t2s_prefill", "t2s_step")
            if (referenceInputVersion >= 1) {
                val config = requireNotNull(referenceConfig) {
                    "QNN reference_input_version=1 package is missing reference capacities"
                }
                require(config.getInt("pcm_16k_samples") > 0)
                require(config.getInt("pcm_32k_samples") > 0)
                require(config.getInt("spectrogram_reflect_pad") > 0)
                require(config.getInt("prompt_semantic_length") > 0)
                require(config.getInt("prompt_phone_capacity") == phoneCapacity)
                require(config.getInt("reference_spectrogram_bins") > 0)
                require(config.getInt("reference_spectrogram_frames") > 0)
                require(config.getInt("speaker_embedding_size") > 0)
                require(config.getInt("prefill_cache_length") + semanticCapacity <= cacheCapacity)
                requiredGraphs += listOf(
                    "reference_ssl",
                    "reference_prompt_semantic",
                    "reference_conditioning",
                    "t2s_reference_prefill",
                )
            }
            requiredGraphs.forEach { name ->
                require(model.runtimeFile(graphs.getString(name)).isFile) { "missing QNN graph $name" }
            }
            (vitsStages + referenceVitsStages + shortVitsStages.orEmpty() +
                shortReferenceVitsStages.orEmpty()).forEach { stage ->
                require(model.runtimeFile(stage.path).isFile) {
                    "missing QNN graph partition ${stage.name}"
                }
            }
            val frontendRoot = model.runtimeFile(frontendConfig.getString("root"))
            val g2pw = model.runtimeFile(frontendConfig.getString("g2pw_model"))
            frontend = FullTextFrontend(
                frontendRoot,
                qnnTarget = target,
                modelFile = g2pw,
                strictQnn = true,
                staticRows = 1,
                staticSequenceLength = frontendConfig.getInt("g2pw_sequence_length"),
                htpFp16Precision = true,
                htpGraphOptimizationMode = "3",
            )
            // Pay the common QNN/HTP context creation cost while the model is being loaded.
            // Reference and short-form graphs stay lazy: opening every variant here can hold
            // several HTP contexts at once, needlessly increasing cold-start memory and making
            // a normal preset request look hung. GraphSessionCache still retains those graphs
            // after their first use, so a temporary reference or short request pays the cost
            // only once.
            listOf("bert", "t2s_prefill", "t2s_step").forEach { name ->
                vitsSessions.warmPath(model.runtimeFile(graphs.getString(name)), name)
            }
            vitsStages.forEach { vitsSessions.warm(it) }
        }

        override val displayName = "QNN HTP (${target.displayName})"

        override fun synthesize(request: SynthesisRequest, output: File): File {
            val trace = TimingTrace("synthesis.qnn", output)
            return try {
                TimingContext.with(trace) { synthesizeInternal(request, output) }
                    .also { trace.finish() }
            } catch (error: Throwable) {
                trace.finish(success = false, error = error)
                throw error
            }
        }

        private fun synthesizeInternal(request: SynthesisRequest, output: File): File {
            require(request.text.isNotBlank()) { "text must not be empty" }
            requireOptions(model, request.options, request.seed)
            requireReferenceInput(model, request.reference)
            require(request.options.speedFactor == 1.0f) {
                "this QNN V2PP executor does not yet expose speed control"
            }
            require(request.text.codePointCount(0, request.text.length) <= maxTextCodePoints) {
                "text exceeds the QNN limit of $maxTextCodePoints Unicode code points"
            }
            val preparedSegments = TimingContext.measure("qnn.frontend.prepare") {
                prepareSegments(request.text, request.language)
            }
            val runtimeReference by lazy {
                request.reference?.let { reference ->
                    TimingContext.measure("qnn.reference.prepare") {
                        prepareRuntimeReference(reference)
                    }
                }
            }
            // Every request is synthesized afresh.  Completed PCM must never be reused:
            // stochastic sampling, reference changes, and user retries all need a new result.
            val pcmSegments = preparedSegments.mapIndexed { index, prepared ->
                synthesizePrepared(prepared, request, index, runtimeReference)
            }
            val silence = ShortArray(sampleRate * interSegmentSilenceMs / 1000)
            val totalSamples = pcmSegments.sumOf { it.size.toLong() } +
                silence.size.toLong() * (pcmSegments.size - 1).coerceAtLeast(0)
            require(totalSamples in 1..Int.MAX_VALUE.toLong()) { "synthesized PCM is too large" }
            val pcm = ShortArray(totalSamples.toInt())
            var offset = 0
            pcmSegments.forEachIndexed { index, segment ->
                segment.copyInto(pcm, offset)
                offset += segment.size
                if (index != pcmSegments.lastIndex) offset += silence.size
            }
            require(pcm.any { kotlin.math.abs(it.toInt()) > 100 }) {
                "QNN executor returned silent PCM"
            }
            writeWav(output, pcm, sampleRate)
            return output
        }

        private fun synthesizePrepared(
            prepared: FullTextFrontend.Prepared,
            request: SynthesisRequest,
            segmentIndex: Int,
            runtimeReference: RuntimeReference?,
        ): ShortArray {
            require(prepared.maxTokenCount <= tokenCapacity) {
                "text needs ${prepared.maxTokenCount} BERT tokens in one language span, QNN capacity is $tokenCapacity"
            }
            require(prepared.phoneIds.size <= phoneCapacity) {
                "text needs ${prepared.phoneIds.size} phones, QNN capacity is $phoneCapacity"
            }
            val phoneIds = IntArray(prepared.phoneIds.size) { prepared.phoneIds[it].toInt() }
            val bert = runBertFeatures(prepared, vitsSessions)
            val prefill = if (runtimeReference == null) {
                runPrefill(
                    environment,
                    model.runtimeFile(graphs.getString("t2s_prefill")),
                    phoneIds,
                    bert,
                    phoneCapacity,
                    paddedInputs,
                    vitsSessions,
                )
            } else {
                runReferencePrefill(
                    environment,
                    model.runtimeFile(graphs.getString("t2s_reference_prefill")),
                    phoneIds,
                    bert,
                    runtimeReference.promptSemantic,
                    runtimeReference.promptPhoneIds,
                    runtimeReference.promptBert,
                    phoneCapacity,
                    vitsSessions,
                )
            }
            val activePromptSemantic = runtimeReference?.promptSemantic ?: promptSemantic
            val activeCompactLength = runtimeReference?.compactLength ?: compactLength
            val initialCacheValid = if (runtimeReference == null) {
                presetCacheValid(phoneIds.size)
            } else {
                runtimeReferenceCacheValid(runtimeReference.promptPhoneIds.size, phoneIds.size)
            }
            val attempts = if (request.seed >= 0) 1 else maximumQualityAttempts
            repeat(attempts) { attempt ->
                val attemptSeed = segmentSeed(request.seed, segmentIndex)
                val semantics = runSteps(
                    environment,
                    model.runtimeFile(graphs.getString("t2s_step")),
                    prefill,
                    activePromptSemantic,
                    activePromptSemantic.size,
                    activeCompactLength,
                    semanticCapacity,
                    request.options,
                    attemptSeed,
                    cacheCapacity,
                    layers,
                    hidden,
                    eosToken,
                    initialCacheValid,
                    vitsSessions,
                )
                val useShort = shortCapacity > 0 && semantics.validLength <= shortCapacity &&
                    (runtimeReference == null || shortReferenceVitsStages != null)
                val selectedCapacity = if (useShort) shortCapacity else semanticCapacity
                val selectedSemantics = if (useShort) {
                    semantics.copy(values = semantics.values.copyOf(shortCapacity))
                } else semantics
                TimingContext.mark("qnn.vits.capacity", selectedCapacity.toString())
                val audio = if (runtimeReference == null) {
                    runVits(
                        environment, model, if (useShort) requireNotNull(shortVitsStages) else vitsStages,
                        phoneIds, selectedSemantics, attemptSeed, phoneCapacity,
                        selectedCapacity, samplesPerSemantic, paddedInputs, vitsSessions,
                    )
                } else {
                    runReferenceVits(
                        environment, model,
                        if (useShort) requireNotNull(shortReferenceVitsStages) else referenceVitsStages,
                        phoneIds, selectedSemantics,
                        runtimeReference.referenceSpectrogram, runtimeReference.speakerEmbedding,
                        attemptSeed, phoneCapacity, selectedCapacity, samplesPerSemantic, vitsSessions,
                    )
                }
                val pcm = ShortArray(audio.size) { index ->
                    (audio[index].coerceIn(-1.0f, 1.0f) * 32767.0f)
                        .toInt().coerceIn(-32768, 32767).toShort()
                }
                if (isUsablePcm(pcm)) return pcm
                Log.w(tag, "QNN attempt ${attempt + 1}/$attempts produced low-amplitude PCM; resampling")
            }
            error("QNN executor produced low-amplitude PCM after $attempts sampling attempt(s)")
        }

        private fun isUsablePcm(pcm: ShortArray): Boolean {
            if (pcm.isEmpty()) return false
            var peak = 0
            var sumSquares = 0.0
            pcm.forEach { sample ->
                val value = sample.toInt()
                peak = maxOf(peak, kotlin.math.abs(value))
                sumSquares += value.toDouble() * value
            }
            return peak > 100 && sqrt(sumSquares / pcm.size) > 20.0
        }

        private fun prepareRuntimeReference(reference: ReferenceInput): RuntimeReference {
            val config = requireNotNull(referenceConfig)
            val prompt = frontend.prepare(reference.text, reference.language)
            require(prompt.maxTokenCount <= tokenCapacity) {
                "reference transcript needs ${prompt.maxTokenCount} BERT tokens in one language span, QNN capacity is $tokenCapacity"
            }
            val promptPhoneCapacity = config.getInt("prompt_phone_capacity")
            require(prompt.phoneIds.size <= promptPhoneCapacity) {
                "reference transcript needs ${prompt.phoneIds.size} phones, QNN capacity is $promptPhoneCapacity"
            }
            val pcm16Capacity = config.getInt("pcm_16k_samples")
            val pcm32Capacity = config.getInt("pcm_32k_samples")
            require(config.getString("duration_policy") == "exact_samples") {
                "unsupported QNN reference duration policy"
            }
            require(reference.pcm16k.size == pcm16Capacity && reference.pcm32k.size == pcm32Capacity) {
                "QNN temporary reference audio must be exactly " +
                    "${pcm16Capacity / 16000.0} seconds; received " +
                    "${reference.pcm16k.size / 16000.0} seconds"
            }
            val pcm16 = reference.pcm16k
            val pcm32 = reference.pcm32k
            val ssl = runReferenceSsl(
                environment,
                model.runtimeFile(graphs.getString("reference_ssl")),
                pcm16,
                vitsSessions,
            )
            val runtimePromptSemantic = runPromptSemantic(
                environment,
                model.runtimeFile(graphs.getString("reference_prompt_semantic")),
                ssl,
                vitsSessions,
            )
            require(runtimePromptSemantic.size == config.getInt("prompt_semantic_length")) {
                "QNN reference prompt semantic output has an unexpected length"
            }
            val conditioning = runReferenceConditioning(
                environment,
                model.runtimeFile(graphs.getString("reference_conditioning")),
                pcm16,
                reflectPad(pcm32, config.getInt("spectrogram_reflect_pad")),
                vitsSessions,
            )
            require(
                conditioning.spectrogram.size ==
                    config.getInt("reference_spectrogram_bins") *
                    config.getInt("reference_spectrogram_frames")
            ) { "QNN reference spectrogram output has an unexpected shape" }
            require(conditioning.speakerEmbedding.size == config.getInt("speaker_embedding_size")) {
                "QNN reference speaker embedding output has an unexpected shape"
            }
            val promptBert = runBertFeatures(prompt, vitsSessions)
            return RuntimeReference(
                promptPhoneIds = IntArray(prompt.phoneIds.size) { prompt.phoneIds[it].toInt() },
                promptBert = promptBert,
                promptSemantic = runtimePromptSemantic,
                referenceSpectrogram = conditioning.spectrogram,
                speakerEmbedding = conditioning.speakerEmbedding,
                compactLength = config.getInt("prefill_cache_length"),
            )
        }

        private fun presetCacheValid(targetPhoneLength: Int): BooleanArray =
            BooleanArray(compactLength).also { valid ->
                repeat(presetPromptPhoneLength) { valid[it] = true }
                repeat(targetPhoneLength) { valid[presetPromptPhoneLength + it] = true }
                val semanticStart = presetPromptPhoneLength + phoneCapacity
                repeat(promptLength) { valid[semanticStart + it] = true }
            }

        private fun runtimeReferenceCacheValid(
            promptPhoneLength: Int,
            targetPhoneLength: Int,
        ): BooleanArray {
            val config = requireNotNull(referenceConfig)
            val promptCapacity = config.getInt("prompt_phone_capacity")
            val referenceCompactLength = config.getInt("prefill_cache_length")
            return BooleanArray(referenceCompactLength).also { valid ->
                repeat(promptPhoneLength) { valid[it] = true }
                repeat(targetPhoneLength) { valid[promptCapacity + it] = true }
                val semanticStart = promptCapacity + phoneCapacity
                repeat(config.getInt("prompt_semantic_length")) { valid[semanticStart + it] = true }
            }
        }

        private fun runBertFeatures(
            prepared: FullTextFrontend.Prepared,
            sessions: GraphSessionCache? = null,
        ): ShortArray {
            val output = ShortArray(prepared.phoneIds.size * 1024)
            prepared.bertSpans.forEach { span ->
                val tokenIds = IntArray(span.tokenIds.size) { span.tokenIds[it].toInt() }
                val features = runBert(
                    environment,
                    model.runtimeFile(graphs.getString("bert")),
                    tokenIds,
                    span.word2ph,
                    tokenCapacity,
                    sessions = sessions,
                )
                require(features.size == span.phoneCount * 1024)
                features.copyInto(output, span.phoneOffset * 1024)
            }
            return output
        }

        private fun prepareSegments(text: String, language: String): List<FullTextFrontend.Prepared> {
            val queue = ArrayDeque<String>()
            roughSplit(text).forEach(queue::addLast)
            val output = ArrayList<FullTextFrontend.Prepared>()
            while (queue.isNotEmpty()) {
                val value = queue.removeFirst().trim()
                if (value.isEmpty()) continue
                val prepared = frontend.prepare(value, language)
                if (prepared.maxTokenCount <= tokenCapacity && prepared.phoneIds.size <= phoneCapacity) {
                    output += prepared
                    continue
                }
                val pieces = splitOversized(value)
                require(pieces.size == 2) {
                    "one text symbol exceeds the QNN token or phone capacity"
                }
                queue.addFirst(pieces[1])
                queue.addFirst(pieces[0])
            }
            require(output.isNotEmpty()) { "text has no supported symbols" }
            return output
        }

        private fun roughSplit(text: String): List<String> {
            val codePoints = text.codePoints().toArray()
            val limit = (tokenCapacity - 2).coerceAtLeast(1)
            val output = ArrayList<String>()
            var start = 0
            while (start < codePoints.size) {
                val hardEnd = (start + limit).coerceAtMost(codePoints.size)
                val end = if (hardEnd == codePoints.size) {
                    hardEnd
                } else {
                    findBoundary(codePoints, start, hardEnd, start + limit / 2)
                }
                output += String(codePoints, start, end - start)
                start = end
            }
            return output
        }

        private fun splitOversized(text: String): List<String> {
            val codePoints = text.codePoints().toArray()
            if (codePoints.size < 2) return listOf(text)
            val middle = codePoints.size / 2
            var split = middle
            for (index in middle downTo 1) {
                if (isSplitBoundary(codePoints[index - 1])) {
                    split = index
                    break
                }
            }
            return listOf(
                String(codePoints, 0, split),
                String(codePoints, split, codePoints.size - split),
            )
        }

        private fun findBoundary(
            codePoints: IntArray,
            start: Int,
            end: Int,
            minimum: Int,
        ): Int {
            for (index in end - 1 downTo minimum.coerceAtLeast(start)) {
                if (isSplitBoundary(codePoints[index])) return index + 1
            }
            return end
        }

        private fun isSplitBoundary(codePoint: Int): Boolean =
            Character.isWhitespace(codePoint) || codePoint in intArrayOf(
                '。'.code, '！'.code, '？'.code, '；'.code, '，'.code,
                '.'.code, '!'.code, '?'.code, ';'.code, ','.code,
            )

        private fun segmentSeed(seed: Long, segmentIndex: Int): Long =
            if (seed >= 0) seed + segmentIndex else System.nanoTime()

        override fun close() {
            vitsSessions.close()
            frontend.close()
        }
    }

    fun run(root: File, output: File, resultFile: File): String {
        verifyStaticPackage(root)
        val device = QualcommTargetPolicy.current()
        require(device.target == QualcommTargetSoc.SNAPDRAGON_8_ELITE) {
            "static acceptance requires SM8750, observed ${device.observed.joinToString()}"
        }
        val configFile = File(root, "executor.json")
        require(configFile.isFile) { "missing static executor manifest: $configFile" }
        val config = JSONObject(configFile.readText())
        require(config.getString("format") == "gsv-qnn-v2pp-static-acceptance")
        require(config.getString("target_soc") == "snapdragon_8_elite")
        require(config.getString("target_soc_family") == "qualcomm_snapdragon_8")
        require(config.getString("target_asic") == "SM8750")
        require(config.getInt("target_soc_model") == 69)
        require(config.getString("htp_arch") == "V79")
        require(config.getString("qairt_version") == "2.48.0.260626")
        require(config.getString("qnn_runtime_version") == "2.48.0")
        require(config.getString("precision") == "fp16")
        require(!config.getBoolean("cpu_neural_fallback"))
        val tokenIds = config.intArray("token_ids")
        val phoneIds = config.intArray("phone_ids")
        val word2ph = config.intArray("word2ph")
        val promptSemantic = config.intArray("prompt_semantic")
        val semanticLength = config.getInt("semantic_length")
        val compactLength = config.getInt("prefill_cache_length")
        val promptLength = config.getInt("prompt_semantic_length")
        val sampleRate = config.getInt("sample_rate")
        require(tokenIds.size == 6 && phoneIds.size == 8 && word2ph.contentEquals(intArrayOf(2, 2, 2, 2)))
        require(semanticLength == 69 && compactLength == 300 && promptLength == promptSemantic.size)

        val environment = OrtEnvironment.getEnvironment()
        val timings = linkedMapOf<String, Long>()
        val bert = timed("bert", timings) {
            runBert(environment, File(root, "bert_tokens_6.onnx"), tokenIds, word2ph)
        }
        val prefill = timed("t2s_prefill", timings) {
            runPrefill(environment, File(root, "t2s_prefill_p8.onnx"), phoneIds, bert)
        }
        val semantics = timed("t2s_steps", timings) {
            runSteps(
                environment,
                File(root, "t2s_step_c512.onnx"),
                prefill,
                promptSemantic,
                promptLength,
                compactLength,
                semanticLength,
            )
        }
        val audio = timed("vits", timings) {
            runAcceptanceVits(
                environment,
                File(root, "vits_p8_s69.onnx"),
                phoneIds,
                semantics,
                config.optLong("noise_seed", 1234L),
            )
        }
        val pcm = ShortArray(audio.size)
        var peak = 0.0f
        var sumSquares = 0.0
        for (index in audio.indices) {
            val value = audio[index].coerceIn(-1.0f, 1.0f)
            peak = maxOf(peak, kotlin.math.abs(value))
            sumSquares += value.toDouble() * value
            pcm[index] = (value * 32767.0f).toInt().coerceIn(-32768, 32767).toShort()
        }
        val rms = sqrt(sumSquares / audio.size)
        require(peak > 0.005f && rms > 0.001) { "QNN VITS returned silent audio: peak=$peak rms=$rms" }
        writeWav(output, pcm, sampleRate)
        val summary = buildString {
            append("PASS target=SM8750 text=").append(config.getString("text"))
            append(" neural_backend=QNN_HTP cpu_fallback=disabled")
            append(" samples=").append(audio.size)
            append(" sample_rate=").append(sampleRate)
            append(" peak=").append(peak)
            append(" rms=").append(rms)
            append(" semantics=").append(
                semantics.values.take(semantics.validLength).joinToString(",")
            )
            append(" timings_ms=").append(timings.entries.joinToString(",") { "${it.key}:${it.value}" })
            append(" wav=").append(output.path)
        }
        resultFile.parentFile?.mkdirs()
        resultFile.writeText(summary)
        Log.i(tag, summary)
        return summary
    }

    private fun verifyStaticPackage(root: File) {
        val manifestFile = File(root, "manifest.json")
        require(manifestFile.isFile) { "missing static package manifest: $manifestFile" }
        val manifest = JSONObject(manifestFile.readText())
        require(manifest.getString("format") == "gsv-qnn-static-acceptance-directory")
        require(manifest.getBoolean("acceptance_only"))
        require(manifest.getString("executor") == "qnn-htp")
        require(manifest.getString("target_soc") == "snapdragon_8_elite")
        require(manifest.getString("target_soc_family") == "qualcomm_snapdragon_8")
        require(manifest.getString("target_asic") == "SM8750")
        require(manifest.getInt("target_soc_model") == 69)
        require(manifest.getString("htp_arch") == "V79")
        require(manifest.getString("qairt_version") == "2.48.0.260626")
        require(manifest.getString("precision") == "fp16")
        require(manifest.getString("quantization") == "none")
        require(!manifest.getBoolean("cpu_neural_fallback"))
        require(manifest.getString("backend_artifact") == "executor.json")
        val files = manifest.getJSONArray("files")
        repeat(files.length()) { index ->
            val entry = files.getJSONObject(index)
            val file = File(root, entry.getString("path"))
            require(file.canonicalPath.startsWith(root.canonicalPath + File.separator)) {
                "invalid static package path: ${entry.getString("path")}"
            }
            require(file.isFile && file.length() == entry.getLong("size")) {
                "missing or truncated static package file: ${entry.getString("path")}"
            }
            require(sha256(file) == entry.getString("sha256")) {
                "static package hash mismatch: ${entry.getString("path")}"
            }
        }
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun runBert(
        environment: OrtEnvironment,
        model: File,
        tokenIds: IntArray,
        word2ph: IntArray,
        tokenCapacity: Int = 6,
        sessions: GraphSessionCache? = null,
    ): ShortArray = withSession(environment, model, "bert", sessions = sessions) { session ->
        require(tokenIds.size <= tokenCapacity)
        val padded = IntArray(tokenCapacity).also { tokenIds.copyInto(it) }
        val attention = IntArray(tokenCapacity) { if (it < tokenIds.size) 1 else 0 }
        int32(environment, padded, session.inputShape("input_ids", padded.size)).use { ids ->
            int32(
                environment,
                IntArray(tokenCapacity),
                session.inputShape("token_type_ids", tokenCapacity),
            ).use { types ->
                int32(
                    environment,
                    attention,
                    session.inputShape("attention_mask", attention.size),
                ).use { mask ->
                    session.run(mapOf("input_ids" to ids, "token_type_ids" to types, "attention_mask" to mask)).use { outputs ->
                        val hiddenFeatures = outputs.half(0)
                        require(hiddenFeatures.size == tokenCapacity * 1024)
                        Log.i(tag, "bert output ${halfStats(hiddenFeatures)}")
                        ShortArray(word2ph.sum() * 1024).also { expanded ->
                            var phone = 0
                            word2ph.forEachIndexed { token, count ->
                                repeat(count) {
                                    hiddenFeatures.copyInto(expanded, phone * 1024, (token + 1) * 1024, (token + 2) * 1024)
                                    phone++
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private data class Prefill(val logits: FloatArray, val keys: ShortArray, val values: ShortArray)

    private data class ReferenceConditioning(
        val spectrogram: ShortArray,
        val speakerEmbedding: ShortArray,
    )

    private fun runReferenceSsl(
        environment: OrtEnvironment,
        model: File,
        pcm16: FloatArray,
        sessions: GraphSessionCache? = null,
    ): ShortArray = withSession(environment, model, "reference_ssl", sessions = sessions) { session ->
        val values = ShortArray(pcm16.size) { floatToHalf(pcm16[it]) }
        half(environment, values, session.inputShape("reference_pcm_16k", values.size)).use { pcm ->
            session.run(mapOf("reference_pcm_16k" to pcm)).use { outputs -> outputs.half(0) }
        }
    }

    private fun runPromptSemantic(
        environment: OrtEnvironment,
        model: File,
        ssl: ShortArray,
        sessions: GraphSessionCache? = null,
    ): IntArray = withSession(environment, model, "reference_prompt_semantic", sessions = sessions) { session ->
        half(environment, ssl, session.inputShape("ssl_content", ssl.size)).use { content ->
            session.run(mapOf("ssl_content" to content)).use { outputs -> outputs.ints(0) }
        }
    }

    private fun runReferenceConditioning(
        environment: OrtEnvironment,
        model: File,
        pcm16: FloatArray,
        reflectedPcm32: FloatArray,
        sessions: GraphSessionCache? = null,
    ): ReferenceConditioning = withSession(environment, model, "reference_conditioning", sessions = sessions) { session ->
        val pcm16Half = ShortArray(pcm16.size) { floatToHalf(pcm16[it]) }
        val pcm32Half = ShortArray(reflectedPcm32.size) { floatToHalf(reflectedPcm32[it]) }
        half(
            environment,
            pcm16Half,
            session.inputShape("reference_pcm_16k", pcm16Half.size),
        ).use { pcm16Tensor ->
            half(
                environment,
                pcm32Half,
                session.inputShape("reference_pcm_32k_reflected", pcm32Half.size),
            ).use { pcm32Tensor ->
                session.run(
                    mapOf(
                        "reference_pcm_16k" to pcm16Tensor,
                        "reference_pcm_32k_reflected" to pcm32Tensor,
                    )
                ).use { outputs ->
                    // Component compilation is permitted to reorder graph outputs.  Bind
                    // reference tensors by their exported names, never result position.
                    ReferenceConditioning(
                        spectrogram = outputs.half("reference_spectrogram"),
                        speakerEmbedding = outputs.half("speaker_embedding"),
                    )
                }
            }
        }
    }

    private fun runPrefill(
        environment: OrtEnvironment,
        model: File,
        phoneIds: IntArray,
        bert: ShortArray,
        phoneCapacity: Int = 8,
        paddingMaskInput: Boolean = false,
        sessions: GraphSessionCache? = null,
    ): Prefill = withSession(environment, model, "t2s_prefill", sessions = sessions) { session ->
        require(phoneIds.isNotEmpty() && phoneIds.size <= phoneCapacity)
        require(bert.size == phoneIds.size * 1024)
        val paddedPhones = IntArray(phoneCapacity).also { phoneIds.copyInto(it) }
        val paddedBert = ShortArray(phoneCapacity * 1024).also { bert.copyInto(it) }
        val valid = ShortArray(phoneCapacity) { index ->
            floatToHalf(if (index < phoneIds.size) 1.0f else 0.0f)
        }
        int32(
            environment,
            paddedPhones,
            session.inputShape("text_seq", paddedPhones.size),
        ).use { phones ->
            half(
                environment,
                paddedBert,
                session.inputShape("text_bert", paddedBert.size),
            ).use { features ->
                val inputs = linkedMapOf<String, OnnxTensor>("text_seq" to phones, "text_bert" to features)
                if (paddingMaskInput) {
                    half(
                        environment,
                        valid,
                        session.inputShape("text_valid", valid.size),
                    ).use { mask ->
                        inputs["text_valid"] = mask
                        session.run(inputs).use { outputs ->
                            return@withSession readPrefill(outputs)
                        }
                    }
                }
                session.run(inputs).use { outputs -> readPrefill(outputs) }
            }
        }
    }

    private fun runReferencePrefill(
        environment: OrtEnvironment,
        model: File,
        phoneIds: IntArray,
        bert: ShortArray,
        promptSemantic: IntArray,
        promptPhoneIds: IntArray,
        promptBert: ShortArray,
        phoneCapacity: Int,
        sessions: GraphSessionCache? = null,
    ): Prefill = withSession(environment, model, "t2s_reference_prefill", sessions = sessions) { session ->
        require(phoneIds.isNotEmpty() && phoneIds.size <= phoneCapacity)
        require(promptPhoneIds.isNotEmpty() && promptPhoneIds.size <= phoneCapacity)
        require(bert.size == phoneIds.size * 1024)
        require(promptBert.size == promptPhoneIds.size * 1024)
        val paddedPhones = IntArray(phoneCapacity).also { phoneIds.copyInto(it) }
        val paddedBert = ShortArray(phoneCapacity * 1024).also { bert.copyInto(it) }
        val targetValid = ShortArray(phoneCapacity) { index ->
            floatToHalf(if (index < phoneIds.size) 1.0f else 0.0f)
        }
        val paddedPromptPhones = IntArray(phoneCapacity).also { promptPhoneIds.copyInto(it) }
        val paddedPromptBert = ShortArray(phoneCapacity * 1024).also { promptBert.copyInto(it) }
        val promptValid = ShortArray(phoneCapacity) { index ->
            floatToHalf(if (index < promptPhoneIds.size) 1.0f else 0.0f)
        }
        int32(environment, paddedPhones, session.inputShape("text_seq", paddedPhones.size)).use { phones ->
            half(environment, paddedBert, session.inputShape("text_bert", paddedBert.size)).use { features ->
                half(environment, targetValid, session.inputShape("text_valid", targetValid.size)).use { valid ->
                    int32(
                        environment,
                        promptSemantic,
                        session.inputShape("prompt_semantic", promptSemantic.size),
                    ).use { semantic ->
                        int32(
                            environment,
                            paddedPromptPhones,
                            session.inputShape("prompt_phone_ids", paddedPromptPhones.size),
                        ).use { promptPhones ->
                            half(
                                environment,
                                paddedPromptBert,
                                session.inputShape("prompt_bert", paddedPromptBert.size),
                            ).use { promptFeatures ->
                                half(
                                    environment,
                                    promptValid,
                                    session.inputShape("prompt_phone_valid", promptValid.size),
                                ).use { promptMask ->
                                    session.run(
                                        mapOf(
                                            "text_seq" to phones,
                                            "text_bert" to features,
                                            "text_valid" to valid,
                                            "prompt_semantic" to semantic,
                                            "prompt_phone_ids" to promptPhones,
                                            "prompt_bert" to promptFeatures,
                                            "prompt_phone_valid" to promptMask,
                                        )
                                    ).use { outputs -> return@withSession readPrefill(outputs) }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private fun readPrefill(outputs: OrtSession.Result): Prefill {
        val logits = outputs.floats(0)
        val keys = outputs.half(1)
        val values = outputs.half(2)
        Log.i(tag, "prefill logits ${floatStats(logits)} keys ${halfStats(keys)} values ${halfStats(values)}")
        return Prefill(logits, keys, values)
    }

    private data class GeneratedSemantics(val values: IntArray, val validLength: Int)

    /** Keeps the full FP16 K/V cache in direct memory so each token updates only one slot. */
    private class DirectHalfCache(
        environment: OrtEnvironment,
        compact: ShortArray,
        compactLength: Int,
        private val cacheCapacity: Int,
        private val layers: Int,
        private val hidden: Int,
        shape: LongArray,
    ) : java.io.Closeable {
        private val buffer: ShortBuffer
        val tensor: OnnxTensor

        init {
            require(compact.size == layers * compactLength * hidden)
            val elements = Math.multiplyExact(Math.multiplyExact(layers, cacheCapacity), hidden)
            buffer = ByteBuffer.allocateDirect(Math.multiplyExact(elements, 2))
                .order(ByteOrder.nativeOrder())
                .asShortBuffer()
            repeat(layers) { layer ->
                buffer.position(layer * cacheCapacity * hidden)
                buffer.put(compact, layer * compactLength * hidden, compactLength * hidden)
            }
            buffer.position(0)
            tensor = OnnxTensor.createTensor(environment, buffer, shape, OnnxJavaType.FLOAT16)
        }

        fun writeSlot(slot: Int, update: ShortArray) {
            require(update.size == layers * hidden)
            repeat(layers) { layer ->
                val destination = layer * cacheCapacity * hidden + slot * hidden
                val source = layer * hidden
                repeat(hidden) { index -> buffer.put(destination + index, update[source + index]) }
            }
        }

        override fun close() = tensor.close()
    }

    private fun runSteps(
        environment: OrtEnvironment,
        model: File,
        prefill: Prefill,
        promptSemantic: IntArray,
        promptLength: Int,
        compactLength: Int,
        semanticCapacity: Int,
        options: SynthesisOptions = SynthesisOptions(),
        seed: Long = 1234L,
        cacheCapacity: Int = acceptanceCacheCapacity,
        layers: Int = acceptanceLayers,
        hidden: Int = acceptanceHidden,
        eosToken: Int = acceptanceEos,
        initialCacheValid: BooleanArray? = null,
        sessions: GraphSessionCache? = null,
    ): GeneratedSemantics = withSession(environment, model, "t2s_step", sessions = sessions) { session ->
        require(initialCacheValid == null || initialCacheValid.size == compactLength)
        val cacheElements = Math.multiplyExact(Math.multiplyExact(layers, cacheCapacity), hidden)
        DirectHalfCache(
            environment, prefill.keys, compactLength, cacheCapacity, layers, hidden,
            session.inputShape("k_cache", cacheElements),
        ).use { keys ->
            DirectHalfCache(
                environment, prefill.values, compactLength, cacheCapacity, layers, hidden,
                session.inputShape("v_cache", cacheElements),
            ).use { values ->
                val generated = IntArray(semanticCapacity)
                val previous = ArrayList<Int>(promptSemantic.size + semanticCapacity).apply {
                    promptSemantic.forEach(::add)
                }
                val random = Random(seed.takeIf { it >= 0 } ?: System.nanoTime())
                var logits = prefill.logits
                var validLength = 0
                for (iteration in 0 until semanticCapacity) {
                    // V2 Pro Plus's infer_panel_naive excludes EOS for its first 11 decoding
                    // iterations. Without this, a valid early EOS can produce unusably short audio.
                    val token = sampleToken(
                        logits, previous, random, options, eosToken,
                        excludeEos = iteration < minimumSemanticIterations,
                    )
                    if (token == eosToken) break
                    generated[iteration] = token
                    validLength++
                    previous.add(token)
                    if (iteration == semanticCapacity - 1) break
                    val slot = compactLength + iteration
                    val position = sinePosition(promptLength + iteration, hidden)
                    val writeMask = ShortArray(cacheCapacity).also { it[slot] = floatToHalf(1.0f) }
                    val attentionBias = ShortArray(cacheCapacity) { index ->
                        val prefillValid = index < compactLength && (initialCacheValid?.get(index) ?: true)
                        val generatedValid = index in compactLength..slot
                        floatToHalf(if (prefillValid || generatedValid) 0.0f else -10000.0f)
                    }
                    int32(
                        environment, intArrayOf(token), session.inputShape("last_token", 1),
                    ).use { lastToken ->
                        half(
                            environment, position, session.inputShape("position_embedding", position.size),
                        ).use { positionTensor ->
                            half(
                                environment, writeMask, session.inputShape("write_mask", writeMask.size),
                            ).use { maskTensor ->
                                half(
                                    environment, attentionBias,
                                    session.inputShape("attention_bias", attentionBias.size),
                                ).use { biasTensor ->
                                    session.run(
                                        mapOf(
                                            "last_token" to lastToken,
                                            "position_embedding" to positionTensor,
                                            "k_cache" to keys.tensor,
                                            "v_cache" to values.tensor,
                                            "write_mask" to maskTensor,
                                            "attention_bias" to biasTensor,
                                        )
                                    ).use { outputs ->
                                        logits = outputs.floats(0)
                                        if (iteration == 0) Log.i(tag, "step logits ${floatStats(logits)}")
                                        keys.writeSlot(slot, outputs.half(1))
                                        values.writeSlot(slot, outputs.half(2))
                                    }
                                }
                            }
                        }
                    }
                    Log.i(tag, "t2s token ${iteration + 1}/$semanticCapacity=$token")
                }
                if (validLength == 0) {
                    generated[0] = 0
                    validLength = 1
                    Log.w(tag, "T2S predicted EOS before the first semantic token; using upstream-compatible zero token")
                }
                GeneratedSemantics(generated, validLength)
            }
        }
    }

    private fun runAcceptanceVits(
        environment: OrtEnvironment,
        model: File,
        phoneIds: IntArray,
        semantics: GeneratedSemantics,
        seed: Long,
    ): FloatArray = withSession(environment, model, "acceptance_vits") { session ->
        val random = Random(seed)
        val noise = ShortArray(semantics.values.size * 2 * 192) {
            floatToHalf(nextGaussian(random).toFloat())
        }
        int32(
            environment,
            semantics.values,
            session.inputShape("pred_semantic", semantics.values.size),
        ).use { semanticTensor ->
            int32(
                environment,
                phoneIds,
                session.inputShape("text_seq", phoneIds.size),
            ).use { phoneTensor ->
                half(environment, noise, session.inputShape("noise", noise.size)).use { noiseTensor ->
                    session.run(
                        mapOf(
                            "pred_semantic" to semanticTensor,
                            "text_seq" to phoneTensor,
                            "noise" to noiseTensor,
                        )
                    ).use { outputs -> outputs.floats(0) }
                }
            }
        }
    }

    private fun runVits(
        environment: OrtEnvironment,
        model: ModelPackage,
        stages: List<GraphStage>,
        phoneIds: IntArray,
        semantics: GeneratedSemantics,
        seed: Long,
        phoneCapacity: Int = 8,
        semanticCapacity: Int = semantics.values.size,
        samplesPerSemantic: Int = 1280,
        paddingMaskInputs: Boolean = false,
        sessions: GraphSessionCache? = null,
    ): FloatArray {
        require(phoneIds.isNotEmpty() && phoneIds.size <= phoneCapacity)
        require(semantics.values.size == semanticCapacity)
        require(semantics.validLength in 1..semanticCapacity)
        val paddedPhones = IntArray(phoneCapacity).also { phoneIds.copyInto(it) }
        val semanticValid = ShortArray(semanticCapacity) { index ->
            floatToHalf(if (index < semantics.validLength) 1.0f else 0.0f)
        }
        val textValid = ShortArray(phoneCapacity) { index ->
            floatToHalf(if (index < phoneIds.size) 1.0f else 0.0f)
        }
        val random = Random(seed)
        val noiseFrames = semanticCapacity * 2
        val noise = ShortArray(noiseFrames * 192) { floatToHalf(nextGaussian(random).toFloat()) }
        val inputs = linkedMapOf<String, TensorPayload>(
            "pred_semantic" to TensorPayload.integer(semantics.values),
            "text_seq" to TensorPayload.integer(paddedPhones),
            "noise" to TensorPayload.half(noise),
        )
        if (paddingMaskInputs) {
            inputs["semantic_valid"] = TensorPayload.half(semanticValid)
            inputs["text_valid"] = TensorPayload.half(textValid)
        }
        val raw = runGraphStages(environment, model, stages, inputs, "audio", sessions).floats()
        val validSamples = (semantics.validLength * samplesPerSemantic).coerceAtMost(raw.size)
        require(validSamples > 0) { "QNN VITS returned no valid PCM samples" }
        Log.i(tag, "vits partitioned output count=${raw.size} validSamples=$validSamples")
        return raw.copyOf(validSamples)
    }

    private fun runReferenceVits(
        environment: OrtEnvironment,
        model: ModelPackage,
        stages: List<GraphStage>,
        phoneIds: IntArray,
        semantics: GeneratedSemantics,
        referenceSpectrogram: ShortArray,
        speakerEmbedding: ShortArray,
        seed: Long,
        phoneCapacity: Int,
        semanticCapacity: Int,
        samplesPerSemantic: Int,
        sessions: GraphSessionCache,
    ): FloatArray {
        require(phoneIds.isNotEmpty() && phoneIds.size <= phoneCapacity)
        require(semantics.values.size == semanticCapacity)
        require(semantics.validLength in 1..semanticCapacity)
        val paddedPhones = IntArray(phoneCapacity).also { phoneIds.copyInto(it) }
        val semanticValid = ShortArray(semanticCapacity) { index ->
            floatToHalf(if (index < semantics.validLength) 1.0f else 0.0f)
        }
        val textValid = ShortArray(phoneCapacity) { index ->
            floatToHalf(if (index < phoneIds.size) 1.0f else 0.0f)
        }
        val random = Random(seed)
        val noise = ShortArray(semanticCapacity * 2 * 192) {
            floatToHalf(nextGaussian(random).toFloat())
        }
        val inputs = linkedMapOf(
            "pred_semantic" to TensorPayload.integer(semantics.values),
            "text_seq" to TensorPayload.integer(paddedPhones),
            "noise" to TensorPayload.half(noise),
            "semantic_valid" to TensorPayload.half(semanticValid),
            "text_valid" to TensorPayload.half(textValid),
            "reference_spectrogram" to TensorPayload.half(referenceSpectrogram),
            "speaker_embedding" to TensorPayload.half(speakerEmbedding),
        )
        val raw = runGraphStages(environment, model, stages, inputs, "audio", sessions).floats()
        val validSamples = (semantics.validLength * samplesPerSemantic).coerceAtMost(raw.size)
        require(validSamples > 0) { "QNN reference VITS returned no valid PCM samples" }
        return raw.copyOf(validSamples)
    }

    private class TensorPayload private constructor(
        private val type: OnnxJavaType?,
        private val shape: LongArray?,
        private val values: Any,
    ) {
        private val size: Int
            get() = when (values) {
                is ByteArray -> values.size
                is ShortArray -> values.size
                is IntArray -> values.size
                is LongArray -> values.size
                is FloatArray -> values.size
                is DoubleArray -> values.size
                else -> error("unsupported QNN partition tensor storage")
            }

        fun tensor(environment: OrtEnvironment, info: ai.onnxruntime.TensorInfo): OnnxTensor {
            val expectedShape = info.shape
            val elements = expectedShape.fold(1L) { total, value ->
                require(value > 0) { "QNN partition input has a dynamic shape" }
                Math.multiplyExact(total, value)
            }
            require(elements == size.toLong()) {
                "QNN partition input needs $elements values, received $size"
            }
            shape?.let { require(it.contentEquals(expectedShape)) { "QNN partition shape changed" } }
            type?.let { require(it == info.type) { "QNN partition tensor type changed: $it != ${info.type}" } }
            return when (val stored = values) {
                is ByteArray -> OnnxTensor.createTensor(
                    environment,
                    ByteBuffer.wrap(stored),
                    expectedShape,
                    info.type,
                )
                is ShortArray -> OnnxTensor.createTensor(
                    environment,
                    ShortBuffer.wrap(stored),
                    expectedShape,
                    info.type,
                )
                is IntArray -> when (info.type) {
                    OnnxJavaType.INT32 -> OnnxTensor.createTensor(
                        environment,
                        IntBuffer.wrap(stored),
                        expectedShape,
                    )
                    OnnxJavaType.INT64 -> OnnxTensor.createTensor(
                        environment,
                        LongBuffer.wrap(LongArray(stored.size) { stored[it].toLong() }),
                        expectedShape,
                    )
                    else -> error("integer QNN input cannot feed ${info.type}")
                }
                is LongArray -> OnnxTensor.createTensor(
                    environment,
                    LongBuffer.wrap(stored),
                    expectedShape,
                )
                is FloatArray -> OnnxTensor.createTensor(
                    environment,
                    FloatBuffer.wrap(stored),
                    expectedShape,
                )
                is DoubleArray -> OnnxTensor.createTensor(
                    environment,
                    DoubleBuffer.wrap(stored),
                    expectedShape,
                )
                else -> error("unsupported QNN partition tensor storage")
            }
        }

        fun floats(): FloatArray = when {
            type == OnnxJavaType.FLOAT && values is FloatArray -> values
            type == OnnxJavaType.FLOAT16 && values is ShortArray ->
                FloatArray(values.size) { halfToFloat(values[it]) }
            else -> error("QNN partition output is not floating point audio")
        }

        companion object {
            fun integer(values: IntArray) = TensorPayload(null, null, values)
            fun half(values: ShortArray) = TensorPayload(OnnxJavaType.FLOAT16, null, values)

            fun read(tensor: OnnxTensor): TensorPayload {
                val info = tensor.info
                val shape = info.shape.copyOf()
                return when (info.type) {
                    OnnxJavaType.INT8,
                    OnnxJavaType.UINT8,
                    OnnxJavaType.BOOL,
                    -> tensor.byteBuffer.let { buffer ->
                        TensorPayload(info.type, shape, ByteArray(buffer.remaining()).also(buffer::get))
                    }
                    OnnxJavaType.INT16,
                    OnnxJavaType.FLOAT16,
                    OnnxJavaType.BFLOAT16,
                    -> tensor.shortBuffer.let { buffer ->
                        TensorPayload(info.type, shape, ShortArray(buffer.remaining()).also(buffer::get))
                    }
                    OnnxJavaType.INT32 -> tensor.intBuffer.let { buffer ->
                        TensorPayload(info.type, shape, IntArray(buffer.remaining()).also(buffer::get))
                    }
                    OnnxJavaType.INT64 -> tensor.longBuffer.let { buffer ->
                        TensorPayload(info.type, shape, LongArray(buffer.remaining()).also(buffer::get))
                    }
                    OnnxJavaType.FLOAT -> tensor.floatBuffer.let { buffer ->
                        TensorPayload(info.type, shape, FloatArray(buffer.remaining()).also(buffer::get))
                    }
                    OnnxJavaType.DOUBLE -> tensor.doubleBuffer.let { buffer ->
                        TensorPayload(info.type, shape, DoubleArray(buffer.remaining()).also(buffer::get))
                    }
                    else -> error("unsupported QNN partition output type ${info.type}")
                }
            }
        }
    }

    private fun graphStages(array: JSONArray, prefix: String): List<GraphStage> {
        require(array.length() >= 2) { "$prefix must contain at least two QNN partitions" }
        return List(array.length()) { index ->
            val value = array.getJSONObject(index)
            val name = value.getString("name")
            require(name == "%s_%02d".format(prefix, index)) { "$prefix partitions are not ordered" }
            val path = value.getString("path")
            require(path.startsWith("runtime/qnn/") && path.endsWith(".onnx") && ".." !in path) {
                "$prefix partition path is invalid"
            }
            GraphStage(
                name,
                path,
                tensorContracts(value.getJSONArray("inputs"), "$name input"),
                tensorContracts(value.getJSONArray("outputs"), "$name output"),
            )
        }
    }

    private fun tensorContracts(array: JSONArray, label: String): List<TensorContract> {
        val names = HashSet<String>()
        val logicalNames = HashSet<String>()
        return List(array.length()) { index ->
            val value = array.getJSONObject(index)
            val name = value.getString("name")
            require(name.isNotBlank() && names.add(name)) { "$label tensor names must be unique" }
            val logicalName = value.optString("logical_name", name)
            require(logicalName.isNotBlank() && logicalNames.add(logicalName)) {
                "$label logical tensor names must be unique"
            }
            val type = runCatching { OnnxJavaType.valueOf(value.getString("data_type_name")) }
                .getOrElse { error("$label $name has an unsupported tensor type") }
            val rawShape = value.getJSONArray("shape")
            val shape = LongArray(rawShape.length()) { dimension -> rawShape.getLong(dimension) }
            require(shape.all { it > 0 }) { "$label $name has a dynamic shape" }
            TensorContract(name, logicalName, type, shape)
        }
    }

    private fun validateStageSession(session: OrtSession, stage: GraphStage) {
        require(session.inputNames == stage.inputs.mapTo(linkedSetOf(), TensorContract::name)) {
            "${stage.name} compiled input names do not match its executor descriptor"
        }
        require(session.outputNames == stage.outputs.mapTo(linkedSetOf(), TensorContract::name)) {
            "${stage.name} compiled output names do not match its executor descriptor"
        }
        for ((contracts, info) in listOf(stage.inputs to session.inputInfo, stage.outputs to session.outputInfo)) {
            contracts.forEach { contract ->
                val tensor = requireNotNull(info[contract.name]?.info as? ai.onnxruntime.TensorInfo)
                require(tensor.type == contract.type && tensor.shape.contentEquals(contract.shape)) {
                    "${stage.name} tensor ${contract.name} does not match its executor descriptor"
                }
            }
        }
    }

    private fun runGraphStages(
        environment: OrtEnvironment,
        model: ModelPackage,
        stages: List<GraphStage>,
        initial: Map<String, TensorPayload>,
        outputName: String,
        sessions: GraphSessionCache? = null,
    ): TensorPayload {
        val values = HashMap(initial)
        val remainingUses = HashMap<String, Int>()
        stages.flatMap(GraphStage::inputs).forEach { input ->
            remainingUses[input.logicalName] = (remainingUses[input.logicalName] ?: 0) + 1
        }
        stages.forEach { stage ->
            val runStage: ((OrtSession) -> Map<String, TensorPayload>) -> Map<String, TensorPayload> =
                { block ->
                    if (sessions == null) {
                        withSession(environment, model.runtimeFile(stage.path), stage.name, block = block)
                    } else {
                        sessions.use(stage, block)
                    }
                }
            val produced = runStage { session ->
                validateStageSession(session, stage)
                val tensors = linkedMapOf<String, OnnxTensor>()
                try {
                    stage.inputs.forEach { contract ->
                        val payload = requireNotNull(values[contract.logicalName]) {
                            "${stage.name} is missing tensor ${contract.logicalName}"
                        }
                        val info = requireNotNull(
                            session.inputInfo[contract.name]?.info as? ai.onnxruntime.TensorInfo
                        )
                        tensors[contract.name] = payload.tensor(environment, info)
                    }
                    session.run(tensors).use { result ->
                        val contracts = stage.outputs.associateBy(TensorContract::name)
                        result.associate { entry ->
                            val contract = requireNotNull(contracts[entry.key]) {
                                "${stage.name} returned undeclared tensor ${entry.key}"
                            }
                            contract.logicalName to TensorPayload.read(entry.value as OnnxTensor)
                        }
                    }
                } finally {
                    tensors.values.forEach(OnnxTensor::close)
                }
            }
            stage.inputs.forEach { contract ->
                val remaining = requireNotNull(remainingUses[contract.logicalName]) - 1
                remainingUses[contract.logicalName] = remaining
                if (remaining == 0 && contract.logicalName !in produced) {
                    values.remove(contract.logicalName)
                }
            }
            values.putAll(produced)
        }
        return requireNotNull(values[outputName]) { "partitioned QNN graph did not produce $outputName" }
    }

    /** Retains only one VITS variant per loaded artifact. Debug profiling uses one-shot sessions. */
    private class GraphSessionCache(
        private val environment: OrtEnvironment,
        private val model: ModelPackage,
    ) : AutoCloseable {
        private val sessions = LinkedHashMap<String, OrtSession>()

        /** Open a graph during model load so its HTP context is ready before the first request. */
        fun warmPath(path: File, stage: String) {
            sessions.getOrPut(path.canonicalPath) {
                TimingContext.measure("qnn.$stage.session_create") {
                    openQnnSession(environment, path, stage, null)
                }
            }
        }

        fun warm(stage: GraphStage) = warmPath(model.runtimeFile(stage.path), stage.name)

        fun <T> usePath(path: File, stage: String, block: (OrtSession) -> T): T {
            if (DebugQnnProfiles.directory != null) {
                return withSession(environment, path, stage, block = block)
            }
            warmPath(path, stage)
            val session = requireNotNull(sessions[path.canonicalPath])
            return try {
                TimingContext.measure("qnn.$stage.run") { block(session) }
            } catch (error: Throwable) {
                clear()
                throw error
            }
        }

        fun <T> use(stage: GraphStage, block: (OrtSession) -> T): T {
            val path = model.runtimeFile(stage.path)
            return usePath(path, stage.name, block)
        }

        private fun clear() {
            sessions.values.forEach { runCatching { it.close() } }
            sessions.clear()
        }

        override fun close() = clear()
    }

    private fun openQnnSession(
        environment: OrtEnvironment,
        model: File,
        stage: String,
        profilePrefix: String?,
    ): OrtSession {
        require(model.isFile) { "missing EPContext model: $model" }
        val options = OrtSession.SessionOptions().apply {
            setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO)
            addConfigEntry("session.disable_cpu_ep_fallback", "1")
            profilePrefix?.let { enableProfiling(it) }
            addQnn(
                hashMapOf(
                    "backend_type" to "htp",
                    "enable_htp_fp16_precision" to "1",
                    "htp_performance_mode" to "burst",
                    "htp_graph_finalization_optimization_mode" to "3",
                )
            )
        }
        return options.use { sessionOptions ->
            environment.createSession(model.path, sessionOptions).also { session ->
                Log.i(tag, "$stage session inputs=${session.inputNames} outputs=${session.outputNames}")
            }
        }
    }

    private fun <T> withSession(
        environment: OrtEnvironment,
        model: File,
        stage: String,
        sessions: GraphSessionCache? = null,
        block: (OrtSession) -> T,
    ): T {
        sessions?.let { return it.usePath(model, stage, block) }
        val profilePrefix = if (BuildConfig.DEBUG) DebugQnnProfiles.prefix(stage) else null
        return openQnnSession(environment, model, stage, profilePrefix).use { session ->
            try {
                block(session)
            } finally {
                if (profilePrefix != null) {
                    runCatching { session.endProfiling() }
                        .onSuccess { path -> Log.i(tag, "$stage QNN profile=$path cpuFallback=disabled") }
                        .onFailure { error -> Log.w(tag, "$stage QNN profile close failed", error) }
                }
            }
        }
    }

    private fun sampleToken(
        logits: FloatArray,
        previous: List<Int>,
        random: Random,
        options: SynthesisOptions = SynthesisOptions(),
        eosToken: Int = acceptanceEos,
        excludeEos: Boolean = false,
    ): Int {
        require(logits.size == eosToken + 1)
        val adjusted = logits.copyOf()
        previous.toSet().forEach { token ->
            if (token in 0 until eosToken) {
                adjusted[token] = if (adjusted[token] < 0.0f) {
                    adjusted[token] * options.repetitionPenalty
                } else {
                    adjusted[token] / options.repetitionPenalty
                }
            }
        }
        if (excludeEos) adjusted[eosToken] = Float.NEGATIVE_INFINITY
        if (options.temperature == 0.0f) return argmax(adjusted)
        for (index in adjusted.indices) adjusted[index] /= options.temperature
        val top = IntArray(options.topK.coerceAtMost(adjusted.size)) { -1 }
        repeat(adjusted.size) { candidate ->
            var insert = 0
            while (insert < top.size && top[insert] >= 0 && adjusted[top[insert]] >= adjusted[candidate]) insert++
            if (insert < top.size) {
                for (index in top.lastIndex downTo insert + 1) top[index] = top[index - 1]
                top[insert] = candidate
            }
        }
        val maximum = adjusted[top[0]].toDouble()
        val weights = DoubleArray(top.size) { exp(adjusted[top[it]].toDouble() - maximum) }
        if (options.topP < 1.0f) {
            val total = weights.sum()
            var cumulative = 0.0
            for (index in weights.indices) {
                cumulative += weights[index] / total
                if (cumulative >= options.topP) {
                    for (removed in index + 1 until weights.size) weights[removed] = 0.0
                    break
                }
            }
        }
        var selected = random.nextDouble() * weights.sum()
        for (index in top.indices) {
            selected -= weights[index]
            if (selected <= 0.0) return top[index]
        }
        return top.last()
    }

    private fun argmax(values: FloatArray): Int = values.indices.maxBy { values[it] }

    private fun sinePosition(position: Int, hidden: Int = acceptanceHidden): ShortArray = ShortArray(hidden).also { output ->
        var index = 0
        while (index < hidden) {
            val divisor = exp(index.toDouble() * -(ln(10000.0) / hidden))
            output[index] = floatToHalf(sin(position * divisor).toFloat())
            output[index + 1] = floatToHalf(cos(position * divisor).toFloat())
            index += 2
        }
    }

    private fun nextGaussian(random: Random): Double {
        val first = (1.0 - random.nextDouble()).coerceAtLeast(1e-12)
        val second = random.nextDouble()
        return sqrt(-2.0 * ln(first)) * cos(2.0 * PI * second)
    }

    private fun reflectPad(input: FloatArray, padding: Int): FloatArray {
        require(padding >= 0 && input.size > padding) {
            "reference PCM is too short for reflect padding"
        }
        if (padding == 0) return input.copyOf()
        return FloatArray(input.size + padding * 2).also { padded ->
            repeat(padding) { index -> padded[index] = input[padding - index] }
            input.copyInto(padded, padding)
            repeat(padding) { index ->
                padded[padding + input.size + index] = input[input.lastIndex - 1 - index]
            }
        }
    }

    private fun OrtSession.inputShape(name: String, expectedElements: Int): LongArray {
        val tensor = requireNotNull(inputInfo[name]?.info as? ai.onnxruntime.TensorInfo) {
            "QNN session input $name is missing tensor metadata"
        }
        val shape = tensor.shape
        val elements = shape.fold(1L) { total, value ->
            require(value > 0) { "QNN session input $name has a dynamic shape" }
            Math.multiplyExact(total, value)
        }
        require(elements == expectedElements.toLong()) {
            "QNN session input $name needs $elements values, received $expectedElements"
        }
        return shape
    }

    private fun int32(environment: OrtEnvironment, values: IntArray, shape: LongArray) =
        OnnxTensor.createTensor(environment, IntBuffer.wrap(values), shape)

    private fun half(environment: OrtEnvironment, values: ShortArray, shape: LongArray) =
        OnnxTensor.createTensor(environment, ShortBuffer.wrap(values), shape, OnnxJavaType.FLOAT16)

    private fun OrtSession.Result.half(index: Int): ShortArray {
        val buffer = (this[index] as OnnxTensor).shortBuffer
        return ShortArray(buffer.remaining()).also { buffer.get(it) }
    }

    private fun OrtSession.Result.half(name: String): ShortArray {
        val tensor = this.asSequence().firstOrNull { it.key == name }?.value as? OnnxTensor
            ?: error("QNN session did not return FLOAT16 output $name")
        val buffer = tensor.shortBuffer
        return ShortArray(buffer.remaining()).also { buffer.get(it) }
    }

    private fun OrtSession.Result.ints(index: Int): IntArray {
        val buffer = (this[index] as OnnxTensor).intBuffer
        return IntArray(buffer.remaining()).also { buffer.get(it) }
    }

    private fun OrtSession.Result.floats(index: Int): FloatArray =
        half(index).let { values -> FloatArray(values.size) { halfToFloat(values[it]) } }

    private fun halfStats(values: ShortArray): String {
        var nonzero = 0
        var maximum = 0.0f
        var finite = 0
        values.forEach { raw ->
            if (raw.toInt() and 0x7fff != 0) nonzero++
            val value = halfToFloat(raw)
            if (value.isFinite()) {
                finite++
                maximum = maxOf(maximum, kotlin.math.abs(value))
            }
        }
        return "count=${values.size} nonzero=$nonzero finite=$finite absMax=$maximum bits=${values.take(8).joinToString { (it.toInt() and 0xffff).toString(16) }}"
    }

    private fun floatStats(values: FloatArray): String {
        val finite = values.filter(Float::isFinite)
        val top = values.indices.sortedByDescending { values[it] }.take(10)
        return "count=${values.size} finite=${finite.size} min=${finite.minOrNull()} max=${finite.maxOrNull()} top=${top.joinToString { "$it:${values[it]}" }}"
    }

    private inline fun <T> timed(name: String, values: MutableMap<String, Long>, block: () -> T): T {
        val started = System.nanoTime()
        return try {
            block()
        } finally {
            values[name] = (System.nanoTime() - started) / 1_000_000
        }
    }

    private fun JSONObject.intArray(name: String): IntArray = getJSONArray(name).let { values ->
        IntArray(values.length()) { values.getInt(it) }
    }

    private fun writeWav(file: File, pcm: ShortArray, sampleRate: Int) {
        file.parentFile?.mkdirs()
        RandomAccessFile(file, "rw").use { out ->
            out.setLength(0)
            val dataSize = pcm.size * 2
            out.writeBytes("RIFF"); writeIntLE(out, 36 + dataSize); out.writeBytes("WAVE")
            out.writeBytes("fmt "); writeIntLE(out, 16); writeShortLE(out, 1); writeShortLE(out, 1)
            writeIntLE(out, sampleRate); writeIntLE(out, sampleRate * 2)
            writeShortLE(out, 2); writeShortLE(out, 16); out.writeBytes("data"); writeIntLE(out, dataSize)
            pcm.forEach { writeShortLE(out, it.toInt()) }
        }
    }

    private fun writeIntLE(out: RandomAccessFile, value: Int) {
        out.write(value); out.write(value ushr 8); out.write(value ushr 16); out.write(value ushr 24)
    }

    private fun writeShortLE(out: RandomAccessFile, value: Int) {
        out.write(value); out.write(value ushr 8)
    }

    private fun floatToHalf(value: Float): Short {
        val bits = value.toRawBits()
        val sign = (bits ushr 16) and 0x8000
        var exponent = ((bits ushr 23) and 0xff) - 127 + 15
        var mantissa = bits and 0x7fffff
        if (exponent <= 0) {
            if (exponent < -10) return sign.toShort()
            mantissa = (mantissa or 0x800000) shr (1 - exponent)
            return (sign or ((mantissa + 0x1000) shr 13)).toShort()
        }
        if (exponent >= 31) return (sign or 0x7c00).toShort()
        mantissa += 0x1000
        if ((mantissa and 0x800000) != 0) {
            mantissa = 0
            exponent++
            if (exponent >= 31) return (sign or 0x7c00).toShort()
        }
        return (sign or (exponent shl 10) or (mantissa shr 13)).toShort()
    }

    private fun halfToFloat(value: Short): Float {
        val half = value.toInt() and 0xffff
        val sign = (half and 0x8000) shl 16
        var exponent = (half ushr 10) and 0x1f
        var mantissa = half and 0x3ff
        val bits = when (exponent) {
            0 -> if (mantissa == 0) sign else {
                exponent = 1
                while ((mantissa and 0x400) == 0) { mantissa = mantissa shl 1; exponent-- }
                mantissa = mantissa and 0x3ff
                sign or ((exponent + 112) shl 23) or (mantissa shl 13)
            }
            31 -> sign or 0x7f800000 or (mantissa shl 13)
            else -> sign or ((exponent + 112) shl 23) or (mantissa shl 13)
        }
        return Float.fromBits(bits)
    }
}
