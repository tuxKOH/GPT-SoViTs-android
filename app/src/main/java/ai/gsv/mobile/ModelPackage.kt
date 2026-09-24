package ai.gsv.mobile

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.InputStream
import java.security.MessageDigest
import java.util.zip.ZipInputStream

internal const val QNN_ATTACHMENT_SUFFIX = ".qnn.gsvm"

internal class UnsupportedLegacyModelException : IllegalArgumentException(
    "Legacy model packages are not supported. Regenerate the model with the web converter.",
)

data class ModelPackage(
    val root: File,
    val name: String,
    val version: String,
    /** Product/reconstruction version shown to users (independent of the stable model ABI id). */
    val productVersion: String = "",
    val sampleRate: Int,
    val runtime: String,
    val formatVersion: Int,
    val executor: String,
    val entrypoint: String,
    val deployable: Boolean,
    val runtimeOptionsVersion: Int = 0,
    val runtimeOptions: Set<String> = emptySet(),
    val referenceInputVersion: Int = 0,
    val referenceDurationPolicy: String = "",
    val referencePcm16kSamples: Int = 0,
    val referencePcm32kSamples: Int = 0,
    val bundleId: String = "",
    val attachmentFor: String = "",
    val baseModelSha256: String = "",
    val artifactRole: String = "combined",
    val pipelineRoot: File = root,
    val modelRoot: File = root,
    /** Canonical target id for a backend-specific artifact, or `any` for CPU/cross-device data. */
    val targetSoc: String = "any",
    val targetSocFamily: String = "",
    val targetAsic: String = "",
    val targetSocModel: Int = 0,
    /** Exact HTP architecture from the QAIRT SDK used during conversion. */
    val htpArch: String = "",
    val qairtVersion: String = "",
    val qnnRuntimeVersion: String = "",
    /** Relative path to the prepared backend executor artifact. */
    val backendArtifact: String = "",
    val supportedTargetSocs: Set<String> = emptySet(),
    /** QNN packages may explicitly allow CPU partition fallback during bring-up. */
    val strictCpuFallback: Boolean = true,
) {
    fun runtimeFile(path: String): File {
        require(path.isNotBlank() && !File(path).isAbsolute) { "runtime path must be relative" }
        fun resolve(root: File): File {
            val file = File(root, path).canonicalFile
            require(file.path.startsWith(root.canonicalFile.path + File.separator)) {
                "runtime path escapes its package root: $path"
            }
            return file
        }
        val pipelineFile = resolve(pipelineRoot)
        if (pipelineFile.exists()) return pipelineFile
        return resolve(modelRoot)
    }

    fun isSplit() = pipelineRoot.canonicalFile != modelRoot.canonicalFile

    companion object {
        private data class ExpectedFile(val size: Long, val sha256: String)

        fun import(context: Context, uri: Uri): ModelPackage {
            context.contentResolver.openInputStream(uri).use { raw ->
                requireNotNull(raw) { "could not open the selected file" }
                return importCombined(context, raw)
            }
        }

        fun importFile(context:Context,file:File):ModelPackage = file.inputStream().use {
            importCombined(context, it)
        }

        internal fun inspectManifest(raw: InputStream): JSONObject =
            ZipInputStream(raw.buffered()).use { zip ->
                val first = zip.nextEntry ?: error("empty model package")
                require(first.name == "manifest.json") {
                    "manifest.json must be the first package entry"
                }
                JSONObject(readEntryBytes(zip, MAX_MANIFEST_BYTES).toString(Charsets.UTF_8))
            }

        private fun importCombined(context: Context, raw: InputStream): ModelPackage {
            val installedRoot = File(context.filesDir, "models/active")
            val pendingRoot = File(context.filesDir, "models/active.pending")
            try {
                val imported = importStream(context, raw, pendingRoot)
                commitDirectory(pendingRoot, installedRoot, "could not commit the model package")
                return imported.copy(
                    root = installedRoot,
                    pipelineRoot = installedRoot,
                    modelRoot = installedRoot,
                )
            } catch (error: Throwable) {
                pendingRoot.deleteRecursively()
                throw error
            }
        }

        fun hasInstalledPipeline(context: Context): Boolean = installedPipelineVersions(context).isNotEmpty()

        internal fun openInstalledCpuPair(context: Context): ModelPackage {
            val modelRoot = File(context.filesDir, "models/model")
            require(File(modelRoot, "manifest.json").isFile) { "no installed CPU voice model" }
            val modelPackage = openExtracted(modelRoot)
            val pipelineRoot = File(context.filesDir, "models/pipelines/${modelPackage.version}")
            require(File(pipelineRoot, "manifest.json").isFile) {
                "the ${modelPackage.version} pipeline component is not installed"
            }
            return combineSplit(openExtracted(pipelineRoot), modelPackage, pipelineRoot, modelRoot)
        }

        fun installedQnnPipelineTargets(context: Context, version: String): Set<String> =
            File(context.filesDir, "models/qnn-pipelines/$version").also(::recoverPreviousDirectories).listFiles().orEmpty().mapNotNull { root ->
                if (root.name.endsWith(".pending") || !File(root, "manifest.json").isFile) return@mapNotNull null
                runCatching {
                    openExtracted(root).takeIf {
                        it.artifactRole == "qnn-pipeline-attachment" && it.executor == "qnn-htp"
                    }?.targetSoc
                }.getOrNull()
            }.toSet()

        fun restorePipelineArchives(context: Context) {
            val archives = File(context.filesDir, "components").also { it.mkdirs() }
            mapOf("v2ProPlus" to "pipeline-v2ProPlus.gsvm", "v4" to "pipeline-v4.gsvm").forEach { (version, name) ->
                val installed = File(context.filesDir, "models/pipelines/$version/manifest.json")
                val archive = File(archives, name)
                if (!installed.isFile && archive.isFile) {
                    runCatching { installPipeline(context, Uri.fromFile(archive), version) }
                }
            }
        }

        fun installedPipelineVersions(context: Context): Set<String> {
            val root = File(context.filesDir, "models/pipelines")
            recoverPreviousDirectories(root)
            val legacy = File(context.filesDir, "models/pipeline")
            if (File(legacy, "manifest.json").isFile) {
                runCatching {
                    val version = openExtracted(legacy).version
                    val destination = File(root, version)
                    if (!destination.exists()) {
                        root.mkdirs()
                        require(legacy.renameTo(destination)) { "could not migrate the legacy pipeline component" }
                    }
                }
            }
            return root.listFiles().orEmpty().mapNotNull { directory ->
                if (directory.name.endsWith(".pending")) return@mapNotNull null
                runCatching {
                    val manifest = JSONObject(File(directory, "manifest.json").readText())
                    if (manifest.optString("artifact_role") == "pipeline") manifest.getString("model_version") else null
                }.getOrNull()
            }.toSet()
        }

        fun installPipeline(
            context: Context,
            pipeline: Uri,
            expectedVersion: String,
        ): ModelPackage {
            val pipelinesRoot = File(context.filesDir, "models/pipelines")
            val pendingRoot = File(pipelinesRoot, "$expectedVersion.pending")
            val pipelinePackage = context.contentResolver.openInputStream(pipeline).use { raw ->
                requireNotNull(raw) { "could not open the pipeline package" }
                importStream(context, raw, pendingRoot)
            }
            require(pipelinePackage.artifactRole == "pipeline") { "the selected file is not a pipeline package" }
            require(pipelinePackage.version == expectedVersion) {
                "selected $expectedVersion, but the package contains ${pipelinePackage.version}"
            }
            require(pipelinePackage.executor in setOf("torchscript-cpu-single", "torchscript-cpu-staged")) {
                "the base pipeline must be a CPU package; import QNN as a .qnn.gsvm attachment"
            }
            val installedRoot = File(pipelinesRoot, expectedVersion)
            pipelinesRoot.mkdirs()
            commitDirectory(pendingRoot, installedRoot, "could not commit the pipeline component")
            return pipelinePackage.copy(root = installedRoot, pipelineRoot = installedRoot, modelRoot = installedRoot)
        }

        fun installQnnPipelineAttachment(
            context: Context,
            attachment: Uri,
            expectedVersion: String,
        ): ModelPackage {
            requireQnnSuffix(context, attachment)
            val baseRoot = File(context.filesDir, "models/pipelines/$expectedVersion")
            require(File(baseRoot, "manifest.json").isFile) {
                "install the $expectedVersion base pipeline before its QNN attachment"
            }
            val base = openExtracted(baseRoot)
            val qnnRoot = File(context.filesDir, "models/qnn-pipelines")
            val pendingRoot = File(qnnRoot, "$expectedVersion.pending")
            try {
                val qnnPackage = context.contentResolver.openInputStream(attachment).use { raw ->
                    requireNotNull(raw) { "could not open the QNN pipeline attachment" }
                    importStream(context, raw, pendingRoot)
                }
                require(qnnPackage.artifactRole == "qnn-pipeline-attachment") {
                    "the selected .qnn.gsvm is not a QNN pipeline attachment"
                }
                require(qnnPackage.version == expectedVersion) {
                    "QNN attachment is ${qnnPackage.version}, expected $expectedVersion"
                }
                validateQnnAttachment(qnnPackage)
                require(frontendAbi(qnnPackage.attachmentFor) == frontendAbi(base.bundleId)) {
                    "QNN attachment does not match the installed base pipeline ABI"
                }
                val status = QualcommTargetPolicy.current()
                require(QualcommTargetPolicy.isCompatible(qnnPackage, status)) {
                    "QNN pipeline attachment is incompatible: ${QualcommTargetPolicy.explain(qnnPackage, status)}"
                }
                val installedRoot = File(qnnRoot, "$expectedVersion/${qnnPackage.targetSoc}")
                installedRoot.parentFile?.mkdirs()
                commitDirectory(pendingRoot, installedRoot, "could not commit the QNN pipeline attachment")
                return qnnPackage.copy(root = installedRoot, pipelineRoot = installedRoot, modelRoot = installedRoot)
            } catch (error: Throwable) {
                pendingRoot.deleteRecursively()
                throw error
            }
        }

        fun importModelWithInstalledPipeline(context: Context, model: Uri): ModelPackage {
            val modelRoot = File(context.filesDir, "models/model")
            val pendingRoot = File(context.filesDir, "models/model.pending")
            try {
                val modelPackage = context.contentResolver.openInputStream(model).use { raw ->
                    requireNotNull(raw) { "could not open the model package" }
                    importStream(context, raw, pendingRoot)
                }
                val pipelineRoot = File(context.filesDir, "models/pipelines/${modelPackage.version}")
                require(File(pipelineRoot, "manifest.json").isFile) {
                    "the ${modelPackage.version} pipeline component is not installed"
                }
                val pipelinePackage = openExtracted(pipelineRoot)
                val combined = combineSplit(pipelinePackage, modelPackage, pipelineRoot, pendingRoot)
                commitDirectory(pendingRoot, modelRoot, "could not commit the voice model")
                return combined.copy(root = modelRoot, modelRoot = modelRoot)
            } catch (error: Throwable) {
                pendingRoot.deleteRecursively()
                throw error
            }
        }

        fun importQnnModelWithInstalledPipeline(
            context: Context,
            baseModel: Uri,
            attachment: Uri,
        ): ModelPackage {
            requireQnnSuffix(context, attachment)
            val pendingRoot = File(context.filesDir, "models/qnn-model.pending")
            try {
                val modelPackage = context.contentResolver.openInputStream(attachment).use { raw ->
                    requireNotNull(raw) { "could not open the QNN voice attachment" }
                    importStream(context, raw, pendingRoot)
                }
                require(modelPackage.artifactRole == "qnn-model-attachment") {
                    "the selected .qnn.gsvm is not a QNN voice attachment"
                }
                validateQnnAttachment(modelPackage)
                require(modelPackage.baseModelSha256.isNotBlank()) {
                    "QNN voice attachment is not bound to a base model package"
                }
                val selectedBaseHash = context.contentResolver.openInputStream(baseModel).use { raw ->
                    requireNotNull(raw) { "could not open the base voice model" }
                    sha256(raw)
                }
                require(selectedBaseHash == modelPackage.baseModelSha256) {
                    "QNN voice attachment does not match the selected base voice model"
                }
                val pipelineRoot = File(
                    context.filesDir,
                    "models/qnn-pipelines/${modelPackage.version}/${modelPackage.targetSoc}",
                )
                require(File(pipelineRoot, "manifest.json").isFile) {
                    "install the matching ${modelPackage.version} ${modelPackage.targetSoc} QNN pipeline attachment first"
                }
                val pipelinePackage = openExtracted(pipelineRoot)
                val combined = combineQnnAttachments(pipelinePackage, modelPackage, pipelineRoot, pendingRoot)
                val installedRoot = qnnModelRoot(context, selectedBaseHash, modelPackage.targetSoc)
                installedRoot.parentFile?.mkdirs()
                commitDirectory(pendingRoot, installedRoot, "could not commit the QNN voice attachment")
                return combined.copy(root = installedRoot, modelRoot = installedRoot)
            } catch (error: Throwable) {
                pendingRoot.deleteRecursively()
                throw error
            }
        }

        fun openInstalledQnnModelWithPipeline(
            context: Context,
            baseModel: Uri,
        ): ModelPackage {
            val selectedBaseHash = context.contentResolver.openInputStream(baseModel).use { raw ->
                requireNotNull(raw) { "could not open the base voice model" }
                sha256(raw)
            }
            return openInstalledQnnModelWithPipeline(context, selectedBaseHash)
        }

        fun openInstalledQnnModelWithPipeline(
            context: Context,
            selectedBaseHash: String,
        ): ModelPackage {
            require(selectedBaseHash.matches(Regex("[0-9a-f]{64}"))) { "invalid base model SHA-256" }
            migrateLegacyQnnModel(context)
            val status = QualcommTargetPolicy.current()
            val candidateRoot = File(context.filesDir, "models/qnn-models/$selectedBaseHash").also(
                ::recoverPreviousDirectories
            )
            val candidates = candidateRoot
                .listFiles()
                .orEmpty()
                .filter { File(it, "manifest.json").isFile }
                .sortedBy { it.name }
            val selected = candidates.firstNotNullOfOrNull { root ->
                runCatching {
                    val value = openExtracted(root)
                    require(value.artifactRole == "qnn-model-attachment") {
                        "the installed QNN voice attachment has an invalid role"
                    }
                    validateQnnAttachment(value)
                    require(value.baseModelSha256 == selectedBaseHash) {
                        "the installed QNN voice attachment belongs to another base model"
                    }
                    if (QualcommTargetPolicy.isCompatible(value, status)) value to root else null
                }.getOrNull()
            }
            val (modelPackage, modelRoot) = requireNotNull(selected) {
                "the matching installed QNN voice attachment is missing"
            }
            val pipelineRoot = File(
                context.filesDir,
                "models/qnn-pipelines/${modelPackage.version}/${modelPackage.targetSoc}",
            )
            require(File(pipelineRoot, "manifest.json").isFile) {
                "the matching QNN pipeline attachment is not installed"
            }
            val pipelinePackage = openExtracted(pipelineRoot)
            return combineQnnAttachments(pipelinePackage, modelPackage, pipelineRoot, modelRoot)
        }

        private fun qnnModelRoot(context: Context, baseModelSha256: String, targetSoc: String): File {
            require(baseModelSha256.matches(Regex("[0-9a-f]{64}"))) { "invalid base model SHA-256" }
            require(targetSoc.matches(Regex("[a-z0-9_]+"))) { "invalid QNN target name" }
            return File(context.filesDir, "models/qnn-models/$baseModelSha256/$targetSoc")
        }

        private fun migrateLegacyQnnModel(context: Context) {
            val legacy = File(context.filesDir, "models/qnn-model")
            if (!File(legacy, "manifest.json").isFile) return
            val model = runCatching { openExtracted(legacy) }.getOrNull() ?: return
            if (model.artifactRole != "qnn-model-attachment" || model.baseModelSha256.isBlank()) return
            val destination = runCatching {
                qnnModelRoot(context, model.baseModelSha256, model.targetSoc)
            }.getOrNull() ?: return
            if (destination.exists()) {
                legacy.deleteRecursively()
                return
            }
            destination.parentFile?.mkdirs()
            require(legacy.renameTo(destination)) { "could not migrate the installed QNN voice attachment" }
        }

        private fun commitDirectory(pending: File, installed: File, message: String) {
            require(pending.isDirectory) { "pending model directory is missing" }
            installed.parentFile?.mkdirs()
            val backup = File(installed.parentFile, "${installed.name}.previous")
            if (!installed.exists() && backup.exists()) {
                require(backup.renameTo(installed)) { "could not restore the previous model installation" }
            }
            require(!backup.exists() || backup.deleteRecursively()) {
                "could not remove a stale model installation backup"
            }
            if (installed.exists()) {
                require(installed.renameTo(backup)) { "could not preserve the previous model installation" }
            }
            try {
                require(pending.renameTo(installed)) { message }
            } catch (error: Throwable) {
                if (!installed.exists() && backup.exists()) {
                    runCatching { backup.renameTo(installed) }
                }
                throw error
            }
            require(!backup.exists() || backup.deleteRecursively()) {
                "could not remove the previous model installation"
            }
        }

        private fun recoverPreviousDirectories(parent: File) {
            parent.listFiles().orEmpty().filter { it.isDirectory && it.name.endsWith(".previous") }.forEach { backup ->
                val installed = File(parent, backup.name.removeSuffix(".previous"))
                if (installed.exists()) {
                    backup.deleteRecursively()
                } else {
                    require(backup.renameTo(installed)) { "could not recover a previous model installation" }
                }
            }
        }

        /** Import independently packaged pipeline and model artifacts and combine them only after
         * both manifests, hashes and compatibility IDs have been checked. */
        fun importPair(context: Context, pipeline: Uri, model: Uri): ModelPackage {
            val pipelineRoot = File(context.filesDir, "models/pipeline")
            val modelRoot = File(context.filesDir, "models/model")
            val pipelinePackage = context.contentResolver.openInputStream(pipeline).use { raw ->
                requireNotNull(raw) { "could not open the pipeline package" }
                importStream(context, raw, pipelineRoot)
            }
            val modelPackage = context.contentResolver.openInputStream(model).use { raw ->
                requireNotNull(raw) { "could not open the model package" }
                importStream(context, raw, modelRoot)
            }
            return combineSplit(pipelinePackage, modelPackage, pipelineRoot, modelRoot)
        }

        private fun combineSplit(
            pipelinePackage: ModelPackage,
            modelPackage: ModelPackage,
            pipelineRoot: File,
            modelRoot: File,
        ): ModelPackage {
            require(pipelinePackage.version == modelPackage.version) { "pipeline and model versions do not match" }
            require(pipelinePackage.sampleRate == modelPackage.sampleRate) { "pipeline and model sample rates do not match" }
            require(pipelinePackage.entrypoint == modelPackage.entrypoint) { "pipeline and model entrypoints do not match" }
            require(pipelinePackage.artifactRole == "pipeline" && modelPackage.artifactRole == "model") { "pipeline or model artifact role is invalid" }
            val pipelineAbiId = pipelinePackage.bundleId.ifBlank { "<missing>" }
            val modelAbiId = modelPackage.bundleId.ifBlank { "<missing>" }
            require(
                pipelinePackage.bundleId.isNotEmpty() &&
                frontendAbi(pipelinePackage.bundleId) == frontendAbi(modelPackage.bundleId)
            ) {
                "pipeline and model frontend ABIs do not match (pipeline=$pipelineAbiId, model=$modelAbiId)"
            }
            require(targetsCompatible(pipelinePackage, modelPackage)) { "pipeline and model target backends do not match" }
            // The model package owns the executor/backend identity; pipelineRoot contributes
            // frontend assets while modelRoot contributes the selected backend artifact.
            return modelPackage.copy(
                pipelineRoot = pipelineRoot,
                modelRoot = modelRoot,
                artifactRole = "combined",
            )
        }

        private fun combineQnnAttachments(
            pipelinePackage: ModelPackage,
            modelPackage: ModelPackage,
            pipelineRoot: File,
            modelRoot: File,
        ): ModelPackage {
            require(pipelinePackage.artifactRole == "qnn-pipeline-attachment") {
                "installed QNN pipeline role is invalid"
            }
            require(modelPackage.artifactRole == "qnn-model-attachment") {
                "selected QNN voice role is invalid"
            }
            require(pipelinePackage.version == modelPackage.version) { "QNN pipeline and voice versions do not match" }
            require(pipelinePackage.sampleRate == modelPackage.sampleRate) { "QNN pipeline and voice sample rates do not match" }
            require(pipelinePackage.entrypoint == modelPackage.entrypoint) { "QNN attachment entrypoints do not match" }
            require(pipelinePackage.bundleId.isNotBlank() && pipelinePackage.bundleId == modelPackage.bundleId) {
                "QNN pipeline and voice backend ABIs do not match"
            }
            require(pipelinePackage.attachmentFor.isNotBlank() && pipelinePackage.attachmentFor == modelPackage.attachmentFor) {
                "QNN attachments target different base pipeline ABIs"
            }
            require(targetsCompatible(pipelinePackage, modelPackage)) { "QNN attachments target different SoCs" }
            validateQnnAttachment(pipelinePackage)
            validateQnnAttachment(modelPackage)
            return modelPackage.copy(
                root = modelRoot,
                pipelineRoot = pipelineRoot,
                modelRoot = modelRoot,
                artifactRole = "combined",
            )
        }

        fun openExtracted(root:File):ModelPackage {
            val manifest=JSONObject(File(root,"manifest.json").readText())
            val canonicalRoot = root.canonicalFile
            val files=manifest.getJSONArray("files");for(i in 0 until files.length()){
                val item=files.getJSONObject(i)
                val path=item.getString("path")
                val file=File(root,path).canonicalFile
                require(file.path.startsWith(canonicalRoot.path + File.separator)) { "invalid installed model path: $path" }
                require(file.isFile&&file.length()==item.getLong("size")&&sha256(file)==item.getString("sha256")){"$path verification failed"}
            }
            return fromManifest(root,manifest)
        }

        private fun fromManifest(root:File,manifest:JSONObject):ModelPackage {
            val supported = manifest.optJSONArray("supported_target_socs")?.let { values ->
                mutableSetOf<String>().apply {
                    for (index in 0 until values.length()) add(values.getString(index))
                }
            } ?: emptySet()
            val runtimeOptions = manifest.optJSONObject("runtime_options")?.let { values ->
                buildSet {
                    val keys = values.keys()
                    while (keys.hasNext()) add(keys.next())
                }
            } ?: emptySet()
            val referenceInput = manifest.optJSONObject("reference_input")
            val referencePcm = referenceInput?.optJSONArray("pcm")
            fun referenceSamples(sampleRate: Int): Int {
                if (referencePcm == null) return 0
                for (index in 0 until referencePcm.length()) {
                    val value = referencePcm.getJSONObject(index)
                    if (value.optInt("sample_rate") == sampleRate) return value.optInt("samples")
                }
                return 0
            }
            val parsed = ModelPackage(
                root = root,
                name = manifest.getString("name"),
                version = manifest.getString("model_version"),
                productVersion = manifest.optString("product_version", ""),
                sampleRate = manifest.optInt("sample_rate", 32000),
                runtime = manifest.optString("runtime", manifest.optString("executor", "unknown")),
                formatVersion = manifest.getInt("format_version"),
                executor = manifest.optString("executor", "unknown"),
                entrypoint = manifest.optString("entrypoint", "unknown"),
                deployable = manifest.optBoolean("deployable", false),
                runtimeOptionsVersion = manifest.optInt("runtime_options_version", 0),
                runtimeOptions = runtimeOptions,
                referenceInputVersion = manifest.optInt("reference_input_version", 0),
                referenceDurationPolicy = referenceInput?.optString("duration_policy").orEmpty(),
                referencePcm16kSamples = referenceSamples(16000),
                referencePcm32kSamples = referenceSamples(32000),
                bundleId = manifest.optString("bundle_id", ""),
                attachmentFor = manifest.optString("attachment_for", ""),
                baseModelSha256 = manifest.optString("base_model_sha256", ""),
                artifactRole = manifest.optString("artifact_role", "combined"),
                targetSoc = manifest.optString("target_soc", "any"),
                targetSocFamily = manifest.optString("target_soc_family", ""),
                targetAsic = manifest.optString("target_asic", ""),
                targetSocModel = manifest.optInt("target_soc_model", 0),
                htpArch = manifest.optString("htp_arch", ""),
                qairtVersion = manifest.optString("qairt_version", ""),
                qnnRuntimeVersion = manifest.optString("qnn_runtime_version", ""),
                backendArtifact = manifest.optString("backend_artifact", ""),
                supportedTargetSocs = supported,
                strictCpuFallback = if (manifest.has("cpu_neural_fallback")) {
                    !manifest.getBoolean("cpu_neural_fallback")
                } else {
                    manifest.optJSONObject("npu_plan")?.optBoolean("strict_cpu_fallback", true) ?: true
                },
            )
            if (parsed.executor == "qnn-htp") {
                QualcommTargetPolicy.requireArtifactIdentity(parsed)
                require(parsed.strictCpuFallback) { "QNN package enables CPU neural fallback" }
                if (parsed.referenceInputVersion >= 1) {
                    require(parsed.referenceDurationPolicy == "exact_samples") {
                        "QNN package has an unsupported reference duration policy"
                    }
                    require(
                        parsed.referencePcm16kSamples > 0 && parsed.referencePcm32kSamples > 0 &&
                            parsed.referencePcm32kSamples == parsed.referencePcm16kSamples * 2
                    ) { "QNN package has invalid reference PCM capacities" }
                }
            }
            return parsed
        }

        private fun validateQnnAttachment(value: ModelPackage) {
            require(value.executor == "qnn-htp") { "QNN attachment executor must be qnn-htp" }
            require(value.deployable) { "QNN attachment is a development artifact and is not deployable" }
            require(value.strictCpuFallback) { "QNN attachment must disable CPU neural fallback" }
            require(value.entrypoint == "synthesize_utf8_to_pcm16") { "QNN attachment entrypoint is unsupported" }
            require(value.attachmentFor.isNotBlank()) { "QNN attachment is missing attachment_for" }
            require(value.runtimeFile(value.backendArtifact).isFile) {
                "QNN attachment is missing its prepared backend artifact"
            }
        }

        private fun frontendAbi(value: String) = value.replace(Regex(":options\\d+$"), "")

        private fun requireQnnSuffix(context: Context, uri: Uri) {
            val displayName = context.contentResolver.query(
                uri,
                arrayOf(OpenableColumns.DISPLAY_NAME),
                null,
                null,
                null,
            )?.use { cursor ->
                if (cursor.moveToFirst()) cursor.getString(0) else null
            } ?: uri.lastPathSegment.orEmpty()
            require(displayName.endsWith(QNN_ATTACHMENT_SUFFIX, ignoreCase = true)) {
                "QNN attachments must use the $QNN_ATTACHMENT_SUFFIX suffix"
            }
        }

        private fun targetsCompatible(pipeline: ModelPackage, model: ModelPackage): Boolean {
            fun compatible(left: String, right: String): Boolean =
                left.isBlank() || left == "any" || right.isBlank() || right == "any" || left == right
            if (!compatible(pipeline.targetSoc, model.targetSoc)) return false
            if (!compatible(pipeline.targetSocFamily, model.targetSocFamily)) return false
            if (!compatible(pipeline.targetAsic, model.targetAsic)) return false
            if (
                pipeline.targetSocModel > 0 && model.targetSocModel > 0 &&
                pipeline.targetSocModel != model.targetSocModel
            ) return false
            if (!compatible(pipeline.htpArch, model.htpArch)) return false
            if (!compatible(pipeline.qairtVersion, model.qairtVersion)) return false
            if (!compatible(pipeline.qnnRuntimeVersion, model.qnnRuntimeVersion)) return false
            val pipelineTargets = pipeline.supportedTargetSocs
            val modelTargets = model.supportedTargetSocs
            return pipelineTargets.isEmpty() || modelTargets.isEmpty() || pipelineTargets.intersect(modelTargets).isNotEmpty()
        }

        private fun importStream(context:Context,raw:InputStream,unpack:File):ModelPackage {
            unpack.deleteRecursively(); unpack.mkdirs()
            try {
                ZipInputStream(raw.buffered()).use { zip ->
                    val first = zip.nextEntry ?: error("empty model package")
                    require(first.name == "manifest.json") { "manifest.json must be the first package entry" }
                    val manifestBytes = readEntryBytes(zip, MAX_MANIFEST_BYTES)
                    val manifest = JSONObject(manifestBytes.toString(Charsets.UTF_8))
                    rejectUnsupportedLegacyModel(manifest)
                    val expected = buildMap<String, ExpectedFile> {
                        val files = manifest.getJSONArray("files")
                        for (i in 0 until files.length()) {
                            val item = files.getJSONObject(i)
                            val path = item.getString("path")
                            require(path.isNotBlank() && path != "manifest.json") {
                                "model package declares an invalid payload path"
                            }
                            val size = item.getLong("size")
                            require(size >= 0L && size <= MAX_PACKAGE_FILE_BYTES) {
                                "model package file $path has an invalid size"
                            }
                            require(put(path, ExpectedFile(size, item.getString("sha256"))) == null) {
                                "model package declares $path more than once"
                            }
                        }
                    }.toMutableMap()
                    val requiredBytes = expected.values.fold(manifestBytes.size.toLong()) { total, item ->
                        Math.addExact(total, item.size)
                    }
                    require(requiredBytes <= MAX_PACKAGE_EXTRACT_BYTES) {
                        "model package expands beyond the supported size"
                    }
                    val usableBytes = unpack.parentFile?.usableSpace ?: unpack.usableSpace
                    require(usableBytes >= requiredBytes + MIN_FREE_SPACE_AFTER_IMPORT) {
                        "not enough storage to install this model package"
                    }
                    // Persist the package identity alongside extracted payloads. Startup discovery
                    // and archive recovery must not depend on the in-memory import result.
                    File(unpack, "manifest.json").writeBytes(manifestBytes)
                    while (true) {
                        val entry = zip.nextEntry ?: break
                        if (entry.isDirectory) continue
                        val expectedFile = expected.remove(entry.name) ?: error("undeclared file ${entry.name}")
                        val out = File(unpack, entry.name)
                        require(out.canonicalPath.startsWith(unpack.canonicalPath + File.separator)) { "invalid model path" }
                        out.parentFile?.mkdirs()
                        var written = 0L
                        out.outputStream().buffered().use { destination ->
                            val buffer = ByteArray(1024 * 1024)
                            while (true) {
                                val count = zip.read(buffer)
                                if (count < 0) break
                                written = Math.addExact(written, count.toLong())
                                require(written <= expectedFile.size) {
                                    "${entry.name} exceeds its declared size"
                                }
                                destination.write(buffer, 0, count)
                            }
                        }
                        require(written == expectedFile.size) { "${entry.name} is truncated" }
                        require(sha256(out) == expectedFile.sha256) { "${entry.name} verification failed" }
                    }
                    require(expected.isEmpty()) { "model package is missing ${expected.keys.joinToString()}" }
                    return fromManifest(unpack,manifest)
                }
            } catch (error: Throwable) {
                unpack.deleteRecursively()
                throw error
            }
        }

        private fun sha256(file: File): String {
            return file.inputStream().use(::sha256)
        }

        private fun rejectUnsupportedLegacyModel(manifest: JSONObject) {
            val role = manifest.optString("artifact_role", "combined")
            if (role != "model" && role != "combined") return
            val profile = manifest.optString("frontend_profile")
            val bundleId = manifest.optString("bundle_id")
            if (profile == LEGACY_FRONTEND_PROFILE || ":$LEGACY_FRONTEND_PROFILE:" in bundleId) {
                throw UnsupportedLegacyModelException()
            }
        }

        private fun readEntryBytes(input: InputStream, limit: Int): ByteArray {
            val output = ByteArrayOutputStream(minOf(limit, 64 * 1024))
            val buffer = ByteArray(16 * 1024)
            var total = 0
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                total = Math.addExact(total, count)
                require(total <= limit) { "model manifest is too large" }
                output.write(buffer, 0, count)
            }
            return output.toByteArray()
        }

        private fun sha256(input: InputStream): String {
            val digest = MessageDigest.getInstance("SHA-256")
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val n = input.read(buffer); if (n < 0) break
                digest.update(buffer, 0, n)
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }

        private const val MAX_MANIFEST_BYTES = 4 * 1024 * 1024
        private const val MAX_PACKAGE_FILE_BYTES = 16L * 1024 * 1024 * 1024
        private const val MAX_PACKAGE_EXTRACT_BYTES = 32L * 1024 * 1024 * 1024
        private const val MIN_FREE_SPACE_AFTER_IMPORT = 512L * 1024 * 1024
        private const val LEGACY_FRONTEND_PROFILE = "full-g2pw-v2"
    }
}
