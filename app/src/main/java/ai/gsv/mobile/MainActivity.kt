package ai.gsv.mobile

import android.Manifest
import android.media.MediaPlayer
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.ResultReceiver
import android.provider.OpenableColumns
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.animation.animateContentSize
import androidx.compose.foundation.clickable
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.OpenInNew
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.SaveAlt
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.UploadFile
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.res.stringResource
import androidx.lifecycle.lifecycleScope
import androidx.core.content.ContextCompat
import androidx.core.view.WindowInsetsControllerCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject

@OptIn(ExperimentalMaterial3Api::class)
class MainActivity : ComponentActivity() {
    private val engine: TtsEngine get() = GsvRuntime.engine
    private val engineLock: Any get() = GsvRuntime.engineLock
    private var output: File? = null
    private var player: MediaPlayer? = null
    private var pendingQnnModelRecord: ModelRecord? = null
    private var pendingApiPort: Int? = null

    private var ui by mutableStateOf(UiState())

    private val pickCombined = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { rememberAndLoadModel(it, split = false) }
    }
    private val pickPipelines = registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isNotEmpty()) installPipelines(uris)
    }
    private val pickQnnPipelineAttachment = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let(::installQnnPipelineAttachment)
    }
    private val pickModel = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { rememberAndLoadModel(it, split = true) }
    }
    private val pickQnnModelAttachment = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        val record = pendingQnnModelRecord
        pendingQnnModelRecord = null
        if (uri != null && record != null) {
            runCatching { contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) }
            loadQnnModelUri(record, uri, remember = true)
        }
    }
    private val pickReference = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let {
            runCatching { contentResolver.takePersistableUriPermission(it, Intent.FLAG_GRANT_READ_URI_PERMISSION) }
            ui = ui.copy(referenceUri = it, referenceName = displayName(it))
        }
    }
    private val saveOutput = registerForActivityResult(ActivityResultContracts.CreateDocument("audio/wav")) { uri ->
        val source = output
        if (uri != null && source != null) {
            setBusy(getString(R.string.saving_audio))
            lifecycleScope.launch {
                runCatching {
                    withContext(Dispatchers.IO) {
                        require(source.isFile) { getString(R.string.audio_output_missing) }
                        contentResolver.openOutputStream(uri, "w").use { destination ->
                            requireNotNull(destination) { getString(R.string.audio_output_open_failed) }
                            source.inputStream().use { it.copyTo(destination) }
                        }
                    }
                }.onSuccess {
                    ui = ui.copy(busy = false, status = getString(R.string.audio_saved))
                }.onFailure {
                    ui = ui.copy(
                        busy = false,
                        status = getString(R.string.audio_save_failed, it.message.orEmpty()),
                    )
                }
            }
        }
    }
    private val requestExternalRead = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) scanExternalModels(silent = false)
        else ui = ui.copy(status = getString(R.string.external_models_permission_required))
    }
    private val manageExternalStorage = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
        if (hasExternalModelAccess()) scanExternalModels(silent = false)
        else ui = ui.copy(status = getString(R.string.external_models_permission_required))
    }
    private val requestNotifications = registerForActivityResult(ActivityResultContracts.RequestPermission()) {
        pendingApiPort?.let(::startApiService)
        pendingApiPort = null
    }

    override fun attachBaseContext(newBase: Context) {
        super.attachBaseContext(AppLocale.wrap(newBase))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        applySystemBars()
        GsvRuntime.retainActivity()
        val prefs = getSharedPreferences("components", MODE_PRIVATE)
        ModelPackage.restorePipelineArchives(this)
        val loaded = engine.loadedPackage
        val restoredOutput = File(cacheDir, "tts.wav").takeIf {
            savedInstanceState?.getBoolean(STATE_CAN_PLAY) == true && it.isFile
        }
        output = restoredOutput
        ui = restoreSynthesisState(savedInstanceState).copy(
            pipelineInstalled = ModelPackage.hasInstalledPipeline(this),
            installedVersions = ModelPackage.installedPipelineVersions(this),
            qnnPipelines = installedQnnPipelines(),
            selectedPipelineVersion = selectedInstalledPipelineVersion(
                ModelPackage.installedPipelineVersions(this),
            ),
            componentUrl = prefs.getString("pipeline_url_v2pp", null) ?: ComponentVersion.V2PP.defaultUrl,
            showFirstRun = savedInstanceState?.getBoolean(STATE_SHOW_FIRST_RUN)
                ?: !ModelPackage.hasInstalledPipeline(this),
            port = intent.getIntExtra("api_port", 9880).toString(),
            models = readModelRecords(),
            appLanguage = AppLocale.selected(this),
            modelInfo = loaded?.let { "${it.name} / ${it.productVersion.ifBlank { it.version }} / ${it.sampleRate} Hz" }
                ?: getString(R.string.model_not_loaded),
            modelLoaded = loaded != null,
            backend = loaded?.let { engine.backendName } ?: getString(R.string.backend_not_loaded),
            runtimeOptions = loaded?.runtimeOptions ?: emptySet(),
            referenceOverrideSupported = loaded?.referenceInputVersion?.let { it >= 1 } ?: false,
            referenceExactPcm16kSamples = loaded?.referencePcm16kSamples?.takeIf {
                loaded.referenceDurationPolicy == "exact_samples" && it > 0
            },
            canPlay = restoredOutput != null,
            serverEnabled = GsvRuntime.apiEndpoint != null,
            serverStatus = GsvRuntime.apiEndpoint?.let { "$it · POST /v1/audio/speech" }
                ?: GsvRuntime.apiError?.let { getString(R.string.api_start_failed, it) }
                ?: getString(R.string.api_stopped),
        )
        setContent { GsvTheme { MainScreen() } }
        DebugAcceptanceRunner.launch(this, intent)
        if (hasExternalModelAccess()) scanExternalModels(silent = true)
        if (intent.getBooleanExtra("start_api", false)) startApiServer()
        if (!intent.hasExtra("qnn_product_model") && !intent.hasExtra("cpu_product_model")) {
            checkForUpdates(automatic = true)
        }
    }

    private fun applySystemBars() {
        val dark = resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK ==
            Configuration.UI_MODE_NIGHT_YES
        window.statusBarColor = if (dark) android.graphics.Color.rgb(16, 25, 22)
            else android.graphics.Color.rgb(245, 248, 244)
        window.navigationBarColor = if (dark) android.graphics.Color.rgb(16, 25, 22)
            else android.graphics.Color.rgb(245, 248, 244)
        WindowInsetsControllerCompat(window, window.decorView).apply {
            isAppearanceLightStatusBars = !dark
            isAppearanceLightNavigationBars = !dark
        }
    }

    private fun requestExternalModelScan() {
        if (hasExternalModelAccess()) {
            scanExternalModels(silent = false)
            return
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val appSettings = Intent(
                Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                Uri.parse("package:$packageName"),
            )
            runCatching { manageExternalStorage.launch(appSettings) }
                .onFailure { manageExternalStorage.launch(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)) }
        } else {
            requestExternalRead.launch(Manifest.permission.READ_EXTERNAL_STORAGE)
        }
    }

    private fun hasExternalModelAccess(): Boolean = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
        Environment.isExternalStorageManager()
    } else {
        checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
    }

    private fun scanExternalModels(silent: Boolean) {
        lifecycleScope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    val directory = externalModelDirectory()
                    require(directory.exists() || directory.mkdirs()) { "directory could not be created" }
                    val records = directory.listFiles()
                        .orEmpty()
                        .filter {
                            it.isFile &&
                                it.name.endsWith(".gsvm", ignoreCase = true) &&
                                !it.name.endsWith(QNN_ATTACHMENT_SUFFIX, ignoreCase = true)
                        }
                        .sortedBy { it.name.lowercase() }
                        .mapNotNull(::readExternalModelRecord)
                    directory to records
                }
            }.onSuccess { (directory, discovered) ->
                val directoryUri = Uri.fromFile(directory).toString().trimEnd('/') + "/"
                val existing = ui.models.associateBy(ModelRecord::uri)
                val refreshed = discovered.map { record ->
                    existing[record.uri]?.let { previous ->
                        record.copy(
                            qnnUri = previous.qnnUri,
                            baseModelSha256 = previous.baseModelSha256,
                        )
                    } ?: record
                }
                val retained = ui.models.filterNot { it.uri.startsWith(directoryUri) }
                writeModelRecords((refreshed + retained).distinctBy { it.uri })
                if (!silent || discovered.isNotEmpty()) {
                    ui = ui.copy(status = getString(R.string.external_models_scanned, discovered.size))
                }
            }.onFailure { error ->
                if (!silent) ui = ui.copy(
                    status = getString(R.string.external_models_scan_failed, error.message.orEmpty()),
                )
            }
        }
    }

    private fun readExternalModelRecord(file: File): ModelRecord? = runCatching {
        val manifest = file.inputStream().use(ModelPackage::inspectManifest)
        val role = manifest.optString("artifact_role", "combined")
        if (role == "pipeline" || role.startsWith("qnn-") || manifest.optString("executor") == "qnn-htp") {
            return@runCatching null
        }
        ModelRecord(
            uri = Uri.fromFile(file).toString(),
            name = manifest.getString("name"),
            version = manifest.getString("model_version"),
            split = role == "model",
            productVersion = manifest.optString("product_version", ""),
        )
    }.getOrNull()

    private fun externalModelDirectory(): File = File(Environment.getExternalStorageDirectory(), "models/gs")

    private fun installPipelines(uris: List<Uri>) {
        setBusy(getString(R.string.pipeline_installing))
        lifecycleScope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    val selectedIds = ui.selectedVersions.map { it.manifestId }.toSet()
                    uris.map { uri ->
                        val version = inspectPipelineVersion(uri)
                        require(version in selectedIds) {
                            getString(R.string.pipeline_version_not_selected, version)
                        }
                        ModelPackage.installPipeline(this@MainActivity, uri, version)
                    }
                }
            }
                .onSuccess {
                    val installed = ModelPackage.installedPipelineVersions(this@MainActivity)
                    ui = ui.copy(
                        busy = false,
                        pipelineInstalled = installed.isNotEmpty(),
                        installedVersions = installed,
                        selectedPipelineVersion = selectedInstalledPipelineVersion(installed),
                        showFirstRun = false,
                        status = getString(R.string.pipeline_installed_versions, it.joinToString { item -> item.version }),
                    )
                }
                .onFailure { ui = ui.copy(busy = false, status = getString(R.string.pipeline_import_failed, it.message.orEmpty())) }
        }
    }

    private fun installQnnPipelineAttachment(uri: Uri) {
        runCatching { contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) }
        setBusy(getString(R.string.qnn_pipeline_installing))
        lifecycleScope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    val version = inspectPipelineVersion(uri)
                    require(version == ui.selectedPipelineVersion) {
                        getString(R.string.pipeline_attachment_version_mismatch, version)
                    }
                    ModelPackage.installQnnPipelineAttachment(this@MainActivity, uri, version)
                }
            }.onSuccess { attachment ->
                ui = ui.copy(
                    busy = false,
                    qnnPipelines = installedQnnPipelines(),
                    selectedPipelineVersion = attachment.version,
                    status = getString(
                        R.string.qnn_pipeline_installed,
                        attachment.version,
                        attachment.targetSoc,
                    ),
                )
            }.onFailure { error ->
                ui = ui.copy(
                    busy = false,
                    status = getString(R.string.qnn_pipeline_import_failed, error.message.orEmpty()),
                )
            }
        }
    }

    private fun installedQnnPipelines(): Set<String> = ComponentVersion.entries.flatMapTo(linkedSetOf()) { version ->
        ModelPackage.installedQnnPipelineTargets(this, version.manifestId).map { target ->
            "${version.manifestId}:$target"
        }
    }

    private fun selectedInstalledPipelineVersion(installed: Set<String>): String? {
        val saved = getSharedPreferences("components", MODE_PRIVATE).getString("selected_pipeline_version", null)
        return saved?.takeIf { it in installed }
            ?: ComponentVersion.entries.firstOrNull { it.manifestId in installed }?.manifestId
    }

    private fun selectInstalledPipeline(version: ComponentVersion) {
        if (version.manifestId !in ui.installedVersions || ui.busy) return
        getSharedPreferences("components", MODE_PRIVATE)
            .edit().putString("selected_pipeline_version", version.manifestId).apply()
        ui = ui.copy(selectedPipelineVersion = version.manifestId, status = "")
    }

    private fun hasQnnPipeline(version: ComponentVersion): Boolean =
        ui.qnnPipelines.any { it.substringBefore(':') == version.manifestId }

    private fun inspectPipelineVersion(uri: Uri): String = contentResolver.openInputStream(uri).use { raw ->
        requireNotNull(raw) { getString(R.string.pipeline_open_failed) }
        ModelPackage.inspectManifest(raw).getString("model_version")
    }

    private fun selectComponentVersion(version: ComponentVersion) {
        val prefs = getSharedPreferences("components", MODE_PRIVATE)
        ui = ui.copy(
            componentVersion = version,
            componentUrl = prefs.getString(version.preferenceKey, null) ?: version.defaultUrl,
        )
    }

    private fun downloadPipelines() {
        val versions = ui.selectedVersions.sortedBy { it.ordinal }
        if (versions.isEmpty()) { ui = ui.copy(status = getString(R.string.pipeline_at_least_one)); return }
        setBusy(getString(R.string.pipeline_preparing_download))
        ui = ui.copy(downloadProgress = 0f)
        lifecycleScope.launch {
            runCatching {
                versions.forEachIndexed { index, version ->
                    withContext(Dispatchers.IO) {
                        val prefs = getSharedPreferences("components", MODE_PRIVATE)
                        val requestedUrl = (prefs.getString(version.preferenceKey, null) ?: version.defaultUrl).trim()
                            .replace("https://huggingface.co/", "https://hf-mirror.com/")
                        require(requestedUrl.startsWith("https://")) {
                            getString(R.string.component_url_https, version.label)
                        }
                        downloadPipeline(version, requestedUrl, index, versions.size)
                    }
                }
            }.onSuccess {
                val installed = ModelPackage.installedPipelineVersions(this@MainActivity)
                ui = ui.copy(
                    busy = false,
                    downloadProgress = null,
                    pipelineInstalled = installed.isNotEmpty(),
                    installedVersions = installed,
                    selectedPipelineVersion = selectedInstalledPipelineVersion(installed),
                    showFirstRun = false,
                    status = getString(R.string.pipeline_install_complete, versions.joinToString { it.label }),
                )
            }.onFailure {
                val installed = ModelPackage.installedPipelineVersions(this@MainActivity)
                ui = ui.copy(busy = false, downloadProgress = null, pipelineInstalled = installed.isNotEmpty(), installedVersions = installed, status = getString(R.string.pipeline_download_failed, it.message.orEmpty()))
            }
        }
    }

    private fun installRecommendedFirefly() {
        if (!hasExternalModelAccess()) {
            ui = ui.copy(status = getString(R.string.external_models_permission_required))
            requestExternalModelScan()
            return
        }
        val target = QualcommTargetSoc.detect()
        if (target == null) {
            ui = ui.copy(status = getString(R.string.recommended_qnn_unsupported_device))
            return
        }
        setBusy(getString(R.string.recommended_install_preparing, target.displayName))
        ui = ui.copy(downloadProgress = 0f)
        lifecycleScope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    val directory = externalModelDirectory().also { require(it.exists() || it.mkdirs()) }
                    val artifacts = listOf(
                        ReleaseArtifact.CPU_PIPELINE,
                        ReleaseArtifact.CPU_FIREFLY,
                        ReleaseArtifact.qnnPipeline(target),
                        ReleaseArtifact.qnnFirefly(target),
                    )
                    val files = artifacts.mapIndexed { index, artifact ->
                        downloadReleaseArtifact(directory, artifact, index, artifacts.size)
                    }
                    val cpuPipeline = files[0]
                    val cpuModel = files[1]
                    val qnnPipeline = files[2]
                    val qnnModel = files[3]
                    ModelPackage.installPipeline(this@MainActivity, Uri.fromFile(cpuPipeline), ComponentVersion.V2PP.manifestId)
                    val cpu = ModelPackage.importModelWithInstalledPipeline(this@MainActivity, Uri.fromFile(cpuModel))
                    ModelPackage.installQnnPipelineAttachment(this@MainActivity, Uri.fromFile(qnnPipeline), ComponentVersion.V2PP.manifestId)
                    val paired = ModelPackage.importQnnModelWithInstalledPipeline(
                        this@MainActivity,
                        Uri.fromFile(cpuModel),
                        Uri.fromFile(qnnModel),
                    )
                    synchronized(engineLock) { engine.load(paired) }
                    cpu to paired
                }
            }.onSuccess { (cpu, paired) ->
                val record = ModelRecord(
                    uri = Uri.fromFile(File(externalModelDirectory(), ReleaseArtifact.CPU_FIREFLY.fileName)).toString(),
                    name = cpu.name,
                    version = cpu.version,
                    split = true,
                    qnnUri = Uri.fromFile(File(externalModelDirectory(), ReleaseArtifact.qnnFirefly(target).fileName)).toString(),
                    baseModelSha256 = paired.baseModelSha256,
                )
                saveModelRecord(record)
                val installed = ModelPackage.installedPipelineVersions(this@MainActivity)
                ui = ui.copy(
                    busy = false,
                    downloadProgress = null,
                    pipelineInstalled = installed.isNotEmpty(),
                    installedVersions = installed,
                    qnnPipelines = installedQnnPipelines(),
                    selectedPipelineVersion = ComponentVersion.V2PP.manifestId,
                    modelInfo = "${paired.name} / ${paired.productVersion.ifBlank { paired.version }} / ${paired.sampleRate} Hz",
                    modelLoaded = true,
                    backend = engine.backendName,
                    runtimeOptions = engine.loadedPackage?.runtimeOptions ?: emptySet(),
                    referenceOverrideSupported = engine.loadedPackage?.referenceInputVersion?.let { it >= 1 } ?: false,
                    status = getString(R.string.recommended_install_complete, target.displayName),
                )
            }.onFailure { error ->
                ui = ui.copy(
                    busy = false,
                    downloadProgress = null,
                    status = getString(R.string.recommended_install_failed, error.message.orEmpty()),
                )
            }
        }
    }

    private suspend fun downloadReleaseArtifact(
        directory: File,
        artifact: ReleaseArtifact,
        index: Int,
        count: Int,
    ): File {
        val destination = File(directory, artifact.fileName)
        if (destination.isFile && fileSha256(destination) == artifact.sha256) return destination
        val partial = File(directory, "${artifact.fileName}.partial")
        try {
            val connection = URL(artifact.url).openConnection() as HttpURLConnection
            connection.instanceFollowRedirects = true
            connection.connectTimeout = 20_000
            connection.readTimeout = 60_000
            connection.setRequestProperty("User-Agent", "GSV-Mobile/3")
            try {
                require(connection.responseCode in 200..299) { getString(R.string.download_http_failed, connection.responseCode) }
                val total = connection.contentLengthLong
                connection.inputStream.buffered().use { input ->
                    partial.outputStream().buffered().use { output ->
                        val buffer = ByteArray(1024 * 1024)
                        var received = 0L
                        while (true) {
                            val read = input.read(buffer)
                            if (read < 0) break
                            output.write(buffer, 0, read)
                            received += read
                            if (total > 0) withContext(Dispatchers.Main) {
                                ui = ui.copy(
                                    downloadProgress = (index + received.toFloat() / total) / count,
                                    status = getString(R.string.recommended_downloading, artifact.fileName, received / 1024 / 1024, total / 1024 / 1024),
                                )
                            }
                        }
                    }
                }
            } finally {
                connection.disconnect()
            }
            require(fileSha256(partial) == artifact.sha256) { getString(R.string.download_sha_mismatch) }
            if (destination.exists()) require(destination.delete()) { getString(R.string.download_persist_failed) }
            require(partial.renameTo(destination)) { getString(R.string.download_persist_failed) }
            return destination
        } finally {
            if (partial.exists()) partial.delete()
        }
    }

    private suspend fun downloadPipeline(version: ComponentVersion, requestedUrl: String, index: Int, count: Int) {
        val archiveDir = File(filesDir, "components").also { it.mkdirs() }
        val archive = File(archiveDir, "pipeline-${version.manifestId}.gsvm")
        val download = File(archiveDir, "pipeline-${version.manifestId}.gsvm.partial")
        try {
                    if (archive.isFile && fileSha256(archive) == version.sha256) {
                        ModelPackage.installPipeline(this@MainActivity, Uri.fromFile(archive), version.manifestId)
                        withContext(Dispatchers.Main) { ui = ui.copy(status = getString(R.string.pipeline_restored, version.label)) }
                        return
                    }
                    val connection = URL(requestedUrl).openConnection() as HttpURLConnection
                    connection.instanceFollowRedirects = true
                    connection.connectTimeout = 20_000
                    connection.readTimeout = 60_000
                    connection.setRequestProperty("User-Agent", "GSV-Mobile/0.1")
                    try {
                        require(connection.responseCode in 200..299) {
                            getString(R.string.download_http_failed, connection.responseCode)
                        }
                        val total = connection.contentLengthLong
                        connection.inputStream.buffered().use { input ->
                            download.outputStream().buffered().use { output ->
                                val buffer = ByteArray(1024 * 1024)
                                var received = 0L
                                while (true) {
                                    val read = input.read(buffer)
                                    if (read < 0) break
                                    output.write(buffer, 0, read)
                                    received += read
                                    if (total > 0) withContext(Dispatchers.Main) {
                                        val fileProgress = received.toFloat() / total.toFloat()
                                        ui = ui.copy(downloadProgress = (index + fileProgress) / count, status = getString(R.string.pipeline_downloading, version.label, received / 1024 / 1024, total / 1024 / 1024))
                                    }
                                }
                            }
                        }
                    } finally { connection.disconnect() }
                    require(fileSha256(download) == version.sha256) {
                        getString(R.string.download_sha_mismatch)
                    }
                    ModelPackage.installPipeline(
                        this@MainActivity, Uri.fromFile(download), version.manifestId,
                    )
                    if (archive.exists()) archive.delete()
                    require(download.renameTo(archive)) {
                        getString(R.string.download_persist_failed)
                    }
        } finally { if (download.exists()) download.delete() }
    }

    private fun fileSha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) { val count = input.read(buffer); if (count < 0) break; digest.update(buffer, 0, count) }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun rememberAndLoadModel(uri: Uri, split: Boolean) {
        runCatching {
            contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        loadModelUri(uri, split, remember = true)
    }

    private fun loadModelUri(uri: Uri, split: Boolean, remember: Boolean) {
        if (!modelUriExists(uri)) {
            removeModelRecord(uri.toString())
            ui = ui.copy(status = getString(R.string.model_missing_removed))
            return
        }
        val message = getString(if (split) R.string.model_loading else R.string.model_importing)
        val selectedPipelineVersion = ui.selectedPipelineVersion
        loadPackage(message, importer = {
            if (split && selectedPipelineVersion != null) {
                val version = contentResolver.openInputStream(uri).use { input ->
                    requireNotNull(input) { getString(R.string.pipeline_open_failed) }
                    ModelPackage.inspectManifest(input).getString("model_version")
                }
                require(version == selectedPipelineVersion) {
                    getString(R.string.voice_model_pipeline_mismatch, version)
                }
            }
            if (split) ModelPackage.importModelWithInstalledPipeline(this, uri) else ModelPackage.import(this, uri)
        }) { model ->
            if (remember) saveModelRecord(ModelRecord(uri.toString(), model.name, model.version, split))
        }
    }

    private fun chooseQnnModelAttachment(record: ModelRecord) {
        pendingQnnModelRecord = record
        pickQnnModelAttachment.launch(qnnPackageTypes)
    }

    private fun loadQnnModelUri(record: ModelRecord, attachment: Uri, remember: Boolean) {
        val base = Uri.parse(record.uri)
        if (remember && !modelUriExists(base)) {
            removeModelRecord(record.uri)
            ui = ui.copy(status = getString(R.string.model_missing_removed))
            return
        }
        loadPackage(getString(R.string.qnn_model_loading), importer = {
            if (remember) {
                ModelPackage.importQnnModelWithInstalledPipeline(this, base, attachment)
            } else {
                // Re-hash the current CPU package before selecting an installed QNN copy. The
                // source file may have been replaced in-place since the model record was saved.
                runCatching {
                    ModelPackage.openInstalledQnnModelWithPipeline(this, base)
                }.getOrElse {
                    require(modelUriExists(base)) { "the installed QNN voice and its base model are unavailable" }
                    ModelPackage.importQnnModelWithInstalledPipeline(this, base, attachment)
                }
            }
        }) { model ->
            if (remember) saveModelRecord(
                record.copy(
                    qnnUri = attachment.toString(),
                    baseModelSha256 = model.baseModelSha256,
                )
            )
        }
    }

    private fun loadModelRecord(record: ModelRecord) {
        ComponentVersion.entries.firstOrNull { it.manifestId == record.version }?.let(::selectInstalledPipeline)
        val qnn = record.qnnUri
        if (qnn.isNullOrBlank()) {
            loadModelUri(Uri.parse(record.uri), record.split, remember = false)
        } else {
            // The verified attachment is copied into app-private storage at install time. Prefer
            // that durable copy so deleting or moving the original download does not disable NPU
            // loading. loadQnnModelUri falls back to the persisted source URI only when the
            // installed copy is absent or belongs to another base model.
            loadQnnModelUri(record, Uri.parse(qnn), remember = false)
        }
    }

    private fun modelUriExists(uri: Uri): Boolean = runCatching {
        contentResolver.openInputStream(uri).use { requireNotNull(it); it.read() }
        true
    }.getOrDefault(false)

    private fun readModelRecords(): List<ModelRecord> {
        val stateFile = File(filesDir, "state/model-records.json")
        val raw = runCatching { stateFile.readText() }.getOrNull()
            ?: getSharedPreferences("models", MODE_PRIVATE).getString("records", "[]") ?: "[]"
        return runCatching {
            val array = JSONArray(raw)
            List(array.length()) { index ->
                array.getJSONObject(index).let {
                    ModelRecord(
                        it.getString("uri"),
                        it.getString("name"),
                        it.getString("version"),
                        it.optBoolean("split", true),
                        it.optString("qnn_uri").takeIf(String::isNotBlank),
                        it.optString("base_model_sha256").takeIf(String::isNotBlank),
                        it.optString("product_version").takeIf(String::isNotBlank).orEmpty(),
                    )
                }
            }
        }.getOrDefault(emptyList())
    }

    private fun writeModelRecords(records: List<ModelRecord>) {
        val array = JSONArray()
        records.forEach { record ->
            array.put(
                JSONObject()
                    .put("uri", record.uri)
                    .put("name", record.name)
                    .put("version", record.version)
                    .put("split", record.split)
                    .put("qnn_uri", record.qnnUri ?: "")
                    .put("base_model_sha256", record.baseModelSha256 ?: "")
                    .put("product_version", record.productVersion),
            )
        }
        getSharedPreferences("models", MODE_PRIVATE).edit().putString("records", array.toString()).apply()
        val stateFile = File(filesDir, "state/model-records.json")
        stateFile.parentFile?.mkdirs()
        val pending = File(stateFile.parentFile, "${stateFile.name}.pending")
        pending.writeText(array.toString())
        if (stateFile.exists()) stateFile.delete()
        require(pending.renameTo(stateFile)) { getString(R.string.model_list_save_failed) }
        ui = ui.copy(models = records)
    }

    private fun saveModelRecord(record: ModelRecord) {
        writeModelRecords(listOf(record) + ui.models.filterNot { it.uri == record.uri })
    }

    private fun removeModelRecord(uri: String) = writeModelRecords(ui.models.filterNot { it.uri == uri })

    private fun loadPackage(message: String, importer: () -> ModelPackage, loaded: (ModelPackage) -> Unit = {}) {
        setBusy(message)
        lifecycleScope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    TimingContext.measure("model.import_and_engine_load") {
                        importer().also { synchronized(engineLock) { engine.load(it) } }
                    }
                }
            }.onSuccess {
                loaded(it)
                ui = ui.copy(
                    busy = false,
                    modelInfo = "${it.name} / ${it.productVersion.ifBlank { it.version }} / ${it.sampleRate} Hz",
                    modelLoaded = true,
                    backend = engine.backendName,
                    runtimeOptions = it.runtimeOptions,
                    referenceOverrideSupported = it.referenceInputVersion >= 1,
                    referenceExactPcm16kSamples = it.referencePcm16kSamples.takeIf { samples ->
                        it.referenceDurationPolicy == "exact_samples" && samples > 0
                    },
                    temperature = if ("temperature" in it.runtimeOptions) ui.temperature else "1.0",
                    topP = if ("top_p" in it.runtimeOptions) ui.topP else "1.0",
                    topK = if ("top_k" in it.runtimeOptions) ui.topK else "10",
                    penalty = if ("repetition_penalty" in it.runtimeOptions) ui.penalty else "1.35",
                    speed = if ("speed_factor" in it.runtimeOptions) ui.speed else "1.0",
                    steps = if ("sample_steps" in it.runtimeOptions) ui.steps else "32",
                    referenceUri = null,
                    referenceName = "",
                    referencePrompt = "",
                    status = getString(R.string.model_loaded),
                )
            }.onFailure { error ->
                ui = ui.copy(
                    busy = false,
                    status = if (error is UnsupportedLegacyModelException) {
                        getString(R.string.legacy_model_unsupported)
                    } else {
                        getString(R.string.import_failed, error.message.orEmpty())
                    },
                )
            }
        }
    }

    private fun synthesize() {
        val state = ui
        if (state.referenceUri != null && state.referencePrompt.isBlank()) {
            ui = ui.copy(status = getString(R.string.reference_prompt_required))
            return
        }
        if (state.referenceUri != null && !state.referenceOverrideSupported) {
            ui = ui.copy(status = getString(R.string.reference_not_supported))
            return
        }
        // A new request invalidates the previous playable result immediately.  This prevents
        // a failed/in-flight synthesis from accidentally playing an older WAV.
        File(cacheDir, "tts.wav").delete()
        output = null
        ui = ui.copy(
            busy = true,
            canPlay = false,
            status = if (state.referenceUri == null) getString(R.string.synthesizing)
            else getString(R.string.reference_decoding),
        )
        lifecycleScope.launch {
            runCatching {
                val options = SynthesisOptions(state.temperature.toFloat(), state.topP.toFloat(), state.topK.toInt(), state.penalty.toFloat(), state.speed.toFloat(), state.steps.toInt())
                val reference = state.referenceUri?.let { uri ->
                    withContext(Dispatchers.IO) {
                        ReferenceAudioDecoder.decode(
                            this@MainActivity,
                            uri,
                            state.referencePrompt,
                            state.referenceLanguage,
                            state.referenceExactPcm16kSamples,
                        )
                    }
                }
                withContext(Dispatchers.Default) {
                    synchronized(engineLock) {
                        engine.synthesize(
                            SynthesisRequest(
                                state.text,
                                language = state.textLanguage,
                                options = options,
                                reference = reference,
                            ),
                            File(cacheDir, "tts.wav"),
                        )
                    }
                }
            }.onSuccess { output = it; ui = ui.copy(busy = false, canPlay = true, status = getString(R.string.synthesis_complete)) }
                .onFailure { error ->
                    val message = if (error is ReferenceDurationMismatch) {
                        getString(
                            R.string.reference_exact_duration_mismatch,
                            error.expectedSeconds,
                            error.actualSeconds,
                        )
                    } else {
                        getString(R.string.synthesis_failed, error.message.orEmpty())
                    }
                    ui = ui.copy(busy = false, status = message)
                }
        }
    }

    private fun play() {
        output?.let { file -> player?.release(); player = MediaPlayer().apply { setDataSource(file.path); prepare(); start() } }
    }

    private fun save() {
        if (output?.isFile != true) {
            ui = ui.copy(status = getString(R.string.audio_output_missing))
            return
        }
        saveOutput.launch("gsv-${System.currentTimeMillis()}.wav")
    }

    private fun startApiServer() {
        val port = ui.port.toIntOrNull()
        if (port == null || port !in 1024..65535) { ui = ui.copy(serverEnabled = false, serverStatus = getString(R.string.api_port_invalid)); return }
        if (
            Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            pendingApiPort = port
            ui = ui.copy(serverEnabled = true, serverStatus = getString(R.string.api_starting))
            requestNotifications.launch(Manifest.permission.POST_NOTIFICATIONS)
            return
        }
        startApiService(port)
    }

    private fun startApiService(port: Int) {
        val receiver = object : ResultReceiver(Handler(mainLooper)) {
            override fun onReceiveResult(resultCode: Int, resultData: Bundle?) {
                when (resultCode) {
                    LocalOpenAiService.RESULT_STARTED -> {
                        val endpoint = resultData?.getString(LocalOpenAiService.EXTRA_ENDPOINT).orEmpty()
                        ui = ui.copy(
                            serverEnabled = true,
                            serverStatus = "$endpoint · POST /v1/audio/speech",
                        )
                    }
                    LocalOpenAiService.RESULT_ERROR -> {
                        val error = resultData?.getString(LocalOpenAiService.EXTRA_ERROR).orEmpty()
                        ui = ui.copy(
                            serverEnabled = false,
                            serverStatus = getString(R.string.api_start_failed, error),
                        )
                    }
                }
            }
        }
        ui = ui.copy(serverEnabled = true, serverStatus = getString(R.string.api_starting))
        runCatching {
            ContextCompat.startForegroundService(
                this,
                LocalOpenAiService.startIntent(this, port, receiver),
            )
        }.onFailure {
            ui = ui.copy(
                serverEnabled = false,
                serverStatus = getString(R.string.api_start_failed, it.message.orEmpty()),
            )
        }
    }

    private fun stopApiServer() {
        stopService(LocalOpenAiService.stopIntent(this))
        GsvRuntime.apiEndpoint = null
        GsvRuntime.apiError = null
        ui = ui.copy(serverEnabled = false, serverStatus = getString(R.string.api_stopped))
    }
    private fun setBusy(status: String) { ui = ui.copy(busy = true, status = status) }
    private fun openWebsite(url: String) = startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))

    private fun checkForUpdates(automatic: Boolean) {
        if (ui.updateChecking) return
        ui = ui.copy(updateChecking = true, updateStatus = if (automatic) "" else getString(R.string.update_checking))
        lifecycleScope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    ReleaseUpdateChecker.check(BuildConfig.VERSION_NAME)
                }
            }.onSuccess { release ->
                val dismissed = getSharedPreferences("updates", MODE_PRIVATE)
                    .getString("dismissed_tag", null)
                ui = ui.copy(
                    updateChecking = false,
                    releaseCheck = release,
                    updateStatus = getString(
                        if (release.isNewer) R.string.update_available_version else R.string.update_up_to_date,
                        release.latestTag,
                    ),
                    showUpdatePrompt = release.isNewer && (!automatic || dismissed != release.latestTag),
                )
            }.onFailure { error ->
                ui = ui.copy(
                    updateChecking = false,
                    updateStatus = getString(R.string.update_check_failed, error.message.orEmpty()),
                )
            }
        }
    }

    private fun dismissUpdatePrompt() {
        ui.releaseCheck?.takeIf { it.isNewer }?.let { release ->
            getSharedPreferences("updates", MODE_PRIVATE).edit()
                .putString("dismissed_tag", release.latestTag).apply()
        }
        ui = ui.copy(showUpdatePrompt = false)
    }

    private fun displayName(uri: Uri): String {
        val value = contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) cursor.getString(0) else null
        }
        return value ?: uri.lastPathSegment ?: "reference audio"
    }

    private fun selectAppLanguage(language: AppLanguage) {
        if (ui.busy || ui.appLanguage == language) return
        AppLocale.set(this, language)
        recreate()
    }

    private fun restoreSynthesisState(state: Bundle?): UiState {
        if (state == null) return ui
        return ui.copy(
            text = state.getString(STATE_TEXT, ui.text),
            textLanguage = state.getString(STATE_TEXT_LANGUAGE, ui.textLanguage),
            temperature = state.getString(STATE_TEMPERATURE, ui.temperature),
            topP = state.getString(STATE_TOP_P, ui.topP),
            topK = state.getString(STATE_TOP_K, ui.topK),
            penalty = state.getString(STATE_PENALTY, ui.penalty),
            speed = state.getString(STATE_SPEED, ui.speed),
            steps = state.getString(STATE_STEPS, ui.steps),
            referenceUri = state.getString(STATE_REFERENCE_URI)?.takeIf(String::isNotBlank)?.let(Uri::parse),
            referenceName = state.getString(STATE_REFERENCE_NAME, ui.referenceName),
            referencePrompt = state.getString(STATE_REFERENCE_PROMPT, ui.referencePrompt),
            referenceLanguage = state.getString(STATE_REFERENCE_LANGUAGE, ui.referenceLanguage),
        )
    }

    @Composable private fun MainScreen() {
        GsvAppUi(
            state = ui,
            onStateChange = { ui = it },
            onSynthesize = ::synthesize,
            onPlay = ::play,
            onSave = ::save,
            onPickReference = { pickReference.launch(arrayOf("audio/*", "audio/wav", "audio/flac", "audio/mpeg")) },
            onInstallFirefly = ::installRecommendedFirefly,
            onSelectPipeline = ::selectInstalledPipeline,
            onInstallPipelines = { ui = ui.copy(showFirstRun = true) },
            onPickPipeline = { pickPipelines.launch(packageTypes) },
            onPickQnnPipeline = { pickQnnPipelineAttachment.launch(qnnPackageTypes) },
            onPickModel = { pickModel.launch(packageTypes) },
            onPickCombined = { pickCombined.launch(packageTypes) },
            onScan = ::requestExternalModelScan,
            onChooseQnnModel = ::chooseQnnModelAttachment,
            onLoadModel = ::loadModelRecord,
            onRemoveQnnModel = { saveModelRecord(it.copy(qnnUri = null, baseModelSha256 = null)) },
            onOpenConverter = { openWebsite(MODEL_CONVERTER_URL) },
            onOpenProject = { openWebsite(PROJECT_URL) },
            onOpenUpstream = { openWebsite(UPSTREAM_URL) },
            onSelectLanguage = ::selectAppLanguage,
            onCheckUpdate = { checkForUpdates(automatic = false) },
            onOpenRelease = { ui.releaseCheck?.let { openWebsite(it.openUrl) } },
            onDismissUpdate = ::dismissUpdatePrompt,
            onStartApi = ::startApiServer,
            onStopApi = ::stopApiServer,
            onDownloadPipelines = ::downloadPipelines,
        )
    }

    override fun onDestroy() {
        player?.release()
        GsvRuntime.releaseActivity(isFinishing)
        super.onDestroy()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putString(STATE_TEXT, ui.text)
        outState.putString(STATE_TEXT_LANGUAGE, ui.textLanguage)
        outState.putString(STATE_TEMPERATURE, ui.temperature)
        outState.putString(STATE_TOP_P, ui.topP)
        outState.putString(STATE_TOP_K, ui.topK)
        outState.putString(STATE_PENALTY, ui.penalty)
        outState.putString(STATE_SPEED, ui.speed)
        outState.putString(STATE_STEPS, ui.steps)
        outState.putString(STATE_REFERENCE_URI, ui.referenceUri?.toString())
        outState.putString(STATE_REFERENCE_NAME, ui.referenceName)
        outState.putString(STATE_REFERENCE_PROMPT, ui.referencePrompt)
        outState.putString(STATE_REFERENCE_LANGUAGE, ui.referenceLanguage)
        outState.putBoolean(STATE_CAN_PLAY, ui.canPlay && output?.isFile == true)
        outState.putBoolean(STATE_SHOW_FIRST_RUN, ui.showFirstRun)
        super.onSaveInstanceState(outState)
    }

    companion object {
        private val packageTypes = arrayOf("application/zip", "application/octet-stream")
        private val qnnPackageTypes = arrayOf("application/zip", "application/octet-stream")
        private const val MODEL_CONVERTER_URL = "https://gs.cutefireflyuwu.sbs"
        private const val PROJECT_URL = "https://github.com/tuxKOH/GPT-SoViTs-android"
        private const val UPSTREAM_URL = "https://github.com/RVC-Boss/GPT-SoVITS"
        private const val STATE_TEXT = "synthesis.text"
        private const val STATE_TEXT_LANGUAGE = "synthesis.text_language"
        private const val STATE_TEMPERATURE = "synthesis.temperature"
        private const val STATE_TOP_P = "synthesis.top_p"
        private const val STATE_TOP_K = "synthesis.top_k"
        private const val STATE_PENALTY = "synthesis.penalty"
        private const val STATE_SPEED = "synthesis.speed"
        private const val STATE_STEPS = "synthesis.steps"
        private const val STATE_REFERENCE_URI = "synthesis.reference_uri"
        private const val STATE_REFERENCE_NAME = "synthesis.reference_name"
        private const val STATE_REFERENCE_PROMPT = "synthesis.reference_prompt"
        private const val STATE_REFERENCE_LANGUAGE = "synthesis.reference_language"
        private const val STATE_CAN_PLAY = "synthesis.can_play"
        private const val STATE_SHOW_FIRST_RUN = "components.show_first_run"
    }
}

private const val RELEASE_REPOSITORY = "https://hf-mirror.com/tuxjhtd/GPT-Sovits_pipeline_Torchscript/resolve/main/"

private data class ReleaseArtifact(val fileName: String, val sha256: String) {
    val url: String get() = "$RELEASE_REPOSITORY$fileName?download=true"

    companion object {
        val CPU_PIPELINE = ReleaseArtifact("v2pp-cpu-pipeline.gsvm", "7fcbc0bb520fa9c3eedd1ccccace443ced2185b8ca1ab2c915fd764f5e6656d7")
        val CPU_FIREFLY = ReleaseArtifact("firefly-v2pp-cpu.gsvm", "cc6fb5cdd51e0d4638de1d018a7fcbf642fb0f0b4540770a15f02f167ceeb3af")
        fun qnnPipeline(target: QualcommTargetSoc): ReleaseArtifact = when (target) {
            QualcommTargetSoc.SNAPDRAGON_8_GEN_3 -> ReleaseArtifact("v2pp-sm8650-pipeline.qnn.gsvm", "933af9777eb5095d46b904cfb083922acf76e2e7c645b7533ac295d01c5700c8")
            QualcommTargetSoc.SNAPDRAGON_8_ELITE -> ReleaseArtifact("v2pp-sm8750-pipeline.qnn.gsvm", "d3805f7c8de4db2bb733a5737aea7404d81133ef3d4c8caf1a272f6943e6fe7a")
            QualcommTargetSoc.SNAPDRAGON_8_ELITE_GEN_5 -> ReleaseArtifact("v2pp-sm8850-pipeline.qnn.gsvm", "a90d92bb80cebd276e1f7d42887aefb396549d6579de3f0ff8049a5a345ac172")
        }
        fun qnnFirefly(target: QualcommTargetSoc): ReleaseArtifact = when (target) {
            QualcommTargetSoc.SNAPDRAGON_8_GEN_3 -> ReleaseArtifact("firefly-v2pp-sm8650.qnn.gsvm", "c35484ad63218af892667a9f2a9de7887f884850c5cb05c4ed2ebefea5206f1a")
            QualcommTargetSoc.SNAPDRAGON_8_ELITE -> ReleaseArtifact("firefly-v2pp-sm8750.qnn.gsvm", "8ad26fc196f345b24071ee14e5c3aa1c4494119c407edebdba4cee3ec9d55ea4")
            QualcommTargetSoc.SNAPDRAGON_8_ELITE_GEN_5 -> ReleaseArtifact("firefly-v2pp-sm8850.qnn.gsvm", "b1cdec977f787a0bdc9f9f61f4c83d4754703552430c6ce05c19662a1130beb2")
        }
    }
}

internal data class UiState(
    val pipelineInstalled: Boolean = false, val showFirstRun: Boolean = false, val componentUrl: String = "",
    val componentVersion: ComponentVersion = ComponentVersion.V2PP, val selectedVersions: Set<ComponentVersion> = setOf(ComponentVersion.V2PP),
    val installedVersions: Set<String> = emptySet(), val qnnPipelines: Set<String> = emptySet(), val selectedPipelineVersion: String? = null, val downloadProgress: Float? = null,
    val models: List<ModelRecord> = emptyList(),
    val expandedModelUri: String? = null,
    val modelInfo: String = "", val modelLoaded: Boolean = false, val backend: String = "", val status: String = "",
    val text: String = "", val textLanguage: String = "auto", val temperature: String = "1.0", val topP: String = "1.0", val topK: String = "10",
    val penalty: String = "1.35", val speed: String = "1.0", val steps: String = "32", val busy: Boolean = false, val canPlay: Boolean = false,
    val referenceUri: Uri? = null, val referenceName: String = "", val referencePrompt: String = "",
    val referenceLanguage: String = "auto", val runtimeOptions: Set<String> = emptySet(), val referenceOverrideSupported: Boolean = false,
    val referenceExactPcm16kSamples: Int? = null,
    val port: String = "9880", val serverEnabled: Boolean = false, val serverStatus: String = "",
    val appLanguage: AppLanguage = AppLanguage.ENGLISH,
    val updateChecking: Boolean = false, val releaseCheck: ReleaseCheckResult? = null,
    val updateStatus: String = "", val showUpdatePrompt: Boolean = false,
)

internal data class ModelRecord(
    val uri: String,
    val name: String,
    val version: String,
    val split: Boolean,
    val qnnUri: String? = null,
    val baseModelSha256: String? = null,
    val productVersion: String = "",
)

internal enum class ComponentVersion(
    val label: String,
    val shortLabel: String,
    val manifestId: String,
    val preferenceKey: String,
    val defaultUrl: String,
    val sha256: String,
) {
    // This is the shared, frontend-only V2PP pipeline.  Keep the URL/hash bound to the
    // currently released artifact so a first-run download cannot silently restore the old
    // pre-ABI pipeline-v2pp-cpu-fp32.gsvm package.
    V2PP("V2 Pro Plus", "V2PP", "v2ProPlus", "pipeline_url_v2pp", "https://hf-mirror.com/tuxjhtd/GPT-Sovits_pipeline_Torchscript/resolve/main/v2pp-cpu-pipeline.gsvm?download=true", "7fcbc0bb520fa9c3eedd1ccccace443ced2185b8ca1ab2c915fd764f5e6656d7"),
    V4("V4", "V4", "v4", "pipeline_url_v4", "https://hf-mirror.com/tuxjhtd/GPT-Sovits_pipeline_Torchscript/resolve/main/pipeline-v4-cpu-fp32.gsvm?download=true", "8cfec15b4a4cb0a7b6cac8df52f4f44e3493694d73e0107b363ddd1012d4f3fa"),
}

@Composable private fun GsvTheme(content: @Composable () -> Unit) {
    val colors = if (androidx.compose.foundation.isSystemInDarkTheme()) {
        darkColorScheme(
            primary = Color(0xFF94DFC0), onPrimary = Color(0xFF063C2B),
            primaryContainer = Color(0xFF174D39), onPrimaryContainer = Color(0xFFD7F5E5),
            secondary = Color(0xFFB6D8CB), secondaryContainer = Color(0xFF293D37),
            onSecondaryContainer = Color(0xFFE2F3EA),
            tertiaryContainer = Color(0xFF33443C), onTertiaryContainer = Color(0xFFDCF3E6),
            background = Color(0xFF101916), onBackground = Color(0xFFE7F0EA),
            surface = Color(0xFF18221E), onSurface = Color(0xFFE7F0EA),
            surfaceVariant = Color(0xFF26352E), onSurfaceVariant = Color(0xFFB9CBC0),
            outline = Color(0xFF73897D),
        )
    } else {
        lightColorScheme(
            primary = Color(0xFF176B50), onPrimary = Color.White,
            primaryContainer = Color(0xFFD9F3E5), onPrimaryContainer = Color(0xFF154634),
            secondary = Color(0xFF426E60), secondaryContainer = Color(0xFFE4F1E9),
            onSecondaryContainer = Color(0xFF234B3C),
            tertiaryContainer = Color(0xFFEAF1EA), onTertiaryContainer = Color(0xFF284A39),
            background = Color(0xFFF5F8F4), onBackground = Color(0xFF17231B),
            surface = Color.White, onSurface = Color(0xFF17231B),
            surfaceVariant = Color(0xFFEAF0EA), onSurfaceVariant = Color(0xFF526459),
            outline = Color(0xFF84988A),
        )
    }
    MaterialTheme(colorScheme = colors, typography = Typography(), content = content)
}
