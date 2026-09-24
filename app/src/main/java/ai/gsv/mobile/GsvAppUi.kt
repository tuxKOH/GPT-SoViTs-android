package ai.gsv.mobile

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.SaveAlt
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.UploadFile
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun GsvAppUi(
    state: UiState,
    onStateChange: (UiState) -> Unit,
    onSynthesize: () -> Unit,
    onPlay: () -> Unit,
    onSave: () -> Unit,
    onPickReference: () -> Unit,
    onInstallFirefly: () -> Unit,
    onSelectPipeline: (ComponentVersion) -> Unit,
    onInstallPipelines: () -> Unit,
    onPickPipeline: () -> Unit,
    onPickQnnPipeline: () -> Unit,
    onPickModel: () -> Unit,
    onPickCombined: () -> Unit,
    onScan: () -> Unit,
    onChooseQnnModel: (ModelRecord) -> Unit,
    onLoadModel: (ModelRecord) -> Unit,
    onRemoveQnnModel: (ModelRecord) -> Unit,
    onOpenConverter: () -> Unit,
    onOpenProject: () -> Unit,
    onOpenUpstream: () -> Unit,
    onSelectLanguage: (AppLanguage) -> Unit,
    onCheckUpdate: () -> Unit,
    onOpenRelease: () -> Unit,
    onDismissUpdate: () -> Unit,
    onStartApi: () -> Unit,
    onStopApi: () -> Unit,
    onDownloadPipelines: () -> Unit,
) {
    var tab by rememberSaveable { mutableStateOf(AppTab.SYNTHESIZE) }
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(stringResource(R.string.app_name), style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.primary)
                        Text(stringResource(tab.title), style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.SemiBold)
                    }
                },
                actions = {
                    if (tab == AppTab.SYNTHESIZE) {
                        IconButton(onClick = { tab = AppTab.MODELS }) {
                            Icon(Icons.Default.Folder, stringResource(R.string.tab_models))
                        }
                    }
                },
            )
        },
        bottomBar = {
            Column {
                if (tab == AppTab.SYNTHESIZE) {
                    Surface(shadowElevation = 5.dp) {
                        Button(
                            onClick = if (state.modelLoaded) onSynthesize else ({ tab = AppTab.MODELS }),
                            enabled = !state.busy && (!state.modelLoaded || state.text.isNotBlank()),
                            modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 10.dp)
                                .height(54.dp),
                        ) {
                            Icon(if (state.modelLoaded) Icons.Default.GraphicEq else Icons.Default.Folder, null)
                            Spacer(Modifier.width(9.dp))
                            Text(stringResource(if (!state.modelLoaded) R.string.ui_choose_model
                                else if (state.busy) R.string.processing else R.string.synthesize_speech))
                        }
                    }
                }
                NavigationBar {
                    AppTab.entries.forEach { item ->
                        NavigationBarItem(
                            selected = tab == item,
                            onClick = { tab = item },
                            icon = { Icon(item.icon, null) },
                            label = { Text(stringResource(item.title)) },
                        )
                    }
                }
            }
        },
    ) { padding ->
        when (tab) {
            AppTab.SYNTHESIZE -> SynthesisScreen(
                state, onStateChange, onPlay, onSave, onPickReference,
                onOpenModels = { tab = AppTab.MODELS }, Modifier.padding(padding),
            )
            AppTab.MODELS -> ModelsScreen(
                state, onInstallFirefly, onSelectPipeline, onInstallPipelines, onPickQnnPipeline,
                onPickModel, onPickCombined, onScan, onChooseQnnModel, onLoadModel,
                onRemoveQnnModel, onOpenConverter, Modifier.padding(padding),
            )
            AppTab.SETTINGS -> SettingsScreen(
                state, onStateChange, onInstallPipelines, onPickPipeline, onStartApi, onStopApi,
                onSelectLanguage, onCheckUpdate, onOpenRelease, onOpenProject, onOpenUpstream,
                Modifier.padding(padding),
            )
        }
    }
    if (state.showFirstRun) {
        PipelineDialog(state, onStateChange, onDownloadPipelines, onPickPipeline)
    }
    if (state.showUpdatePrompt && state.releaseCheck?.isNewer == true) {
        AlertDialog(
            onDismissRequest = onDismissUpdate,
            title = { Text(stringResource(R.string.update_prompt_title)) },
            text = {
                Text(stringResource(R.string.update_prompt_message,
                    BuildConfig.VERSION_NAME, state.releaseCheck.latestTag))
            },
            confirmButton = {
                Button(onClick = { onDismissUpdate(); onOpenRelease() }) {
                    Text(stringResource(R.string.update_open_release))
                }
            },
            dismissButton = {
                TextButton(onClick = onDismissUpdate) { Text(stringResource(R.string.later)) }
            },
        )
    }
}

private enum class AppTab(val title: Int, val icon: androidx.compose.ui.graphics.vector.ImageVector) {
    SYNTHESIZE(R.string.tab_synthesis, Icons.Default.GraphicEq),
    MODELS(R.string.tab_models, Icons.Default.Folder),
    SETTINGS(R.string.tab_settings, Icons.Default.Settings),
}

@Composable
private fun SynthesisScreen(
    state: UiState,
    change: (UiState) -> Unit,
    play: () -> Unit,
    save: () -> Unit,
    pickReference: () -> Unit,
    onOpenModels: () -> Unit,
    modifier: Modifier,
) {
    Column(
        modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        ModelHero(state, onOpenModels)
        SectionCard(R.string.ui_text_section, R.string.ui_text_hint, "01") {
            OutlinedTextField(
                value = state.text,
                onValueChange = { change(state.copy(text = it)) },
                label = { Text(stringResource(R.string.input_text)) },
                placeholder = { Text(stringResource(R.string.ui_text_placeholder)) },
                minLines = 5,
                modifier = Modifier.fillMaxWidth(),
            )
            Text(stringResource(R.string.ui_character_count,
                state.text.codePointCount(0, state.text.length)),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            LanguageChoices(R.string.text_language, state.textLanguage) {
                change(state.copy(textLanguage = it))
            }
        }
        SectionCard(R.string.reference_voice, R.string.ui_reference_hint, "02") {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Mic, null, tint = MaterialTheme.colorScheme.primary)
                Spacer(Modifier.width(10.dp))
                Column(Modifier.weight(1f)) {
                    val referenceTitle = if (state.referenceUri == null) {
                        stringResource(R.string.reference_model_preset)
                    } else {
                        stringResource(R.string.reference_temporary, state.referenceName)
                    }
                    Text(referenceTitle, fontWeight = FontWeight.Medium,
                        maxLines = 2, overflow = TextOverflow.Ellipsis)
                    Text(stringResource(if (state.referenceUri == null) R.string.ui_reference_preset_detail
                        else R.string.ui_reference_temporary_detail),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            if (state.referenceUri != null) {
                OutlinedTextField(
                    value = state.referencePrompt,
                    onValueChange = { change(state.copy(referencePrompt = it)) },
                    label = { Text(stringResource(R.string.reference_prompt)) },
                    supportingText = { Text(stringResource(R.string.ui_reference_transcript_hint)) },
                    minLines = 2,
                    modifier = Modifier.fillMaxWidth(),
                )
                LanguageChoices(R.string.reference_language, state.referenceLanguage) {
                    change(state.copy(referenceLanguage = it))
                }
                TextButton(onClick = { change(state.copy(referenceUri = null, referenceName = "", referencePrompt = "")) }) {
                    Text(stringResource(R.string.reference_use_preset))
                }
            } else {
                OutlinedButton(onClick = pickReference,
                    enabled = !state.busy && state.referenceOverrideSupported,
                    modifier = Modifier.fillMaxWidth()) {
                    Icon(Icons.Default.UploadFile, null)
                    Spacer(Modifier.width(8.dp))
                    Text(stringResource(R.string.reference_choose_audio))
                }
                if (state.modelLoaded && !state.referenceOverrideSupported) {
                    Text(stringResource(R.string.ui_reference_unavailable),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
        if (state.canPlay) {
            SectionCard(R.string.ui_result, R.string.ui_result_hint, "03") {
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    Button(onClick = play, enabled = !state.busy, modifier = Modifier.weight(1f)) {
                        Icon(Icons.Default.PlayArrow, null)
                        Spacer(Modifier.width(4.dp))
                        Text(stringResource(R.string.play))
                    }
                    OutlinedButton(onClick = save, enabled = !state.busy, modifier = Modifier.weight(1f)) {
                        Icon(Icons.Default.SaveAlt, null)
                        Spacer(Modifier.width(4.dp))
                        Text(stringResource(R.string.save_audio))
                    }
                }
            }
        }
        AdvancedOptions(state, change)
        ProgressAndStatus(state)
        Spacer(Modifier.height(4.dp))
    }
}

@Composable
private fun ModelHero(state: UiState, onOpenModels: () -> Unit) {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
            Text(stringResource(if (state.modelLoaded) R.string.ui_ready else R.string.ui_setup_needed),
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onPrimaryContainer)
            Text(if (state.modelLoaded) state.modelInfo else stringResource(R.string.model_not_loaded),
                style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold,
                maxLines = 2, overflow = TextOverflow.Ellipsis)
            Text(if (state.modelLoaded) state.backend else stringResource(R.string.ui_setup_hint),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onPrimaryContainer)
            TextButton(onClick = onOpenModels, contentPadding = ButtonDefaults.TextButtonContentPadding) {
                Text(stringResource(if (state.modelLoaded) R.string.ui_change_model else R.string.ui_choose_model))
            }
        }
    }
}

@Composable
private fun AdvancedOptions(state: UiState, change: (UiState) -> Unit) {
    var expanded by rememberSaveable { mutableStateOf(false) }
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.fillMaxWidth()) {
            Row(
                Modifier.fillMaxWidth().clickable { expanded = !expanded }.padding(16.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(Icons.Default.Tune, null, tint = MaterialTheme.colorScheme.primary)
                Spacer(Modifier.width(12.dp))
                Column(Modifier.weight(1f)) {
                    Text(stringResource(R.string.generation_options), fontWeight = FontWeight.SemiBold)
                    Text(stringResource(R.string.ui_advanced_hint), style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Icon(if (expanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore, null)
            }
            if (expanded) {
                HorizontalDivider()
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    OptionPair(state.temperature, R.string.temperature, "temperature" in state.runtimeOptions,
                        { change(state.copy(temperature = it)) }, state.topP, R.string.top_p,
                        "top_p" in state.runtimeOptions, { change(state.copy(topP = it)) })
                    OptionPair(state.topK, R.string.top_k, "top_k" in state.runtimeOptions,
                        { change(state.copy(topK = it)) }, state.penalty, R.string.repetition_penalty,
                        "repetition_penalty" in state.runtimeOptions, { change(state.copy(penalty = it)) })
                    OptionPair(state.steps, R.string.cfm_steps, "sample_steps" in state.runtimeOptions,
                        { change(state.copy(steps = it)) }, state.speed, R.string.speed,
                        "speed_factor" in state.runtimeOptions, { change(state.copy(speed = it)) })
                    Text(stringResource(R.string.ui_options_capability_hint),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
    }
}

@Composable
private fun OptionPair(
    first: String, firstLabel: Int, firstEnabled: Boolean, onFirst: (String) -> Unit,
    second: String, secondLabel: Int, secondEnabled: Boolean, onSecond: (String) -> Unit,
) {
    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        OutlinedTextField(first, onFirst, modifier = Modifier.weight(1f), singleLine = true,
            enabled = firstEnabled, label = { Text(stringResource(firstLabel)) })
        OutlinedTextField(second, onSecond, modifier = Modifier.weight(1f), singleLine = true,
            enabled = secondEnabled, label = { Text(stringResource(secondLabel)) })
    }
}

@Composable
private fun ModelsScreen(
    state: UiState,
    install: () -> Unit,
    select: (ComponentVersion) -> Unit,
    pipelines: () -> Unit,
    qnnPipeline: () -> Unit,
    model: () -> Unit,
    complete: () -> Unit,
    scan: () -> Unit,
    qnnModel: (ModelRecord) -> Unit,
    load: (ModelRecord) -> Unit,
    remove: (ModelRecord) -> Unit,
    converter: () -> Unit,
    modifier: Modifier,
) {
    Column(
        modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp),
    ) {
        if (state.models.isEmpty()) RecommendedSetupCard(state.busy, install)
        SectionHeader(R.string.model_library, R.string.ui_library_hint)
        if (state.models.isEmpty()) EmptyCard(R.string.ui_no_models_hint)
        state.models.forEach { record ->
            ModelLibraryCard(record, state.busy, qnnModel, load, remove)
        }
        PipelineSection(state, select, pipelines, qnnPipeline)
        if (state.models.isNotEmpty()) RecommendedSetupCard(state.busy, install)
        OutlinedCard(modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text(stringResource(R.string.add_models), fontWeight = FontWeight.SemiBold)
                Button(onClick = model, enabled = !state.busy, modifier = Modifier.fillMaxWidth()) {
                    Icon(Icons.Default.Add, null)
                    Spacer(Modifier.width(8.dp))
                    Text(stringResource(R.string.add_voice_model))
                }
                TextButton(onClick = complete, enabled = !state.busy) {
                    Text(stringResource(R.string.add_complete_package))
                }
                TextButton(onClick = scan, enabled = !state.busy) {
                    Icon(Icons.Default.Refresh, null)
                    Spacer(Modifier.width(6.dp))
                    Text(stringResource(R.string.scan_external_models))
                }
                TextButton(onClick = converter) { Text(stringResource(R.string.how_get_gsvm)) }
                Text(stringResource(R.string.external_model_directory),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        ProgressAndStatus(state)
        Spacer(Modifier.height(6.dp))
    }
}

@Composable
private fun RecommendedSetupCard(busy: Boolean, install: () -> Unit) {
    SectionHeader(R.string.ui_quick_start, R.string.ui_quick_start_hint)
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
            Text(stringResource(R.string.firefly_default_name),
                style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Text(stringResource(R.string.recommended_setup_detail),
                style = MaterialTheme.typography.bodyMedium)
            Button(onClick = install, enabled = !busy && QualcommTargetSoc.detect() != null,
                modifier = Modifier.fillMaxWidth()) {
                Icon(Icons.Default.Download, null)
                Spacer(Modifier.width(8.dp))
                Text(stringResource(R.string.install_firefly_npu))
            }
            if (QualcommTargetSoc.detect() == null) {
                Text(stringResource(R.string.recommended_qnn_unsupported_device),
                    color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}

@Composable
private fun PipelineSection(
    state: UiState,
    select: (ComponentVersion) -> Unit,
    pipelines: () -> Unit,
    qnnPipeline: () -> Unit,
) {
    SectionHeader(R.string.pipeline_components, R.string.pipeline_selection_hint)
    if (state.installedVersions.isEmpty()) {
        EmptyCard(R.string.ui_no_pipeline)
    } else {
        ComponentVersion.entries.filter { it.manifestId in state.installedVersions }.forEach { version ->
            val selected = state.selectedPipelineVersion == version.manifestId
            OutlinedCard(
                Modifier.fillMaxWidth().clickable(enabled = !state.busy) { select(version) },
                colors = CardDefaults.outlinedCardColors(
                    containerColor = if (selected) MaterialTheme.colorScheme.primaryContainer
                    else MaterialTheme.colorScheme.surface,
                ),
            ) {
                Row(Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
                    RadioButton(selected = selected, onClick = { select(version) }, enabled = !state.busy)
                    Column(Modifier.weight(1f).padding(start = 8.dp)) {
                        Text(version.label, fontWeight = FontWeight.SemiBold)
                        Text(stringResource(if (state.qnnPipelines.any { it.startsWith(version.manifestId + ":") })
                            R.string.npu_pipeline_ready else R.string.cpu_pipeline_ready),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    if (selected) Icon(Icons.Default.CheckCircle, null,
                        tint = MaterialTheme.colorScheme.primary)
                }
            }
        }
    }
    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        OutlinedButton(onClick = pipelines, enabled = !state.busy, modifier = Modifier.weight(1f)) {
            Text(stringResource(R.string.install_pipeline))
        }
        OutlinedButton(onClick = qnnPipeline,
            enabled = !state.busy && state.selectedPipelineVersion != null,
            modifier = Modifier.weight(1f)) {
            Text(stringResource(R.string.ui_add_npu_pipeline))
        }
    }
}

@Composable
private fun ModelLibraryCard(
    record: ModelRecord,
    busy: Boolean,
    chooseQnn: (ModelRecord) -> Unit,
    load: (ModelRecord) -> Unit,
    removeQnn: (ModelRecord) -> Unit,
) {
    var details by rememberSaveable(record.uri) { mutableStateOf(false) }
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Mic, null, tint = MaterialTheme.colorScheme.primary)
                Spacer(Modifier.width(12.dp))
                Column(Modifier.weight(1f)) {
                    Text(record.name, style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold, maxLines = 2, overflow = TextOverflow.Ellipsis)
                    Text(record.productVersion.ifBlank { record.version }, style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            Text(stringResource(if (record.qnnUri == null) R.string.ui_model_cpu_only else R.string.ui_model_npu_ready),
                color = if (record.qnnUri == null) MaterialTheme.colorScheme.onSurfaceVariant
                    else MaterialTheme.colorScheme.primary,
                style = MaterialTheme.typography.labelMedium)
            Button(onClick = { load(record) }, enabled = !busy, modifier = Modifier.fillMaxWidth()) {
                Text(stringResource(if (record.qnnUri == null) R.string.load_model else R.string.load_model_npu))
            }
            TextButton(onClick = { details = !details }) {
                Text(stringResource(if (details) R.string.collapse else R.string.ui_attachment_details))
                Icon(if (details) Icons.Default.ExpandLess else Icons.Default.ExpandMore, null)
            }
            if (details) {
                HorizontalDivider()
                Text(stringResource(if (record.qnnUri == null) R.string.qnn_attachment_not_selected
                    else R.string.qnn_attachment_selected),
                    style = MaterialTheme.typography.bodySmall)
                OutlinedButton(onClick = { chooseQnn(record) }, enabled = !busy,
                    modifier = Modifier.fillMaxWidth()) {
                    Text(stringResource(R.string.choose_qnn_attachment))
                }
                if (record.qnnUri != null) {
                    TextButton(onClick = { removeQnn(record) }, enabled = !busy) {
                        Text(stringResource(R.string.remove_qnn_attachment))
                    }
                }
            }
        }
    }
}

@Composable
private fun SettingsScreen(
    state: UiState,
    change: (UiState) -> Unit,
    pipelines: () -> Unit,
    pick: () -> Unit,
    start: () -> Unit,
    stop: () -> Unit,
    language: (AppLanguage) -> Unit,
    checkUpdate: () -> Unit,
    openRelease: () -> Unit,
    project: () -> Unit,
    upstream: () -> Unit,
    modifier: Modifier,
) {
    Column(
        modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        SectionHeader(R.string.ui_device_status, R.string.ui_device_status_hint)
        MemoryPanel()
        ReleaseUpdateCard(state, checkUpdate, openRelease)
        SectionCard(R.string.openai_local_api, R.string.ui_api_hint) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(stringResource(if (state.serverEnabled) R.string.api_enabled else R.string.api_stopped),
                        fontWeight = FontWeight.SemiBold)
                    Text(state.serverStatus, style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Switch(checked = state.serverEnabled, onCheckedChange = { if (it) start() else stop() })
            }
            OutlinedTextField(state.port, { change(state.copy(port = it)) },
                label = { Text(stringResource(R.string.port)) }, singleLine = true,
                enabled = !state.serverEnabled, modifier = Modifier.fillMaxWidth())
        }
        SectionCard(R.string.app_language, R.string.ui_language_hint) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(state.appLanguage == AppLanguage.CHINESE,
                    { language(AppLanguage.CHINESE) }, label = { Text(stringResource(R.string.app_language_chinese)) })
                FilterChip(state.appLanguage == AppLanguage.ENGLISH,
                    { language(AppLanguage.ENGLISH) }, label = { Text(stringResource(R.string.app_language_english)) })
            }
        }
        SectionCard(R.string.pipeline_components, R.string.ui_components_hint) {
            OutlinedButton(onClick = pipelines, modifier = Modifier.fillMaxWidth()) {
                Text(stringResource(R.string.change_components))
            }
            TextButton(onClick = pick) { Text(stringResource(R.string.import_manually)) }
        }
        SectionCard(R.string.project_links, R.string.about_description) {
            TextButton(onClick = project) { Text(stringResource(R.string.project_name)) }
            TextButton(onClick = upstream) { Text(stringResource(R.string.upstream_project)) }
            Text(stringResource(R.string.license_notice),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Spacer(Modifier.height(6.dp))
    }
}

@Composable
private fun ReleaseUpdateCard(state: UiState, check: () -> Unit, open: () -> Unit) {
    SectionCard(R.string.update_title, R.string.update_current_version) {
        Text(BuildConfig.VERSION_NAME, style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold)
        if (state.updateStatus.isNotBlank()) {
            Text(state.updateStatus, style = MaterialTheme.typography.bodyMedium,
                color = if (state.releaseCheck?.isNewer == true) MaterialTheme.colorScheme.primary
                else MaterialTheme.colorScheme.onSurfaceVariant)
        }
        if (state.releaseCheck?.usedMirror == true) {
            Text(stringResource(R.string.update_mirror_used),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            OutlinedButton(onClick = check, enabled = !state.updateChecking,
                modifier = Modifier.weight(1f)) {
                Text(stringResource(if (state.updateChecking) R.string.update_checking
                    else R.string.update_check_now))
            }
            if (state.releaseCheck?.isNewer == true) {
                Button(onClick = open, modifier = Modifier.weight(1f)) {
                    Text(stringResource(R.string.update_open_release))
                }
            }
        }
    }
}

@Composable
private fun MemoryPanel() {
    val context = LocalContext.current
    var snapshot by remember { mutableStateOf<AppMemorySnapshot?>(null) }
    LaunchedEffect(context) {
        while (isActive) {
            snapshot = runCatching { withContext(Dispatchers.Default) { AppMemory.read(context) } }.getOrNull()
            delay(3_000)
        }
    }
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.tertiaryContainer),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Memory, null, tint = MaterialTheme.colorScheme.onTertiaryContainer)
                Spacer(Modifier.width(10.dp))
                Text(stringResource(R.string.ui_memory_title),
                    style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            }
            if (snapshot == null) {
                CircularProgressIndicator(modifier = Modifier.size(24.dp))
            } else {
                val current = requireNotNull(snapshot)
                Text(stringResource(R.string.ui_memory_process, current.processPssBytes / 1024L / 1024L),
                    style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                Text(stringResource(R.string.ui_memory_available,
                    gb(current.systemAvailableBytes), gb(current.systemTotalBytes)),
                    style = MaterialTheme.typography.bodyMedium)
                if (current.systemLowMemory) {
                    Text(stringResource(R.string.ui_memory_low), color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.labelMedium)
                }
            }
            Text(stringResource(R.string.ui_memory_scope),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onTertiaryContainer)
        }
    }
}

private fun gb(bytes: Long): String = String.format(Locale.ROOT, "%.1f", bytes / 1_073_741_824.0)

@Composable
private fun PipelineDialog(
    state: UiState,
    change: (UiState) -> Unit,
    download: () -> Unit,
    import: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = { if (!state.busy) change(state.copy(showFirstRun = false)) },
        title = { Text(stringResource(R.string.pipeline_choose_title)) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(stringResource(R.string.ui_first_run_hint))
                ComponentVersion.entries.forEach { version ->
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Checkbox(version in state.selectedVersions, { checked ->
                            change(state.copy(selectedVersions = if (checked)
                                state.selectedVersions + version else state.selectedVersions - version))
                        })
                        Text(version.label)
                    }
                }
                ProgressAndStatus(state)
            }
        },
        confirmButton = {
            Button(onClick = download, enabled = !state.busy && state.selectedVersions.isNotEmpty()) {
                Text(stringResource(R.string.download_component))
            }
        },
        dismissButton = {
            TextButton(onClick = import, enabled = !state.busy) {
                Text(stringResource(R.string.import_manually))
            }
        },
    )
}

@Composable
private fun SectionHeader(title: Int, subtitle: Int) {
    Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
        Text(stringResource(title), style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold)
        Text(stringResource(subtitle), style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun SectionCard(title: Int, subtitle: Int, step: String? = null,
    content: @Composable ColumnScope.() -> Unit) {
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(verticalAlignment = Alignment.Top) {
                if (step != null) {
                    Surface(color = MaterialTheme.colorScheme.primaryContainer,
                        shape = MaterialTheme.shapes.small) {
                        Text(step, Modifier.padding(horizontal = 8.dp, vertical = 5.dp),
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onPrimaryContainer)
                    }
                    Spacer(Modifier.width(10.dp))
                }
                Column {
                    Text(stringResource(title), style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold)
                    Text(stringResource(subtitle), style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            content()
        }
    }
}

@Composable
private fun EmptyCard(message: Int) {
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Text(stringResource(message), Modifier.padding(18.dp),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun LanguageChoices(label: Int, selected: String, select: (String) -> Unit) {
    Text(stringResource(label), style = MaterialTheme.typography.labelLarge)
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        listOf("auto" to R.string.language_auto, "zh" to R.string.language_chinese,
            "en" to R.string.language_english).forEach { (value, title) ->
            FilterChip(selected == value, { select(value) }, label = { Text(stringResource(title)) })
        }
    }
}

@Composable
private fun ProgressAndStatus(state: UiState) {
    if (state.busy) {
        if (state.downloadProgress == null) LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        else LinearProgressIndicator(progress = { state.downloadProgress.coerceIn(0f, 1f) },
            modifier = Modifier.fillMaxWidth())
    }
    if (state.status.isNotBlank()) {
        Text(state.status, style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}
