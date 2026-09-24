package ai.gsv.mobile

import android.os.Process
import android.os.SystemClock
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.atomic.AtomicLong

/**
 * Low-overhead monotonic timing for end-to-end Android measurements.
 *
 * The trace is deliberately independent of model execution. It records where time went without
 * changing tensors, precision, sampling options, scheduling or backend selection. A trace is
 * emitted both to Logcat (tag GSV_TIMING) and, when an output file is supplied, to a JSON sidecar.
 */
class TimingTrace(
    private val name: String,
    output: File? = null,
) {
    private val startNs = SystemClock.elapsedRealtimeNanos()
    private val startProcessCpuMs = Process.getElapsedCpuTime()
    private val wallStartMs = System.currentTimeMillis()
    private val id = "${name}-${TRACE_IDS.incrementAndGet()}"
    private val outputSidecar = output?.let { File("${it.path}.timing.json") }
    private val events = ArrayList<TimingEvent>()
    private var finished = false

    data class TimingEvent(
        val stage: String,
        val startOffsetNs: Long,
        val durationNs: Long,
        val processCpuMs: Long,
        val ok: Boolean,
        val error: String? = null,
        val detail: String? = null,
    )

    /** Records one nested or top-level stage and returns the block result unchanged. */
    fun <T> measure(stage: String, block: () -> T): T {
        val eventStartNs = SystemClock.elapsedRealtimeNanos()
        val eventStartCpuMs = Process.getElapsedCpuTime()
        Log.i(TAG, "trace=$id stage=$stage phase=begin offset_ms=${elapsedMs(eventStartNs)}")
        var ok = false
        var failure: Throwable? = null
        try {
            return block().also { ok = true }
        } catch (error: Throwable) {
            failure = error
            throw error
        } finally {
            record(
                stage,
                eventStartNs,
                SystemClock.elapsedRealtimeNanos(),
                Process.getElapsedCpuTime() - eventStartCpuMs,
                ok,
                failure,
            )
        }
    }

    /** Records an instantaneous marker such as a backend/profile decision. */
    fun mark(stage: String, detail: String? = null) {
        val now = SystemClock.elapsedRealtimeNanos()
        synchronized(events) {
            events += TimingEvent(stage, now - startNs, 0L, 0L, true, detail = detail)
        }
        val suffix = detail?.let { " detail=${sanitize(it)}" } ?: ""
        Log.i(TAG, "trace=$id stage=$stage phase=mark offset_ms=${elapsedMs(now)}$suffix")
    }

    /** Finishes the trace and writes its JSON sidecar when an output path was provided. */
    fun finish(success: Boolean = true, error: Throwable? = null): File? {
        synchronized(events) {
            if (finished) return outputSidecar
            finished = true
        }
        val endNs = SystemClock.elapsedRealtimeNanos()
        val totalNs = endNs - startNs
        val sidecar = outputSidecar?.let { file ->
            runCatching {
                file.parentFile?.mkdirs()
                val root = JSONObject()
                    .put("trace_id", id)
                    .put("name", name)
                    .put("started_epoch_ms", wallStartMs)
                    .put("finished_epoch_ms", System.currentTimeMillis())
                    .put("monotonic_start_ns", startNs)
                    .put("monotonic_end_ns", endNs)
                    .put("total_ms", totalNs / 1_000_000.0)
                    .put("process_cpu_ms", Process.getElapsedCpuTime() - startProcessCpuMs)
                    .put("success", success)
                error?.let { root.put("error", it.message ?: it::class.java.simpleName) }
                val serialized = JSONArray()
                synchronized(events) {
                    events.forEach { event ->
                        serialized.put(
                            JSONObject()
                                .put("stage", event.stage)
                                .put("offset_ns", event.startOffsetNs)
                                .put("duration_ns", event.durationNs)
                                .put("offset_ms", event.startOffsetNs / 1_000_000.0)
                                .put("duration_ms", event.durationNs / 1_000_000.0)
                                .put("process_cpu_ms", event.processCpuMs)
                                .put("ok", event.ok)
                                .apply {
                                    event.error?.let { put("error", it) }
                                    event.detail?.let { put("detail", it) }
                                },
                        )
                    }
                }
                root.put("events", serialized)
                file.writeText(root.toString(2), Charsets.UTF_8)
                file
            }.onFailure { writeError ->
                Log.w(TAG, "trace=$id sidecar_write_failed path=${file.path}: ${writeError.message}")
            }.getOrNull()
        }
        val suffix = sidecar?.let { " json=${it.path}" } ?: ""
        Log.i(
            TAG,
            "trace=$id phase=end success=$success total_ms=${totalNs / 1_000_000.0}$suffix" +
                (error?.let { " error=${sanitize(it.message ?: it::class.java.simpleName)}" } ?: ""),
        )
        return sidecar
    }

    fun sidecarFile(): File? = outputSidecar

    private fun record(
        stage: String,
        eventStartNs: Long,
        eventEndNs: Long,
        processCpuMs: Long,
        ok: Boolean,
        error: Throwable?,
    ) {
        val durationNs = eventEndNs - eventStartNs
        val event = TimingEvent(
            stage = stage,
            startOffsetNs = eventStartNs - startNs,
            durationNs = durationNs,
            processCpuMs = processCpuMs,
            ok = ok,
            error = error?.message ?: error?.let { it::class.java.simpleName },
        )
        synchronized(events) { events += event }
        val suffix = event.error?.let { " error=${sanitize(it)}" } ?: ""
        Log.i(
            TAG,
            "trace=$id stage=$stage phase=end offset_ms=${elapsedMs(eventStartNs)} " +
                "duration_ms=${durationNs / 1_000_000.0} process_cpu_ms=$processCpuMs ok=$ok$suffix",
        )
    }

    private fun elapsedMs(atNs: Long): Double = (atNs - startNs) / 1_000_000.0

    private fun sanitize(value: String): String = value.replace('\n', ' ').replace('\r', ' ')

    companion object {
        const val TAG = "GSV_TIMING"
        private val TRACE_IDS = AtomicLong(0L)
    }
}

/** Associates lower-level model stages with the currently active synthesis/acceptance trace. */
object TimingContext {
    private val active = ThreadLocal<TimingTrace?>()

    fun <T> with(trace: TimingTrace, block: () -> T): T {
        val previous = active.get()
        active.set(trace)
        return try {
            block()
        } finally {
            active.set(previous)
        }
    }

    fun <T> measure(stage: String, block: () -> T): T = active.get()?.measure(stage, block) ?: run {
        val startNs = SystemClock.elapsedRealtimeNanos()
        var ok = false
        try {
            return block().also { ok = true }
        } finally {
            Log.i(TAG, "stage=$stage duration_ms=${(SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000.0} ok=$ok")
        }
    }

    fun mark(stage: String, detail: String? = null) = active.get()?.mark(stage, detail)

    private const val TAG = TimingTrace.TAG
}
