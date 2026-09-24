package ai.gsv.mobile

/** Lightweight UI progress bridge; the backend remains independent of Compose/Android UI. */
data class InferenceProgress(
    val stage: String,
    val current: Int,
    val total: Int,
    val iterationsPerSecond: Double,
)

object InferenceProgressBus {
    @Volatile
    var listener: ((InferenceProgress) -> Unit)? = null

    private val lock = Any()
    private var stage = ""
    private var total = 0
    private var startedNs = 0L

    fun begin(stage: String, total: Int) = synchronized(lock) {
        this.stage = stage
        this.total = total.coerceAtLeast(1)
        startedNs = System.nanoTime()
        listener?.invoke(InferenceProgress(stage, 0, this.total, 0.0))
    }

    fun report(current: Int, total: Int = this.total) {
        val progress = synchronized(lock) {
            val now = System.nanoTime()
            val elapsedSeconds = ((now - startedNs).coerceAtLeast(1L)) / 1_000_000_000.0
            InferenceProgress(stage, current.coerceAtLeast(0), total.coerceAtLeast(1), current / elapsedSeconds)
        }
        listener?.invoke(progress)
    }

    fun clear() = synchronized(lock) {
        stage = ""
        total = 0
        startedNs = 0L
    }
}
