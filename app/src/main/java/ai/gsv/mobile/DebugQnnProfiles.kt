package ai.gsv.mobile

import java.io.File

/** Optional ADB acceptance profiles; normal app sessions do not write debug files beside models. */
internal object DebugQnnProfiles {
    @Volatile var directory: File? = null

    fun prefix(stage: String): String? {
        if (!BuildConfig.DEBUG) return null
        val root = directory ?: return null
        require(root.isDirectory) { "QNN profile directory is missing: $root" }
        val safeStage = stage.replace(Regex("[^A-Za-z0-9_.-]"), "_")
        return File(root, "$safeStage-${System.nanoTime()}").path
    }
}
