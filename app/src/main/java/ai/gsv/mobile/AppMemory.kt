package ai.gsv.mobile

import android.app.ActivityManager
import android.content.Context
import android.os.Debug

/** Process PSS is an estimate of app-owned RAM, not a CPU/NPU memory split. */
internal data class AppMemorySnapshot(
    val processPssBytes: Long,
    val systemAvailableBytes: Long,
    val systemTotalBytes: Long,
    val systemLowMemory: Boolean,
)

internal object AppMemory {
    fun read(context: Context): AppMemorySnapshot {
        val process = Debug.MemoryInfo()
        Debug.getMemoryInfo(process)
        val system = ActivityManager.MemoryInfo()
        context.getSystemService(ActivityManager::class.java).getMemoryInfo(system)
        return AppMemorySnapshot(
            processPssBytes = process.totalPss.toLong() * 1024L,
            systemAvailableBytes = system.availMem,
            systemTotalBytes = system.totalMem,
            systemLowMemory = system.lowMemory,
        )
    }

}
