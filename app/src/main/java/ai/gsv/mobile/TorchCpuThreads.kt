package ai.gsv.mobile

import android.util.Log

/** Controls PyTorch's CPU operator workers; it does not change model precision or graph execution. */
internal object TorchCpuThreads {
    init {
        System.loadLibrary("pytorch_jni")
        System.loadLibrary("gsv_cpu_threads")
    }

    private external fun nativeGet(): Int
    private external fun nativeSet(count: Int): Boolean

    fun current(): Int = nativeGet().also { require(it > 0) { "PyTorch CPU thread API is unavailable" } }

    fun set(count: Int) {
        require(count in 1..32) { "CPU threads must be between 1 and 32" }
        require(nativeSet(count)) { "PyTorch CPU thread API is unavailable" }
        check(current() == count) { "PyTorch did not apply CPU thread count $count" }
        Log.i("GSV_CPU_THREADS", "configured=$count")
    }
}
