#include <dlfcn.h>
#include <jni.h>

namespace {
using GetThreads = int (*)();
using SetThreads = void (*)(int);

void* pytorchHandle() {
    return dlopen("libpytorch_jni.so", RTLD_NOW | RTLD_NOLOAD);
}
}  // namespace

extern "C" JNIEXPORT jint JNICALL
Java_ai_gsv_mobile_TorchCpuThreads_nativeGet(JNIEnv*, jobject) {
    void* handle = pytorchHandle();
    if (!handle) return -1;
    auto get = reinterpret_cast<GetThreads>(dlsym(handle, "_ZN2at15get_num_threadsEv"));
    const int value = get ? get() : -1;
    dlclose(handle);
    return value;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_ai_gsv_mobile_TorchCpuThreads_nativeSet(JNIEnv*, jobject, jint count) {
    if (count < 1) return JNI_FALSE;
    void* handle = pytorchHandle();
    if (!handle) return JNI_FALSE;
    auto set = reinterpret_cast<SetThreads>(dlsym(handle, "_ZN2at15set_num_threadsEi"));
    if (set) set(count);
    dlclose(handle);
    return set ? JNI_TRUE : JNI_FALSE;
}
