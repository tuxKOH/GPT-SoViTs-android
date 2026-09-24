plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

val acceptanceAbi = providers.gradleProperty("acceptanceAbi").orNull
val qnnRuntimeVersion = "2.48.0"

android {
    namespace = "ai.gsv.mobile"
    compileSdk = 34

    defaultConfig {
        applicationId = "ai.gsv.mobile"
        minSdk = 26
        targetSdk = 34
        versionCode = 8
        versionName = "3.2.0"
        if (acceptanceAbi != null) {
            ndk { abiFilters += acceptanceAbi }
        }
    }

    buildFeatures { compose = true; buildConfig = true }
    externalNativeBuild {
        cmake { path = file("src/main/cpp/CMakeLists.txt") }
    }
    bundle { language { enableSplit = false } }
    packaging {
        jniLibs {
            useLegacyPackaging = true
            excludes += setOf(
                "**/libQnnGpu.so",
                "**/libQnnDsp.so",
                "**/libQnnDspV66Skel.so",
                "**/libQnnDspV66Stub.so",
                "**/libQnnHtpV68Skel.so",
                "**/libQnnHtpV68Stub.so",
                "**/libQnnHtpV69Skel.so",
                "**/libQnnHtpV69Stub.so",
                "**/libQnnHtpV73Skel.so",
                "**/libQnnHtpV73Stub.so",
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation(platform("androidx.compose:compose-bom:2024.09.03"))
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-core")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("org.nanohttpd:nanohttpd:2.3.1")
    implementation("org.pytorch:pytorch_android:2.1.0")
    // QNN is the first-class ONNX Runtime build for the NPU path.  It brings the
    // QNN execution provider. Pin the native runtime to the QAIRT SDK generation used by
    // tools/build_qnn_htp_context.py instead of accepting ORT's older transitive default.
    implementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.28.0")
    implementation("com.qualcomm.qti:qnn-runtime:$qnnRuntimeVersion")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
}
