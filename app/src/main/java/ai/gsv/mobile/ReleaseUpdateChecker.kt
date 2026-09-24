package ai.gsv.mobile

import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

internal data class ReleaseCheckResult(
    val latestTag: String,
    val isNewer: Boolean,
    val openUrl: String,
    val usedMirror: Boolean,
)

/** Checks the stable GitHub Release, falling back to a mirror only on fetch/parse failure. */
internal object ReleaseUpdateChecker {
    const val OFFICIAL_API =
        "https://api.github.com/repos/tuxKOH/GPT-SoViTs-android/releases/latest"
    const val MIRROR_API = "https://gh-proxy.com/$OFFICIAL_API"
    private const val RELEASE_PAGE =
        "https://github.com/tuxKOH/GPT-SoViTs-android/releases/tag/"
    private const val MIRROR_PREFIX = "https://gh-proxy.com/"
    private val tagPattern = Regex("[A-Za-z0-9._-]{1,80}")
    private val versionPattern = Regex("[vV]?(\\d+)(?:\\.(\\d+))?(?:\\.(\\d+))?(?:[-+].*)?")

    fun check(
        currentVersion: String,
        fetch: (String) -> String = ::download,
    ): ReleaseCheckResult {
        val official = runCatching { parse(fetch(OFFICIAL_API), currentVersion, false) }
        return official.getOrElse { primaryError ->
            runCatching { parse(fetch(MIRROR_API), currentVersion, true) }
                .getOrElse { mirrorError ->
                    throw IllegalStateException(
                        "GitHub: ${primaryError.message}; mirror: ${mirrorError.message}", mirrorError,
                    )
                }
        }
    }

    private fun parse(body: String, currentVersion: String, mirror: Boolean): ReleaseCheckResult {
        val json = JSONObject(body)
        require(!json.optBoolean("draft") && !json.optBoolean("prerelease")) {
            "latest release is not a stable publication"
        }
        val tag = json.getString("tag_name")
        require(tagPattern.matches(tag)) { "release tag is invalid" }
        val latest = requireNotNull(versionParts(tag)) { "release tag is not a version" }
        val current = requireNotNull(versionParts(currentVersion)) { "installed app version is not comparable" }
        val releaseUrl = "$RELEASE_PAGE$tag"
        return ReleaseCheckResult(
            latestTag = tag,
            isNewer = compareValuesBy(latest, current, { it[0] }, { it[1] }, { it[2] }) > 0,
            openUrl = if (mirror) "$MIRROR_PREFIX$releaseUrl" else releaseUrl,
            usedMirror = mirror,
        )
    }

    private fun versionParts(version: String): IntArray? {
        val match = versionPattern.matchEntire(version.trim()) ?: return null
        return runCatching {
            intArrayOf(
                match.groupValues[1].toInt(),
                match.groupValues[2].ifBlank { "0" }.toInt(),
                match.groupValues[3].ifBlank { "0" }.toInt(),
            )
        }.getOrNull()
    }

    private fun download(address: String): String {
        val connection = URL(address).openConnection() as HttpURLConnection
        connection.instanceFollowRedirects = true
        connection.connectTimeout = 8_000
        connection.readTimeout = 8_000
        connection.setRequestProperty("Accept", "application/vnd.github+json")
        connection.setRequestProperty("User-Agent", "GSV-Mobile/${BuildConfig.VERSION_NAME}")
        try {
            require(connection.responseCode in 200..299) {
                "HTTP ${connection.responseCode}"
            }
            val bytes = connection.inputStream.use { input ->
                input.readNBytes(256 * 1024 + 1)
            }
            require(bytes.size <= 256 * 1024) { "release response is too large" }
            return bytes.toString(Charsets.UTF_8)
        } finally {
            connection.disconnect()
        }
    }
}
