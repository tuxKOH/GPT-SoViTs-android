package ai.gsv.mobile

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ReleaseUpdateCheckerTest {
    private fun release(tag: String) =
        """{"tag_name":"$tag","draft":false,"prerelease":false}"""

    @Test
    fun comparesNumericVersionsInsteadOfStrings() {
        val newer = ReleaseUpdateChecker.check("3.1.0") { release("v3.10.0") }
        assertTrue(newer.isNewer)
        assertFalse(newer.usedMirror)
        assertEquals("https://github.com/tuxKOH/GPT-SoViTs-android/releases/tag/v3.10.0", newer.openUrl)

        val older = ReleaseUpdateChecker.check("3.1.0") { release("v3") }
        assertFalse(older.isNewer)
        val same = ReleaseUpdateChecker.check("3.1.0") { release("v3.1.0") }
        assertFalse(same.isNewer)
    }

    @Test
    fun fallsBackToMirrorIfOfficialApiFails() {
        val calls = mutableListOf<String>()
        val result = ReleaseUpdateChecker.check("3.1.0") { url ->
            calls += url
            if (url == ReleaseUpdateChecker.OFFICIAL_API) error("unreachable")
            release("v3.2.0")
        }
        assertEquals(listOf(ReleaseUpdateChecker.OFFICIAL_API, ReleaseUpdateChecker.MIRROR_API), calls)
        assertTrue(result.isNewer)
        assertTrue(result.usedMirror)
        assertEquals(
            "https://gh-proxy.com/https://github.com/tuxKOH/GPT-SoViTs-android/releases/tag/v3.2.0",
            result.openUrl,
        )
    }

    @Test
    fun malformedOfficialResponseAlsoUsesMirror() {
        val result = ReleaseUpdateChecker.check("3.1.0") { url ->
            if (url == ReleaseUpdateChecker.OFFICIAL_API) "not JSON" else release("v3.1.1")
        }
        assertTrue(result.usedMirror)
        assertEquals("v3.1.1", result.latestTag)
    }
}
