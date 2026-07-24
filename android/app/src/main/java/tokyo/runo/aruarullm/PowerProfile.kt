package tokyo.runo.aruarullm

import android.content.Context

/**
 * 3電源プロファイル(open-web-server Android版
 * `tokyo.runo.openwebserver.PowerProfile`と同じ設計パターン、2026-07-24
 * 新規作成)。
 *
 * - [POWER_SAVE] 省電力版: バックグラウンドでの常時稼働を避け、Android
 *   Doze/App Standbyに逆らわない(=`WakeLock`を一切取得しない)。LLM推論
 *   リクエストには「ハードウェアアクセラレーター無効化(CPUのみ)」を
 *   指示するパラメータを付与する。
 * - [NORMAL] 通常版: 上記2つの中間。バランス型(既定値)。
 * - [ALWAYS_ON] 常時電源接続版: 充電器に繋ぎっぱなしの端末向け。
 *   `PARTIAL_WAKE_LOCK`を保持し、LLM推論リクエストには
 *   「ハードウェアアクセラレーター利用」を指示するパラメータを付与する。
 *
 * **正直な開示**: aruaru-llm(Rust側)は本セッション時点でこのパラメータを
 * 受け取っても解釈しない(`src/main.rs`/`src/scoring.rs`にアクセラレーター
 * 選択の実装が無い、`opencuda-bert`はCPUパスのみ)。このAndroid側の
 * ヘッダー付与は将来サーバー側が対応した際に効果を持つ先取り実装であり、
 * 現時点で実際に電力・応答内容へ影響するのはWakeLock有無とポーリング
 * 間隔差のみ(詳細は`../CLAUDE.md`のHANDOFF節参照)。
 */
enum class PowerProfile(val prefValue: String, val label: String, val emoji: String) {
    POWER_SAVE("power_save", "省電力", "🔋⚡️✕"),
    NORMAL("normal", "通常", "⚖️"),
    ALWAYS_ON("always_on", "常時電源接続", "🔌");

    companion object {
        private const val PREFS_NAME = "aruaru_llm_prefs"
        private const val KEY_PROFILE = "power_profile"
        private const val KEY_SERVER_URL = "server_url"

        fun fromPrefValue(value: String?): PowerProfile =
            values().firstOrNull { it.prefValue == value } ?: NORMAL

        fun load(context: Context): PowerProfile {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            return fromPrefValue(prefs.getString(KEY_PROFILE, null))
        }

        fun save(context: Context, profile: PowerProfile) {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            prefs.edit().putString(KEY_PROFILE, profile.prefValue).apply()
        }

        fun loadServerUrl(context: Context): String {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            // 10.0.2.2はAndroidエミュレータからホストPCのlocalhostへ到達する
            // ための既定アドレス(実機の場合はユーザーがLAN上のIP/ドメインに
            // 書き換える想定)。
            return prefs.getString(KEY_SERVER_URL, null) ?: "http://10.0.2.2:8080"
        }

        fun saveServerUrl(context: Context, url: String) {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            prefs.edit().putString(KEY_SERVER_URL, url).apply()
        }
    }
}
