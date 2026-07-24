package tokyo.runo.aruarullm

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Bundle
import android.os.PowerManager
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

/**
 * aruaru-llm Android版シェル(2026-07-24新規作成)。
 *
 * open-web-server Android版(`tokyo.runo.openwebserver.MainActivity`)の
 * 3電源プロファイル・電源抜き差し監視ダイアログの設計パターンを踏襲するが、
 * このアプリはネイティブバイナリを端末上で起動するのではなく、
 * **リモートのaruaru-llmサーバー(`POST /v1/chat`)へHTTP接続する薄い
 * クライアント**である点が異なる:
 * (a) サーバーURL設定 + 最小限のチャットUI(`/v1/chat`を叩くだけ)。
 * (b) 電源プロファイル管理(open-web-server版と同じ3モード+ダイアログ)。
 *
 * スコープ(意図的に今回含めない、詳細は`../CLAUDE.md`のHANDOFF節参照):
 * チャット履歴の永続化、複数テナント切替UI、認証(管理APIトークン入力)。
 */
class MainActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_PROFILE = "profile"
    }

    private var wakeLock: PowerManager.WakeLock? = null
    private var healthPollJob: Job? = null
    private var powerConnectionReceiver: BroadcastReceiver? = null
    private lateinit var currentProfile: PowerProfile

    /**
     * `/healthz`の定期ポーリング間隔(open-web-server Android版と同じ
     * 具体施策: 省電力版は間隔を大きく延ばしDoze/App Standbyへの影響を
     * 最小化、常時電源接続版は短い間隔で即応性を優先する)。
     */
    private fun healthPollIntervalMs(profile: PowerProfile): Long = when (profile) {
        PowerProfile.POWER_SAVE -> 5 * 60_000L // 5分
        PowerProfile.NORMAL -> 60_000L // 1分
        PowerProfile.ALWAYS_ON -> 5_000L // 5秒
    }

    /**
     * LLM推論リクエストへ付与するハードウェアアクセラレーター指示ヘッダー
     * の値。**正直な開示**: aruaru-llm(Rust側)は本セッション時点でこの
     * ヘッダーを一切解釈しない(`opencuda-bert`はCPUパスのみ実装済み、
     * GPU/NPU専用パスは未実装——`CLAUDE.md`「開発方針」節参照)。
     * このAndroid側の指定は将来サーバー側が対応した際に効果を持つ
     * 先取り実装であり、`open-web-server`版の
     * `OPEN_WEB_SERVER_ACCEL_BACKEND`環境変数と同じ設計思想をHTTP
     * ヘッダーへ移した形。
     */
    private fun accelBackendHeaderValue(profile: PowerProfile): String = when (profile) {
        PowerProfile.ALWAYS_ON -> "hardware_accelerator"
        PowerProfile.POWER_SAVE, PowerProfile.NORMAL -> "cpu"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        currentProfile = resolveProfile()
        PowerProfile.save(this, currentProfile)

        val statusText = findViewById<TextView>(R.id.statusText)
        val chatLog = findViewById<TextView>(R.id.chatLog)
        val serverUrlInput = findViewById<EditText>(R.id.serverUrlInput)
        val messageInput = findViewById<EditText>(R.id.messageInput)
        val sendButton = findViewById<Button>(R.id.sendButton)
        val changeProfileButton = findViewById<Button>(R.id.changeProfileButton)

        serverUrlInput.setText(PowerProfile.loadServerUrl(this))
        statusText.text =
            "aruaru-llm [${currentProfile.emoji} ${currentProfile.label}モード] (未接続)"

        applyProfilePowerBehavior(chatLog)
        startPeriodicHealthPoll(statusText, serverUrlInput)

        sendButton.setOnClickListener {
            val message = messageInput.text.toString().trim()
            if (message.isEmpty()) return@setOnClickListener
            val serverUrl = serverUrlInput.text.toString().trim()
            PowerProfile.saveServerUrl(this, serverUrl)

            appendChatLine(chatLog, "あなた: $message")
            messageInput.setText("")
            sendButton.isEnabled = false

            CoroutineScope(Dispatchers.Main).launch {
                val result = withContext(Dispatchers.IO) { sendChatMessage(serverUrl, message) }
                appendChatLine(chatLog, result)
                sendButton.isEnabled = true
            }
        }

        changeProfileButton.setOnClickListener {
            startActivity(Intent(this, ProfileSelectActivity::class.java))
            finish()
        }

        registerPowerConnectionReceiver()
    }

    private fun appendChatLine(chatLog: TextView, line: String) {
        chatLog.append(if (chatLog.text.isEmpty()) line else "\n$line")
    }

    /**
     * `POST /v1/chat`を叩く(`CLAUDE.md`「API」節に記載のリクエスト形状:
     * `{"message": "...", "lang": "ja"}` → `{"reply", "engine",
     * "matched_intent", "reply_lang", "lang_fallback"}`)。
     */
    private fun sendChatMessage(serverUrl: String, message: String): String {
        return try {
            val url = URL("$serverUrl/v1/chat")
            val conn = url.openConnection() as HttpURLConnection
            conn.requestMethod = "POST"
            conn.doOutput = true
            conn.connectTimeout = 5000
            conn.readTimeout = 8000
            conn.setRequestProperty("Content-Type", "application/json")
            conn.setRequestProperty(
                "X-Aruaru-Llm-Accel-Backend",
                accelBackendHeaderValue(currentProfile)
            )

            val body = JSONObject()
            body.put("message", message)
            body.put("lang", "ja")

            OutputStreamWriter(conn.outputStream, Charsets.UTF_8).use { it.write(body.toString()) }

            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val responseText = BufferedReader(InputStreamReader(stream, Charsets.UTF_8)).use { it.readText() }
            conn.disconnect()

            if (code !in 200..299) {
                return "aruaru-llm: エラー($code) $responseText"
            }

            val json = JSONObject(responseText)
            val reply = json.optString("reply", "(応答なし)")
            val engine = json.optString("engine", "unknown")
            "aruaru-llm [$engine]: $reply"
        } catch (e: Exception) {
            "aruaru-llm: 接続エラー: ${e.message}"
        }
    }

    /**
     * 電源の抜き差しを監視する(open-web-server Android版と同じ設計)。
     */
    private fun registerPowerConnectionReceiver() {
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                when (intent.action) {
                    Intent.ACTION_POWER_DISCONNECTED -> onPowerDisconnected()
                    Intent.ACTION_POWER_CONNECTED -> onPowerConnected()
                }
            }
        }
        powerConnectionReceiver = receiver
        val filter = IntentFilter().apply {
            addAction(Intent.ACTION_POWER_DISCONNECTED)
            addAction(Intent.ACTION_POWER_CONNECTED)
        }
        registerReceiver(receiver, filter)
    }

    private fun onPowerDisconnected() {
        if (currentProfile != PowerProfile.ALWAYS_ON) return
        if (isFinishing || isDestroyed) return
        AlertDialog.Builder(this)
            .setTitle("電源が外れました")
            .setMessage(
                "常時電源接続モードで動作中に電源が外れました。\n" +
                    "省電力モードに切り替えますか?それとも通常モードの" +
                    "ままにしますか?\n(推奨: 省電力モード)"
            )
            .setPositiveButton("省電力モードへ切替") { _, _ ->
                switchProfileAndRestart(PowerProfile.POWER_SAVE)
            }
            .setNegativeButton("通常モードのままにする") { _, _ ->
                switchProfileAndRestart(PowerProfile.NORMAL)
            }
            .setCancelable(false)
            .show()
    }

    private fun onPowerConnected() {
        if (currentProfile == PowerProfile.ALWAYS_ON) return
        if (isFinishing || isDestroyed) return
        AlertDialog.Builder(this)
            .setTitle("電源が接続されました")
            .setMessage("常時電源接続モード(ハードウェアアクセラレーター対応)に切り替えますか?")
            .setPositiveButton("常時電源接続へ切替") { _, _ ->
                switchProfileAndRestart(PowerProfile.ALWAYS_ON)
            }
            .setNegativeButton("このままにする", null)
            .show()
    }

    private fun switchProfileAndRestart(newProfile: PowerProfile) {
        PowerProfile.save(this, newProfile)
        Toast.makeText(
            this,
            "${newProfile.emoji} ${newProfile.label}モードへ切り替えます",
            Toast.LENGTH_SHORT
        ).show()
        val intent = Intent(this, MainActivity::class.java)
        intent.putExtra(EXTRA_PROFILE, newProfile.prefValue)
        startActivity(intent)
        finish()
    }

    private fun resolveProfile(): PowerProfile {
        return when (intent?.action) {
            "tokyo.runo.aruarullm.LAUNCH_POWER_SAVE" -> PowerProfile.POWER_SAVE
            "tokyo.runo.aruarullm.LAUNCH_NORMAL" -> PowerProfile.NORMAL
            "tokyo.runo.aruarullm.LAUNCH_ALWAYS_ON" -> PowerProfile.ALWAYS_ON
            else -> {
                val extra = intent?.getStringExtra(EXTRA_PROFILE)
                if (extra != null) PowerProfile.fromPrefValue(extra) else PowerProfile.load(this)
            }
        }
    }

    /**
     * プロファイルごとの電源管理の中身(open-web-server Android版と同じ):
     * 省電力/通常は`WakeLock`を一切取得しない、常時電源接続は
     * `PARTIAL_WAKE_LOCK`を保持する。
     */
    private fun applyProfilePowerBehavior(chatLog: TextView) {
        when (currentProfile) {
            PowerProfile.ALWAYS_ON -> {
                try {
                    val pm = getSystemService(POWER_SERVICE) as PowerManager
                    val lock = pm.newWakeLock(
                        PowerManager.PARTIAL_WAKE_LOCK,
                        "AruaruLlm::AlwaysOnWakeLock"
                    )
                    lock.acquire()
                    wakeLock = lock
                    appendChatLine(chatLog, "[system] PARTIAL_WAKE_LOCKを取得しました(常時電源接続プロファイル)")
                } catch (e: Exception) {
                    appendChatLine(chatLog, "[system] WakeLock取得失敗: ${e.message}")
                }
            }
            PowerProfile.POWER_SAVE -> {
                appendChatLine(chatLog, "[system] WakeLockは取得しません(省電力プロファイル、Doze-friendly)")
            }
            PowerProfile.NORMAL -> {
                appendChatLine(chatLog, "[system] WakeLockは取得しません(通常プロファイル)")
            }
        }
    }

    private fun startPeriodicHealthPoll(statusText: TextView, serverUrlInput: EditText) {
        healthPollJob?.cancel()
        val intervalMs = healthPollIntervalMs(currentProfile)
        healthPollJob = CoroutineScope(Dispatchers.Main).launch {
            while (isActive) {
                val serverUrl = serverUrlInput.text.toString().trim()
                val ok = withContext(Dispatchers.IO) { checkHealthz(serverUrl) }
                statusText.text = if (ok) {
                    "aruaru-llm [${currentProfile.emoji} ${currentProfile.label}] " +
                        "接続OK (poll every ${intervalMs / 1000}s)"
                } else {
                    "aruaru-llm [${currentProfile.emoji} ${currentProfile.label}] " +
                        "/healthz応答なし ($serverUrl)"
                }
                delay(intervalMs)
            }
        }
    }

    private fun checkHealthz(serverUrl: String): Boolean {
        return try {
            val url = URL("$serverUrl/healthz")
            val conn = url.openConnection() as HttpURLConnection
            conn.connectTimeout = 3000
            conn.readTimeout = 3000
            val code = conn.responseCode
            conn.disconnect()
            code == 200
        } catch (_: Exception) {
            false
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        healthPollJob?.cancel()
        powerConnectionReceiver?.let {
            try {
                unregisterReceiver(it)
            } catch (_: IllegalArgumentException) {
                // 未登録のまま呼ばれても(onCreateの早期return等)無視する。
            }
        }
        if (wakeLock?.isHeld == true) {
            wakeLock?.release()
        }
    }
}
