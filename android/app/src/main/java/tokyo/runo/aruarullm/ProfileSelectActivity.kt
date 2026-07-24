package tokyo.runo.aruarullm

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity

/**
 * 起動時の電源プロファイル選択画面(LAUNCHER)。open-web-server Android版
 * `ProfileSelectActivity`と同じ構成パターン(2026-07-24新規作成)。
 *
 * 「文字表示」と「アイコン」の両方でプロファイルを区別できるように、
 * 各ボタンは絵文字+日本語ラベルを併記する。加えてホーム画面上にも3
 * プロファイルそれぞれの専用アイコン(`activity-alias`、
 * `AndroidManifest.xml`参照)を用意し、アイコンから直接そのプロファイルで
 * 起動できるようにしている。
 */
class ProfileSelectActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_profile_select)

        findViewById<Button>(R.id.buttonPowerSave).setOnClickListener {
            launchWithProfile(PowerProfile.POWER_SAVE)
        }
        findViewById<Button>(R.id.buttonNormal).setOnClickListener {
            launchWithProfile(PowerProfile.NORMAL)
        }
        findViewById<Button>(R.id.buttonAlwaysOn).setOnClickListener {
            launchWithProfile(PowerProfile.ALWAYS_ON)
        }
    }

    private fun launchWithProfile(profile: PowerProfile) {
        PowerProfile.save(this, profile)
        val intent = Intent(this, MainActivity::class.java)
        intent.putExtra(MainActivity.EXTRA_PROFILE, profile.prefValue)
        startActivity(intent)
        finish()
    }
}
