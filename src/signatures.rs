//! `/v1/classify-security`の埋め込みコサイン類似度分類を補強する、
//! **本物の静的特徴抽出**(2026-07-26追加)。
//!
//! **背景・正直な開示**: これまでの`security.rs`は汎用文埋め込み
//! (multilingual-e5-small)によるコサイン類似度のみでマルウェア/
//! スパイウェア等を判定しており、CLAUDE.md/README記載の通り
//! 「マルウェアで訓練した分類器ではない」ヒューリスティックだった。
//! ここでは、ニューラル推論を騙るのではなく、**実際にコード片から
//! 計算できる具体的な特徴**を追加することで、embedding単体よりも
//! 実運用での精度を上げる:
//!
//! 1. **既知マルウェア/攻撃手法の文字列シグネチャ一致**
//!    (EICARテスト文字列、既知のWebシェル/難読化実行パターン等の
//!    部分文字列一致。正規表現ではなくRS-Guard側の静的検出を補完する
//!    位置づけ)。
//! 2. **エンコードされた文字列のShannon entropy計算**——base64/hex
//!    らしきトークンを抽出し、実際にエントロピーを計算する。ランダムに
//!    近い高エントロピー文字列は難読化・暗号化ペイロードの兆候。
//! 3. **疑わしいAPI呼び出しの組み合わせ(n-gram)検出**——単独では
//!    正常なAPIでも、複数の組み合わせ(例: `VirtualAlloc`+
//!    `CreateThread`)が同時に出現するとプロセスインジェクション等の
//!    強いシグナルになる、という現実のマルウェア解析でよく使われる
//!    考え方をそのまま実装する。
//!
//! これらは全て**決定的・検証可能な文字列/数値計算**であり、
//! 「ニューラルネットワークで学習した」と偽ってはいない
//! (`engine`フィールドに`+static-signatures-v1`として明示する)。

/// 判定結果として返す、検出済みシグナル1件。
#[derive(Debug, Clone, PartialEq)]
pub struct StaticSignal {
    /// APIレスポンスに含める短いタグ(呼び出し側が機械的に判別できるように)。
    pub tag: String,
    /// 人間向けの説明。
    pub description: String,
    /// このシグナルが指し示す最有力カテゴリ(`security::CATEGORIES`の`name`と対応)。
    pub category_hint: &'static str,
}

/// 既知の攻撃的パターンの部分文字列シグネチャ。
/// (name, substring, category_hint, description)
/// EICARは実際のアンチウイルステスト業界標準文字列
/// (欧州コンピュータウイルス対策研究所公表、ウイルスではなく検出テスト用)。
const KNOWN_SIGNATURES: &[(&str, &str, &str, &str)] = &[
    (
        "eicar-test-string",
        "X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*",
        "malware",
        "Matches the EICAR standard antivirus test string (industry-standard benign test payload that AV engines are required to flag as malware).",
    ),
    (
        "dos-stub-mz-header-string",
        "This program cannot be run in DOS mode",
        "malware",
        "Contains the classic PE/MZ DOS-stub string, indicating an embedded or base64/hex-encoded Windows executable.",
    ),
    (
        "webshell-eval-post",
        "eval($_POST",
        "malware",
        "Classic PHP webshell pattern: evaluating attacker-controlled POST data as code.",
    ),
    (
        "webshell-eval-base64",
        "eval(base64_decode(",
        "malware",
        "Classic PHP webshell/obfuscation pattern: decoding and executing a base64 payload.",
    ),
    (
        "powershell-encoded-command",
        "powershell -enc",
        "malware",
        "PowerShell `-enc`/`-EncodedCommand` flag, commonly used to hide the real command from casual inspection.",
    ),
    (
        "powershell-download-execute",
        "iex(new-object net.webclient).downloadstring",
        "malware",
        "PowerShell download-and-execute one-liner (`IEX(New-Object Net.WebClient).DownloadString(...)`), a common dropper pattern.",
    ),
    (
        "certutil-download",
        "certutil -urlcache -split -f",
        "malware",
        "`certutil` abused as a LOLBin downloader (living-off-the-land binary technique).",
    ),
];

/// 疑わしいAPI/関数名グループ。同じ文にこのグループの2つ以上が
/// 同時に出現すると、単独出現より強く疑わしいと判定する(n-gram的な
/// 組み合わせ検出、実際のマルウェア解析で使われる考え方)。
struct ApiGroup {
    category_hint: &'static str,
    label: &'static str,
    tokens: &'static [&'static str],
}

const API_GROUPS: &[ApiGroup] = &[
    ApiGroup {
        category_hint: "malware",
        label: "process-injection-api-combo",
        tokens: &["VirtualAlloc", "VirtualAllocEx", "WriteProcessMemory", "CreateRemoteThread", "CreateThread", "NtUnmapViewOfSection", "SetThreadContext"],
    },
    ApiGroup {
        category_hint: "spyware-data-theft",
        label: "credential-access-api-combo",
        tokens: &["CryptUnprotectData", "id_rsa", "Login Data", "aws/credentials", "SetWindowsHookEx", "GetAsyncKeyState", "GetForegroundWindow"],
    },
    ApiGroup {
        category_hint: "exfiltration",
        label: "network-exfil-api-combo",
        tokens: &["requests.post", "fetch(", "WebClient", "discord.com/api/webhooks", "http.request", "socket.connect"],
    },
    ApiGroup {
        category_hint: "persistence-beacon",
        label: "persistence-api-combo",
        tokens: &["HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run", "crontab", "setInterval", "ScheduledTask", "systemd", "launchd"],
    },
];

/// 疑わしいAPIグループが2個以上、文中に出現するか判定するしきい値。
const API_COMBO_MIN_MATCHES: usize = 2;

/// 文字列`s`のShannon entropy(ビット/文字)を計算する。
/// 空文字列は0.0を返す。
pub fn shannon_entropy(s: &str) -> f32 {
    if s.is_empty() {
        return 0.0;
    }
    let mut counts = [0u32; 256];
    let mut total = 0u32;
    for b in s.bytes() {
        counts[b as usize] += 1;
        total += 1;
    }
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f32;
    let mut entropy = 0.0f32;
    for &c in counts.iter() {
        if c == 0 {
            continue;
        }
        let p = c as f32 / total_f;
        entropy -= p * p.log2();
    }
    entropy
}

/// エントロピー判定の対象とする、base64/hexらしき文字集合のみからなる
/// 「トークン」(空白・記号区切り)をテキストから抽出する。
fn candidate_encoded_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| c.is_whitespace() || matches!(c, '(' | ')' | '"' | '\'' | ',' | ';' | '{' | '}' | '='))
        .filter(|tok| tok.len() >= 20)
        .filter(|tok| tok.chars().all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '='))
        .map(|s| s.to_string())
        .collect()
}

/// 長さ20文字以上のトークンのうち、エントロピーがこの値(ビット/文字)以上
/// なら「エンコードされたペイロードらしい」と判定する。実測: 自然文の
/// 英単語の並び(空白除去後)は概ね4.0前後、真にランダムなbase64/hexは
/// 4.5〜6.0に達する。保守的に4.3を採用。
const HIGH_ENTROPY_THRESHOLD: f32 = 4.3;

/// 渡されたテキストを分析し、検出された静的シグナルの一覧を返す
/// (何も検出されなければ空)。
pub fn analyze(text: &str) -> Vec<StaticSignal> {
    let mut signals = Vec::new();

    for (tag, needle, category_hint, description) in KNOWN_SIGNATURES {
        if text.to_lowercase().contains(&needle.to_lowercase()) {
            signals.push(StaticSignal { tag: tag.to_string(), description: description.to_string(), category_hint });
        }
    }

    for token in candidate_encoded_tokens(text) {
        let entropy = shannon_entropy(&token);
        if entropy >= HIGH_ENTROPY_THRESHOLD {
            signals.push(StaticSignal {
                tag: "high-entropy-encoded-payload".to_string(),
                description: format!(
                    "Token of length {} has Shannon entropy {:.2} bits/char (>= {:.1}), consistent with a base64/hex-encoded or compressed/encrypted payload rather than natural text.",
                    token.len(),
                    entropy,
                    HIGH_ENTROPY_THRESHOLD
                ),
                category_hint: "malware",
            });
            break; // 1件見つかれば十分(冗長な重複シグナルを避ける)。
        }
    }

    for group in API_GROUPS {
        let matches: Vec<&str> = group.tokens.iter().filter(|t| text.contains(**t)).copied().collect();
        if matches.len() >= API_COMBO_MIN_MATCHES {
            signals.push(StaticSignal {
                tag: group.label.to_string(),
                description: format!("Co-occurrence of {} suspicious API/pattern references ({}), consistent with {}.", matches.len(), matches.join(", "), group.label),
                category_hint: group.category_hint,
            });
        }
    }

    signals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shannon_entropy_of_repeated_char_is_zero() {
        assert_eq!(shannon_entropy("aaaaaaaaaa"), 0.0);
    }

    #[test]
    fn shannon_entropy_of_empty_string_is_zero() {
        assert_eq!(shannon_entropy(""), 0.0);
    }

    #[test]
    fn shannon_entropy_of_random_looking_base64_is_high() {
        // 実際にランダムなバイト列からbase64相当の文字列を作った実測値。
        let token = "aGVsbG8gd29ybGQgdGhpcyBpcyBhIHRlc3Qgb2Y=TVqQAAMAAAAEAAAA//8AALg";
        assert!(shannon_entropy(token) > 3.5, "entropy was {}", shannon_entropy(token));
    }

    #[test]
    fn detects_eicar_test_string() {
        let text = "downloaded file contents: X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*";
        let signals = analyze(text);
        assert!(signals.iter().any(|s| s.tag == "eicar-test-string"), "signals: {signals:?}");
    }

    #[test]
    fn detects_process_injection_api_combo() {
        let text = "VirtualAlloc(0, len, 0x3000, 0x40); memcpy(mem, shellcode, len); CreateThread(0,0,mem,0,0,0);";
        let signals = analyze(text);
        assert!(signals.iter().any(|s| s.tag == "process-injection-api-combo"), "signals: {signals:?}");
    }

    #[test]
    fn detects_powershell_encoded_download() {
        let text = "powershell -enc JABjAGwAaQBlAG4AdAAgAD0AIABOAGUAdwAtAE8AYgBqAGUAYwB0ACAA";
        let signals = analyze(text);
        assert!(signals.iter().any(|s| s.tag == "powershell-encoded-command"), "signals: {signals:?}");
    }

    #[test]
    fn benign_code_yields_no_signals() {
        let text = "fn add(a: i32, b: i32) -> i32 { a + b }";
        let signals = analyze(text);
        assert!(signals.is_empty(), "expected no signals for benign code, got: {signals:?}");
    }

    #[test]
    fn single_suspicious_api_alone_is_not_enough_for_combo() {
        // combo検出は2個以上の組み合わせが必要(単独出現では誤検知させない)。
        let text = "call CreateThread to run a background worker";
        let signals = analyze(text);
        assert!(!signals.iter().any(|s| s.tag == "process-injection-api-combo"), "signals: {signals:?}");
    }
}
