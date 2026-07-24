// aruaru-llm Android版シェル(2026-07-24新規作成)。
//
// open-web-server(`F:\runo\open-web-server\android`)のAndroid構成パターンを
// 参考に新規作成。open-web-serverはクロスコンパイル済みネイティブバイナリを
// 端末上で直接起動する構成だったが、このアプリは**リモートのaruaru-llm
// サーバー(`POST /v1/chat`)へHTTP接続する薄いクライアント**という違いが
// あるため、ネイティブバイナリ同梱(jniLibs)は行わない。3電源プロファイル
// (省電力/通常/常時電源接続)+電源抜き差し監視ダイアログの設計パターンのみ
// 踏襲する(詳細は`../CLAUDE.md`のHANDOFF節参照)。
plugins {
    id("com.android.application") version "8.7.2" apply false
    id("org.jetbrains.kotlin.android") version "2.0.21" apply false
}
