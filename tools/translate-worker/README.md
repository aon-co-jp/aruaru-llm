# translate-worker

aruaru-llm の翻訳(`POST /v1/translate`)用の別プロセス。M2M100 418M(重みは MIT と認識。採用前にモデルカードで要確認)を CTranslate2(MIT)で動かす。
`rust-bert`/`tch`(libtorch)をやめた置き換え(2026-11-16。理由は `src/nllb.rs` の冒頭コメント)。

## モデルの用意(一度だけ。変換にだけ Python を使う)
```bash
pip install ctranslate2 sentencepiece transformers torch huggingface_hub
ct2-transformers-converter --model facebook/m2m100_418M --output_dir m2m100_ct2 --quantization int8
python -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('facebook/m2m100_418M','sentencepiece.bpe.model'),'m2m100_ct2/')"
```

## 起動
```bash
cd tools/translate-worker && cargo build --release
ARUARU_TRANSLATE_MODEL_DIR=./m2m100_ct2 ./target/release/translate-worker      # 127.0.0.1:4620
ARUARU_TRANSLATE_URL=http://127.0.0.1:4620 aruaru-llm ...                       # 本体側
```

受け口は Connect: `POST /aruaru_translate.v1.Translate/Translate`(JSON)。127.0.0.1 以外には待ち受けない。
