# aruaru-llm

*日本語*: [README.md](README.md) ·
*English*: [README-English.md](README-English.md) ·
*Other languages*: [Deutsch](README-German.md) · [Italiano](README-Italian.md) ·
[Français](README-French.md) · [Русский](README-Russian.md) ·
[Українська](README-Ukrainian.md) · [فارسی](README-Persian.md) · [العربية](README-Arabic.md)

> 📌 **עדכון אחרון (2026-08-10)**: הפונקציה החדשה של `open-cuda`,
> `GptModel::generate_with_repetition_penalty` (עונש חזרתיות בסגנון
> CTRL), חוברה אל `/v1/generate` והיא **מופעלת כברירת מחדל** (משתנה
> הסביבה `ARUARU_LLM_REPETITION_PENALTY`, ערך ברירת מחדל `1.3`; קבעו
> `1.0` כדי לשחזר את ההתנהגות הישנה ללא עונש). שינוי זה פותר באופן ישיר
> מצב תקלה ידוע במודל הבסיס GPT-2 — חזרה אינסופית על אותו מחרוזת —
> מכיוון שלמודל הבסיס אין כיוונון עדין לדיאלוג. אומת על משקלי GPT-2
> 124M אמיתיים בצד `open-cuda`: ללא העונש הלולאה אכן משוחזרת; עם
> `penalty=1.3` היא נעצרת ומייצרת טקסט שיחתי טבעי מבחינה דקדוקית. עם
> `penalty=1.0` הפלט זהה בייט-בייט ל-API הקיים `generate()`, כך שאין
> נסיגה בבדיקות אחרות. לפרטים ראו [CLAUDE.md](CLAUDE.md) (ביפנית
> בלבד).

> 📌 **עדכון (2026-08-08)**: נוסף נתיב אופציונלי (כבוי כברירת מחדל) של
> דחיסת מטמון KV בסגנון MLA (מימוש של `open-cuda` בהשראת DeepSeek-V3)
> אל `/v1/generate` דרך `ARUARU_LLM_ENABLE_MLA_KV_COMPRESSION=1`. עבור
> GPT-2 124M מדובר ב-head_dim=64 -> d_c=16 (75% פחות אחסון KV לטוקן).
> **גילוי כן**: מטריצות ההיטלים למטה/למעלה מאותחלות אקראית (לא
> מאומנות), ולכן מדובר בדחיסה עם אובדן — בדיקה אמיתית מקצה לקצה על
> משקלי GPT-2 124M האמיתיים הראתה פלט שנפגם בצורה ניכרת/חזרתי יותר
> בהשוואה לנתיב הבלתי-דחוס, ולכן זה כבוי כברירת מחדל. קיימת גם גרסה
> מכוילת באמצעות PCA (`ARUARU_LLM_MLA_CALIBRATED=1`), הנמנעת מלולאות
> החזרתיות הנחותות של הגרסה האקראית, אך עדיין נחותה באופן ברור מהנתיב
> הבלתי-דחוס — גם היא נותרת כבויה כברירת מחדל.

> 📌 משימה ממתינה (2026-08-06): קיימת תוכנית לשלב את טכניקות Toshiba
> SBM ו-DeepSeek. לפרטים ראו [CLAUDE.md](CLAUDE.md).

> 📌 **עדכון (2026-08-07)**: אומת באמצעות בינארי פועל אמיתי + בקשות
> HTTP חיות ש-`/v1/chat` ו-`/v1/classify-security` **אינם** סובלים
> מהבאג "קלט ריק → 503" שתוקן קודם עבור `/v1/generate` ו-`/v1/translate`
> — שניהם מחזירים כראוי 200 עבור קלט ריק. לא נדרש שינוי קוד.

שירות תגובה משותף מסוג "מסחר צ'אט מבוסס בינה מלאכותית" עבור מערכת
האקולוגית `aruaru` (aruaru-tokyo, aruaru-db, e-gov.info, karu.tokyo
ועוד). במקום שכל אתר יממש בעצמו לוגיקת תגובת צ'אט, כולם פונים לשירות
HTTP יחיד זה — ומרכזים את הנקודה היחידה שתצטרך להשתנות כאשר בסופו של
דבר יחובר היסק LLM אמיתי.

> ⚠️ **גילוי כן (חשוב, עודכן ב-2026-07-25)**: החל מ-2026-07-25 שירות
> זה משלב את חבילת `opencuda-llm` של `open-cuda` (משקלי GPT-2 124M
> אמיתיים ומאומנים, `openai-community/gpt2`), כך ש-`POST /v1/generate`
> מבצע כעת **יצירת טקסט אוטורגרסיבית אמיתית** — הטענה "אין יצירה
> אוטורגרסיבית" למטה כבר אינה חלה על נקודת קצה זו. עם זאת, **GPT-2 124M
> הוא מודל קטן משנת 2019 ואינו ברמת מודלי LLM מסחריים מודרניים כמו
> GPT-4** מבחינת יכולת או ידע. זוהי הדגמה לכך שיצירה עצמאית פועלת ללא
> חוזה API חיצוני של LLM, ולא טענה לאיכות ברמת החזית הטכנולוגית — הפלט
> לרוב שוטף מבחינה דקדוקית באנגלית אך אינו מובטח כמדויק מבחינה
> עובדתית (עלול "להזות"). `POST /v1/chat` (סיווג כוונות באמצעות
> הטמעות משפטים של `opencuda-bert` + דמיון קוסינוס, החל מ-2026-07-21)
> נשאר נתיב נפרד, קל וזריז לתגובות מוכנות מראש — ובכוונה אינו מאוחד עם
> היצירה. לפרטים ולנימוקים ראו [CLAUDE.md](CLAUDE.md).

## מותאם ("SET") עם open-cuda

תלוי, דרך תלות נתיב, בחבילות `opencuda-core`/`opencuda-cpu`/
`opencuda-blas`/`opencuda-bert` של
[`open-cuda`](https://github.com/aon-co-jp/open-cuda). בכל בקשת
`/v1/chat`, `opencuda-bert` מריץ מעבר קדימה של multilingual-e5-small
(תוך קריאה בפועל לגרעיני GEMM/Attention האמיתיים של `opencuda-blas`
על `opencuda_cpu::CpuDevice`) כדי להטמיע את ההודעה, ואז משווה אותה
באמצעות דמיון קוסינוס להטמעה הייצוגית השמורה במטמון של כל כוונה. זוהי
קריאת זמן-ריצה אמיתית דרך צינור החישוב של open-cuda, לא רק הפניה
ב-`Cargo.toml` — אומת על ידי הפעלה בפועל של השרת והרצת
`POST /v1/chat`.

עם זאת, זהו אינו היסק LLM נוירוני אמיתי (יצירת דיאלוג) — רק מעבר
קדימה של המקודד; המפענח האוטורגרסיבי נותר בלתי-ממומש. נתיבים מהירים
ייעודיים ל-GPU (`GemmPath::CuBlas`/`RocBlas`/`OneMkl`) עדיין הם
תשתית-שלד (stub) (נתיבי CPU ו-Vulkan גנרי ממומשים). לפרטים ראו את קטע
HANDOFF ב-`CLAUDE.md` של open-cuda.

**עדכון 2026-07-25 (חזרה לגיבוי לצורך זמינות)**: אם
`models/multilingual-e5-small/` (470MB+) חסר או נכשל בטעינה, שירות זה
כעת חוזר אוטומטית למכפלת הנקודה המקורית מסוג bag-of-words
(`src/bow_fallback.rs`) במקום להכשיל בקשות לחלוטין. שדה ה-`engine`
בתגובת `/v1/chat` תמיד מדווח בכנות איזה נתיב שימש בפועל
(`embedding-cosine-v0-opencuda-bert-cpu` או
`bow-dotproduct-v0-opencuda-cpu-fallback`) — איכות הסיווג נמוכה
בצורה ניכרת בנתיב הגיבוי (התאמת מילות מפתח, לא הבנה סמנטית).

## API

- `POST /v1/chat` — `{"message": "...", "tenant": "..."(אופציונלי)}`
  → `{"reply": "...", "engine": "...", "matched_intent": "..."}`
  (סיווג כוונות, תגובות מוכנות מראש קלות/מהירות)
- `POST /v1/generate` — `{"prompt": "...", "max_new_tokens":
  16(אופציונלי, ברירת מחדל 16, מוגבל ל-128), "tenant":
  "..."(אופציונלי)}` → `{"completion": "...", "engine":
  "gpt2-124m-greedy-decode-v0-opencuda-llm-cpu", "disclosure": "..."}`
  (יצירה אוטורגרסיבית אמיתית באמצעות משקלי GPT-2 124M — כבד יותר אך
  אמיתי. **עונש החזרתיות ברירת המחדל הוא `1.3`** — משתנה
  `ARUARU_LLM_REPETITION_PENALTY` לעקיפה, `1.0` מבטל אותו — כדי למנוע
  לולאות חזרה אינסופיות. מומלצות בקשות (prompts) באנגלית, מכיוון
  שאוצר המילים BPE של GPT-2 מאומן בעיקר על טקסט אנגלי. דוגמה, שאומתה
  מקצה לקצה דרך HTTP אמיתי: `{"prompt": "The quick brown fox",
  "max_new_tokens": 16}` → `"completion": "es are a great way to get
  a little bit of a kick out of your"`)
- `GET /v1/models/catalog` — מודלים תואמי GPT-2 הזמינים להתקנה
  (`gpt2`/`distilgpt2`/`gpt2-medium`/`gpt2-large`/`gpt2-xl`, האחרון
  נוסף ב-2026-07-27), אילו כבר מותקנים, ותיקיית המודל הפעיל הנוכחי.
- `POST /v1/models/install` / `POST /v1/models/select` — הורדת מודל
  מהקטלוג מ-Hugging Face, והחלפה חמה של המודל הפעיל ללא הפעלה מחדש
  של התהליך.
- `GET /v1/recommend` (נוסף ב-2026-07-27) — מזהה חומרה (VRAM) דרך
  `open-cuda` (Vulkan) או `open-directx` (DXGI) ומחזיר גודל מודל
  מומלץ ממשפחת GPT-2, מבלי להוריד דבר.
- `POST /v1/recommend-and-download` (נוסף ב-2026-07-27, מפעיל את
  הכפתור "Download recommended LLM") — מזהה חומרה → בוחר גודל מומלץ
  → מוריד אותו מ-Hugging Face אם עדיין אינו קיים (אידמפוטנטי) →
  מחליף חם את `/v1/generate` לשימוש בו. מחזיר
  `{"recommendation": {...}, "already_installed":bool,
  "switched_to_recommended":bool, "message_ja":"..."}`.
- `GET /` (נוסף ב-2026-07-27) — ממשק HTML סטטי מינימלי
  (`static/index.html`, ללא מסגרת עבודה) עם כפתור אחד "Download
  recommended LLM", תצוגת התקדמות, ולוח בדיקת יצירה לאחר ההחלפה.
- `POST /admin/tenants` / `GET /admin/tenants` /
  `DELETE /admin/tenants/:host` — ניהול רישום דיירים (אימות באמצעות
  כותרת `x-admin-token`)
- `GET /healthz` — בדיקת תקינות

### זיהוי חומרה → גודל LLM מומלץ (נוסף ב-2026-07-27)

`src/hardware.rs` מממש היוריסטיקה פשוטה הבוחרת גודל ממשפחת GPT-2
(124M/355M/774M/1.5B) על סמך ה-VRAM שזוהה: <2GB → 124M, 2-4GB → 355M,
4-8GB → 774M, 8GB+ → 1.5B; GPU שלא ניתן לזהות / CPU בלבד → 124M
(ברירת מחדל בטוחה). **גילוי כן**: מדובר בהשוואה גסה של גודל מול
VRAM (מספר פרמטרים × 4 בייטים, אומדן fp32), לא במודל ביצועים מדויק —
הוא מתעלם מזיכרון מטמון KV והפעלות (activations).

זיהוי GPU הוא אופציונלי דרך תכונות Cargo `hw-detect-vulkan` /
`hw-detect-directx` (כבויות כברירת מחדל, כדי שגרסאות בנייה שמיועדות
ל-CPU בלבד או קומפילציה צולבת לא ייאלצו להיות תלויות בטוען Vulkan /
ב-Windows SDK). כאשר מופעל, Vulkan מועדף; אם שתי התכונות מופעלות,
תוצאת DXGI (DirectX) נבדקת צולבות מול תוצאת Vulkan ונרשמת ביומן
(`cross_check_agreement`). **אומת על חומרה אמיתית**: בהרצה עם
`--features hw-detect-vulkan` על ה-NVIDIA GeForce GT 730 של מכונה
זו, דווח `vram_bytes=2104819712` — התואם בדיוק לערך שנרשם קודם דרך
DXGI ב-`CLAUDE.md` של `open-cuda`, ומאשר ששני נתיבי הזיהוי מסכימים
לגבי כרטיס גרפי זה.

### פריקת מטמון KV/משקלים בסגנון "Engram" של DeepSeek: נבדק ונדחה (2026-08-08)

בדקנו האם הטכניקה "Engram" של DeepSeek — פינוי ידע סטטי (מטמון KV או
מקטעי משקלים) מ-VRAM לזיכרון RAM המערכת וטעינה מחדש לפי דרישה — יכולה
לעזור לשירות זה לפעול על GPU בעלי VRAM קטן כמו GT 730. **לאחר קריאת
הקוד בפועל, החלטנו שלא לממש זאת** — לא מכיוון שזה קשה, אלא מכיוון
שנתיב ההיסק של `open-cuda` שממנו תלוי מאגר זה אין לו מלכתחילה מצב
שוכן ב-VRAM לפנות. כל שילוח GEMM/Attention/softmax ב-`opencuda-blas`
(כלומר כל קריאת `sgemm` המופנית ל-Vulkan) מקצה מאגר VRAM דרך שומר
RAII בשם `ScopedAlloc` (`opencuda-blas/src/lib.rs`), מעתיק
מארח→התקן, מחשב, מעתיק התקן→מארח, ומשחרר מיידית — שום דבר אינו נשאר
ב-VRAM לאחר סיום הקריאה. הן משקלי GPT-2 (`word_embeddings` של
`GptModel` ו-`Linear` של כל שכבה) והן מטמון KV
(`k`/`v`/`k_latent`/`v_latent` של `open-cuda-llm::KvCacheHead`) הם
`Vec<f32>` פשוטים החיים בזיכרון RAM המערכת לאורך כל חייהם, גם בהרצה
עם `--features real-vulkan`. במילים אחרות, ארכיטקטורה זו כבר מגיעה —
במקרה של תכנון, לא בכוונה — למצב שאליו Engram שואף: המצב נשאר שוכן
בזיכרון RAM המערכת כל הזמן, וב-GPU נוגעים רק באופן חולף, לפי פעולה.
הוספת שכבת פינוי LRU מעל כך לא הייתה משאירה דבר לפנות, ולכן לא היה
אפקט הניתן למדידה לדווח עליו (איננו הולכים לטעון תועלת שאיננו יכולים
למדוד). לפרטי נתיבי הקוד המדויקים שנקראו ראו את רשומת HANDOFF מיום
2026-08-08 ב-CLAUDE.md.

### סיווג לעומת יצירה — מה להשתמש

`/v1/chat` (סיווג) ו-`/v1/generate` (יצירה) משרתים מטרות שונות
ובכוונה לא אוחדו: `/v1/chat` רק מנתב לתגובות מוכנות מראש והוא
קל/מהיר (מעבר קדימה יחיד של הטמעה); `/v1/generate` מריץ את מודל
GPT-2 124M המלא (548MB של משקלים) והוא כבד יותר אך מייצר טקסט חופשי
אמיתי. בחרו את המתאים לשימוש שלכם.

## ארכיטקטורת "שיבוט הצל" (分身の術)

בהתאם לאותו עיצוב כמו `open-web-server`: מופע רץ יחיד משותף על ידי
מספר תחומים, ללא צורך בהתקנה עבור כל תחום. הניהול אמור להתבצע מתוך
[open-easy-web](https://github.com/aon-co-jp/open-easy-web) (אינטגרציה
זו טרם חוברה). לפרטים ראו [CLAUDE.md](CLAUDE.md).

## מחסנית טכנולוגית

Rust + [RPoem](https://github.com/aon-co-jp/RPoem)
(`open-runo-poem-compat`, קדמת פנים תואמת API Poem הממומשת ישירות מעל
tokio/hyper — ללא תלות בחבילת [Poem](https://github.com/poem-web/poem)
האמיתית, הוגר ב-2026-07-31) + [open-cuda](https://github.com/aon-co-jp/open-cuda).
אין תלות במסד נתונים, קובץ בינארי עצמאי יחיד. ניתן לשימוש מ-Rust או
מכל שפה אחרת דרך HTTP פשוט (שירות זה הוא דלת הכניסה של שירות ה-HTTP
עבור פורטים ב-Rust של ספריות בינה מלאכותית ב-Python —
`opencuda-bert`/`opencuda-llm`/`opencuda-whisper`, כלומר המקבילות של
Transformers/vLLM/Whisper).

ראו [CLAUDE.md](CLAUDE.md) (ביפנית בלבד) לפילוסופיית העיצוב ו-
[PORTING.md](PORTING.md) (ביפנית בלבד) כיצד להעביר תבניות אלו למקום
אחר.

## התקנה

החל מ-2026-07-23, נוספו `install.sh` (לינוקס, רושם שירות systemd),
`install.ps1` (Windows, מדפיס את שלבי רישום שירות ה-Windows), ו-
`.github/workflows/release.yml` (בונה בינאריים Linux x86_64 /
Windows x86_64 בכל דחיפת תג ומצרף אותם ל-
[GitHub Releases](https://github.com/aon-co-jp/aruaru-llm/releases)).
**גילוי כן**: בעת ההפעלה, בינארי זה זקוק למשקלי המודל
`multilingual-e5-small` (470MB+, Hugging Face, רישיון MIT) שיש
להביא בנפרד — לא כלולים במתקין מסיבות רישוי; ראו `install.sh`/
`install.ps1` לפקודת ההורדה. לבנייה יש תלות נתיב "אחות" ב-
`../open-cuda`, ולכן בנייה מקוד המקור מחייבת שיבוט (clone) של
`open-cuda` לתיקייה סמוכה (ה-CI עושה זאת אוטומטית דרך
`release.yml`). **נוסף ב-2026-07-25**: `/v1/generate` (יצירת GPT-2
124M) דורש בנוסף `config.json` / `model.safetensors` (548MB) /
`tokenizer.json` (`openai-community/gpt2`, מ-Hugging Face) תחת
`../open-cuda/crates/opencuda-llm/models/gpt2/` (ניתן לעקוף את הנתיב
עם משתנה הסביבה `ARUARU_LLM_GPT2_DIR`). אם חסר, רק `/v1/generate`
מחזיר 503 — `/v1/chat` ושאר השירות ממשיכים לעבוד כרגיל (עיצוב
המעדיף זמינות, אותה פילוסופיה כמו `bow_fallback`).

```
curl -fsSL https://github.com/aon-co-jp/aruaru-llm/releases/latest/download/aruaru-llm-linux-x86_64.tar.gz | tar xz
sudo ./install.sh
```

## פרויקטים קשורים

- [open-cuda](https://github.com/aon-co-jp/open-cuda) — סביבת ריצה של GPU (השותף ב-SET)
- [e-gov.info](https://github.com/aon-co-jp/e-gov) — הקורא הראשון המיועד
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — מקור מדיניות הפיתוח הקנוני
