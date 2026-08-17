# פילוסופיית עיצוב ומדיניות פיתוח וכללי סביבת פיתוח (aruaru-llm)

> **הערה**: זהו תרגום מקוצר של המצב הנוכחי. יומן ה-HANDOFF ההיסטורי
> המפורט (עשרות רשומות) נשאר זמין רק ביפנית ב-[CLAUDE.md](CLAUDE.md),
> מטעמי תמציתיות — יש לעיין שם לפרטים על כל סשן.

מאגר GitHub: [aon-co-jp/aruaru-llm](https://github.com/aon-co-jp/aruaru-llm).

## תפקיד הפרויקט

שירות HTTP משותף ועצמאי המספק את לוגיקת התגובה של "AI chat commerce"
עבור מערכת האקולוגיה `aruaru` (aruaru-tokyo, aruaru-db, e-gov.info,
karu.tokyo וכו'). במקום שכל אתר יממש לוגיקת תגובת צ'אט משלו, כולם
פונים לשירות המרכזי הזה דרך HTTP — כך המקום שיש לשנות בעת מעבר עתידי
להסקת LLM אמיתית נשאר מרוכז במקום אחד.

## גילוי נאות (חשוב)

החל מ-2026-07-25, `/v1/generate` משתמש ב-crate `opencuda-llm` של
`open-cuda` (משקלים מאומנים אמיתיים של GPT-2 124M,
`openai-community/gpt2`) עבור **יצירת טקסט אוטורגרסיבית אמיתית**. עם
זאת, GPT-2 124M הוא מודל קטן משנת 2019 ואינו ניתן להשוואה ל-LLM
מסחריים מודרניים כמו GPT-4, לא ביכולת ולא בידע. `/v1/chat` (סיווג
כוונות) נשאר נפרד: `opencuda-bert` (multilingual-e5-small) מחשב
הטמעות משפטים אמיתיות ומסווג לפי דמיון קוסינוס מול וקטורי כוונה
מייצגים — זהו **סיווג דמיון סמנטי מבוסס מקודד**, לא יצירת דיאלוג. שתי
היכולות במכוון אינן מאוחדות.

## משטח ה-API הנוכחי

- `POST /v1/chat` — `{"message": "...", "tenant": "..."(אופציונלי)}` →
  `{"reply": "...", "engine": "embedding-cosine-v0-opencuda-bert-cpu",
  "matched_intent": "..."}`.
- `POST /v1/generate` — `{"prompt": "...", "max_new_tokens":
  16(אופציונלי, ברירת מחדל 16, מקסימום 128), "tenant":
  "..."(אופציונלי)}` → `{"completion": "...", "engine":
  "gpt2-124m-greedy-decode-v0-opencuda-llm-cpu", "disclosure": "..."}`.
  אם המשקלים האמיתיים של GPT-2 חסרים, מוחזר בכנות `503` (ללא נפילה
  שקטה כמו ב-`/v1/chat`).
- `POST /admin/tenants` / `GET /admin/tenants` /
  `DELETE /admin/tenants/:host` — ניהול דינמי של דיירים (אימות
  באמצעות כותרת `x-admin-token`).
- `GET /healthz` — בדיקת תקינות.

### חדש: `POST /v1/generate-speculative` (נוסף ב-2026-08-17, קומיט `8f08900`)

פענוח ספקולטיבי חסר-אובדן בסגנון DSpark דרך
`open-cuda-llm::GptModel::generate_speculative`, **אופציונלי**
(אינו מחליף את `/v1/generate` הרגיל). מקבל `draft_id` המציין מודל
קטלוג שכבר הורד (למשל `"distilgpt2"`) כמודל טיוטה. **גילוי נאות
קריטי**: בהרצה על CPU ב-`open-cuda` נמדד שהמסלול הזה **איטי יותר**
מ-`generate()` הרגיל, אפילו בשיעור קבלה של 80% — משום שחישוב GEMM
נאיבי על CPU כמעט ואינו נושא תקורת dispatch שניתן לבטל, כך שהחישוב
הנוסף של מודל הטיוטה על CPU מהווה הפסד נטו. בדיקת מהירות תחת
`real-vulkan` (שם תקורת ה-dispatch דומיננטית — מקרה השימוש המיועד
בפועל) עדיין לא בוצעה. כמו כן נחשף: קנס חזרתיות ומודלים דחוסי MLA
אינם נתמכים במסלול הספקולטיבי הזה.

## מחסנית טכנולוגית

Rust + [RPoem](https://github.com/aon-co-jp/RPoem)
(`open-runo-poem-compat`, פסאדה תואמת API של Poem המיושמת ישירות מעל
tokio/hyper, החל מ-2026-07-31 במקום ה-crate האמיתי
[Poem](https://github.com/poem-web/poem) — אין extractor `Data<T>`,
מצב משותף נלכד באמצעות closure `Arc::clone` בעת רישום הנתיבים) +
[open-cuda](https://github.com/aon-co-jp/open-cuda). ללא תלות במסד
נתונים, קובץ בינארי עצמאי יחיד.

## ארכיטקטורת "שיבוט צל" (分身の術)

כמו `open-web-server`: מופע רץ יחיד משותף בין מספר דומיינים, ללא צורך
בהתקנה לכל דומיין (`TenantRegistry` ב-`src/tenants.rs`, רישום בזמן
ריצה ללא הפעלה מחדש דרך ה-API-ים של `/admin/tenants`). הניהול צפוי
להתבצע מ-[open-easy-web](https://github.com/aon-co-jp/open-easy-web)
(האינטגרציה עדיין לא חוברה).

## פרויקטים קשורים

- [open-cuda](https://github.com/aon-co-jp/open-cuda) — סביבת ריצה של GPU, בן הזוג ב-SET
- [e-gov.info](https://github.com/aon-co-jp/e-gov) — הקורא הראשון
- [open-easy-web](https://github.com/aon-co-jp/open-easy-web) — הניהול הצפוי
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — המקור הקנוני לכללי הפיתוח

---

שפות נוספות: [日本語 (מקור, עם היסטוריית HANDOFF מלאה)](CLAUDE.md) ·
[Deutsch](CLAUDE-German.md) · [Italiano](CLAUDE-Italian.md) ·
[Français](CLAUDE-French.md) · [Русский](CLAUDE-Russian.md) ·
[Українська](CLAUDE-Ukrainian.md) · [فارسی](CLAUDE-Persian.md) · [العربية](CLAUDE-Arabic.md)
