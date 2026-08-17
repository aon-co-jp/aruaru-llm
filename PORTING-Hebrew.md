# PORTING.md — מדריך להעברת aruaru-llm (גרסה מקוצרת)

> **הערה**: זהו תרגום מקוצר. המדריך הטכני המלא עם פרטי קוד ומלכודות
> נשאר זמין רק ביפנית ב-[PORTING.md](PORTING.md) — יש לעיין בו לפני
> אימוץ בפועל של דפוס כלשהו.

סיכום דפוסי היישום הניתנים לשימוש חוזר מפרויקט זה, למקרה שיועברו
לפרויקט אחר:

1. **דפוס צימוד עם open-cuda (תצורת SET)**: תלות נתיב ב-
   `opencuda-core`/`opencuda-cpu`; מפעיל ביצוע ליבת GPU אמיתי
   (`alloc_buffer`→`copy_from_host`→`launch_kernel`→`synchronize`→
   `copy_to_host`).
2. **סיווג כוונות מבוסס כללים, המיועד להחלפה עתידית ב-LLM אמיתי**:
   לשמור על שדה `engine` ולדווח בו תמיד בכנות איזו מימוש נעשה בו
   שימוש בפועל.
3. **שכבת API של HTTP דרך RPoem** (`open-runo-poem-compat`) במקום
   ה-crate האמיתי `poem` — אין extractor `Data<T>`, מצב משותף נלכד
   באמצעות closure ו-`Arc::clone`.
4. **דפוס אימות קלט ריק** (2026-08-06): `400 Bad Request` מפורש
   במקום לאפשר לשגיאות פנימיות של הטוקנייזר לדלוף כ-`503` מטעה.
5. **דפוס רישום דיירים "שיבוט צל"** (משותף עם `open-web-server`):
   `TenantRegistry` + נקודות קצה `/admin/tenants`.
6. **יכולת יצירה אמיתית דרך `opencuda-llm::GptModel`**: משקלי
   GPT-2 124M — לעולם אין להשמיט את שדה `disclosure`, ואין לאחד את
   `/v1/chat` ו-`/v1/generate`.
7. **זיהוי חומרה ← גודל LLM מומלץ ← הורדה אוטומטית** (פיצ'רים
   אופציונליים `hw-detect-vulkan`/`hw-detect-directx`, דפוס בדיקה
   צולבת, גילוי נאות שמדובר רק בהיוריסטיקה גסה של גודל-מול-VRAM).
8. **דפוס פלאגין תרגום** (פיצ'ר `nllb-translate`): תלות כבדה
   אופציונלית של `rust-bert`/`tch`, מבודדת מאחורי פיצ'ר Cargo,
   כבויה כברירת מחדל.
9. **פיצ'ר dispatch של `real-vulkan`** — **הערה**: עדיין לא מומלץ
   להעברה למקום אחר, עקב באג ידוע שלא נפתר (`Linear::forward` אינו
   מחבר את בתי ה-SPIR-V של `matmul.spv` ל-`sgemm`, מה שגורם ל-
   `GemmPath::VulkanGeneric` להיכשל מיידית).
10. **דפוס קנס חזרתיות** (`generate_with_repetition_penalty`, ברירת
    מחדל `1.3`, ניתן לדריסה באמצעות משתנה סביבה).

**הסתייגות חשובה**: GPT-2 124M קטן ומקורו ב-2019 — אינו ניתן להשוואה
ל-LLM מסחריים מודרניים. `/v1/chat` נותר מבוסס כללים + סיווג דמיון
מבוסס מקודד, לא יצירת דיאלוג נוירונית. יש לגלות זאת גם בכל יעד
העברה.

---

שפות נוספות: [日本語 (מקור, פרטים מלאים)](PORTING.md) ·
[Deutsch](PORTING-German.md) · [Italiano](PORTING-Italian.md) ·
[Français](PORTING-French.md) · [Русский](PORTING-Russian.md) ·
[Українська](PORTING-Ukrainian.md) · [فارسی](PORTING-Persian.md) · [العربية](PORTING-Arabic.md)
