# فلسفه طراحی و سیاست توسعه و قوانین محیط توسعه (aruaru-llm)

> **توجه**: این یک ترجمه فشرده از وضعیت فعلی است. گاهشمار تاریخی
> مفصل HANDOFF (ده‌ها مدخل) به دلیل رعایت اختصار تنها به ژاپنی در
> [CLAUDE.md](CLAUDE.md) در دسترس است — برای جزئیات هر نشست به آنجا
> مراجعه کنید.

مخزن گیت‌هاب: [aon-co-jp/aruaru-llm](https://github.com/aon-co-jp/aruaru-llm).

## نقش این پروژه

یک سرویس HTTP مشترک و مستقل که منطق پاسخ‌دهی «AI chat commerce» را
برای اکوسیستم `aruaru` (aruaru-tokyo، aruaru-db، e-gov.info،
karu.tokyo و غیره) فراهم می‌کند. به‌جای اینکه هر سایت منطق پاسخ چت
خود را پیاده‌سازی کند، همه از طریق HTTP به این سرویس واحد مراجعه
می‌کنند — و بدین‌ترتیب جایی که باید هنگام جایگزینی آینده با استنتاج
واقعی LLM تغییر کند، در یک نقطه متمرکز باقی می‌ماند.

## افشای صادقانه (مهم)

از تاریخ 2026-07-25، `/v1/generate` از crate `opencuda-llm` متعلق به
`open-cuda` (وزن‌های واقعیِ آموزش‌دیده‌ی GPT-2 124M،
`openai-community/gpt2`) برای **تولید متن خودبازگشتی واقعی** استفاده
می‌کند. با این حال، GPT-2 124M مدلی کوچک و متعلق به سال 2019 است و
نه در توانایی و نه در دانش با LLMهای تجاری مدرن مانند GPT-4 قابل
مقایسه نیست. `/v1/chat` (طبقه‌بندی قصد) جدا باقی می‌ماند:
`opencuda-bert` (multilingual-e5-small) embeddingهای جمله را به‌طور
واقعی محاسبه کرده و بر اساس شباهت کسینوسی با بردارهای قصدِ نماینده
طبقه‌بندی می‌کند — یک **طبقه‌بندی شباهت معنایی مبتنی بر رمزگذار**، نه
تولید گفت‌وگو. این دو قابلیت عمداً ادغام نشده‌اند.

## سطح API فعلی

- `POST /v1/chat` — `{"message": "...", "tenant": "..."(اختیاری)}` →
  `{"reply": "...", "engine": "embedding-cosine-v0-opencuda-bert-cpu",
  "matched_intent": "..."}`.
- `POST /v1/generate` — `{"prompt": "...", "max_new_tokens":
  16(اختیاری، پیش‌فرض 16، حداکثر 128), "tenant": "..."(اختیاری)}` →
  `{"completion": "...", "engine":
  "gpt2-124m-greedy-decode-v0-opencuda-llm-cpu", "disclosure": "..."}`.
  در صورت نبود وزن‌های واقعی GPT-2، به‌طور صادقانه `503` برمی‌گرداند
  (بدون بازگشت خاموش مانند `/v1/chat`).
- `POST /admin/tenants` / `GET /admin/tenants` /
  `DELETE /admin/tenants/:host` — مدیریت پویای مستأجران (احراز هویت
  از طریق هدر `x-admin-token`).
- `GET /healthz` — بررسی سلامت.

### جدید: `POST /v1/generate-speculative` (افزوده‌شده در 2026-08-17، کامیت `8f08900`)

رمزگشایی حدسی بدون‌اتلاف به سبک DSpark از طریق
`open-cuda-llm::GptModel::generate_speculative`، **اختیاری** (جایگزین
`/v1/generate` پیش‌فرض نمی‌شود). یک `draft_id` می‌پذیرد که یک مدل
کاتالوگِ از پیش دانلودشده (مانند `"distilgpt2"`) را به‌عنوان مدل
پیش‌نویس معرفی می‌کند. **افشای صادقانه‌ی بحرانی**: در اجرای روی CPU
در `open-cuda`، اندازه‌گیری شده که این مسیر حتی با نرخ پذیرش 80٪
نسبت به `generate()` ساده **کندتر** است — زیرا GEMM ساده‌لوحانه‌ی CPU
تقریباً هیچ سربار dispatch قابل‌حذفی ندارد، بنابراین محاسبه‌ی اضافیِ
مدل پیش‌نویس روی CPU منجر به زیان خالص می‌شود. بررسی سرعت تحت
`real-vulkan` (جایی که سربار dispatch غالب است — مورد استفاده‌ی واقعاً
مورد نظر) هنوز انجام نشده است. همچنین افشا شده: جریمه‌ی تکرار و
مدل‌های فشرده‌شده با MLA توسط این مسیر حدسی پشتیبانی نمی‌شوند.

## پشته‌ی فناوری

Rust + [RPoem](https://github.com/aon-co-jp/RPoem)
(`open-runo-poem-compat`، نمایی سازگار با API پوئم که مستقیماً روی
tokio/hyper پیاده‌سازی شده، از تاریخ 2026-07-31 به‌جای crate واقعی
[Poem](https://github.com/poem-web/poem) — بدون استخراج‌کننده‌ی
`Data<T>`، وضعیت مشترک از طریق closure و `Arc::clone` هنگام ثبت
مسیرها گرفته می‌شود) + [open-cuda](https://github.com/aon-co-jp/open-cuda).
بدون وابستگی به پایگاه‌داده، یک باینری مستقل واحد.

## معماری «شبیه‌سازِ سایه» (分身の術)

مانند `open-web-server`: یک نمونه‌ی در حال اجرا توسط چندین دامنه به
اشتراک گذاشته می‌شود، بدون نیاز به نصب جداگانه برای هر دامنه
(`TenantRegistry` در `src/tenants.rs`، ثبت در زمان اجرا بدون راه‌اندازی
مجدد از طریق APIهای `/admin/tenants`). مدیریت قرار است از
[open-easy-web](https://github.com/aon-co-jp/open-easy-web) انجام شود
(یکپارچه‌سازی هنوز متصل نشده است).

## پروژه‌های مرتبط

- [open-cuda](https://github.com/aon-co-jp/open-cuda) — زمان اجرای GPU، همتای پیکربندی SET
- [e-gov.info](https://github.com/aon-co-jp/e-gov) — اولین فراخوان‌کننده
- [open-easy-web](https://github.com/aon-co-jp/open-easy-web) — مدیریت مورد انتظار
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — منبع معیار قوانین توسعه

---

زبان‌های دیگر: [日本語 (اصلی، با تاریخچه‌ی کامل HANDOFF)](CLAUDE.md) ·
[Deutsch](CLAUDE-German.md) · [Italiano](CLAUDE-Italian.md) ·
[Français](CLAUDE-French.md) · [Русский](CLAUDE-Russian.md) ·
[Українська](CLAUDE-Ukrainian.md) · [עברית](CLAUDE-Hebrew.md) · [العربية](CLAUDE-Arabic.md)
