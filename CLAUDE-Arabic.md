# فلسفة التصميم وسياسة التطوير وقواعد بيئة التطوير (aruaru-llm)

> **ملاحظة**: هذه ترجمة مختصرة للوضع الحالي. يظل سجل HANDOFF
> التاريخي المفصّل (عشرات الإدخالات) متاحاً باليابانية فقط في
> [CLAUDE.md](CLAUDE.md) توخياً للإيجاز — راجعه هناك لتفاصيل كل جلسة.

مستودع GitHub: [aon-co-jp/aruaru-llm](https://github.com/aon-co-jp/aruaru-llm).

## دور هذا المشروع

خدمة HTTP مشتركة ومستقلة توفر منطق الاستجابة الخاص بـ«تجارة الدردشة
بالذكاء الاصطناعي» لمنظومة `aruaru` (aruaru-tokyo وaruaru-db
وe-gov.info وkaru.tokyo وغيرها). بدلاً من أن يطبّق كل موقع منطق
استجابة الدردشة الخاص به، تستعلم جميع المواقع عن هذه الخدمة الواحدة
عبر HTTP — مما يركّز المكان الذي يجب تغييره لاحقاً عند التحول إلى
استدلال LLM حقيقي في مكان واحد.

## إفصاح صادق (مهم)

اعتباراً من 2026-07-25، يستخدم `/v1/generate` حزمة `opencuda-llm`
التابعة لـ`open-cuda` (أوزان GPT-2 124M المدرَّبة فعلياً،
`openai-community/gpt2`) لتحقيق **توليد نص انحداري ذاتي حقيقي**. مع
ذلك، فإن GPT-2 124M نموذج صغير من عام 2019 ولا يُقارَن بنماذج اللغة
التجارية الحديثة مثل GPT-4 لا في القدرة ولا في المعرفة. يبقى
`/v1/chat` (تصنيف النوايا) منفصلاً: يحسب `opencuda-bert`
(multilingual-e5-small) تضمينات جمل حقيقية ويصنّف عبر تشابه جيب
التمام مع متجهات نوايا تمثيلية — وهو **تصنيف تشابه دلالي قائم على
مُرمِّز**، وليس توليد حوار. القدرتان غير مدمَجتين عمداً.

## واجهة برمجة التطبيقات الحالية

- `POST /v1/chat` — `{"message": "...", "tenant": "..."(اختياري)}` ←
  `{"reply": "...", "engine": "embedding-cosine-v0-opencuda-bert-cpu",
  "matched_intent": "..."}`.
- `POST /v1/generate` — `{"prompt": "...", "max_new_tokens":
  16(اختياري، الافتراضي 16، الحد الأقصى 128), "tenant": "..."(اختياري)}`
  ← `{"completion": "...", "engine":
  "gpt2-124m-greedy-decode-v0-opencuda-llm-cpu", "disclosure": "..."}`.
  إذا كانت أوزان GPT-2 الحقيقية غير موجودة، تُعاد `503` بصدق (دون
  رجوع صامت كما في `/v1/chat`).
- `POST /admin/tenants` / `GET /admin/tenants` /
  `DELETE /admin/tenants/:host` — إدارة ديناميكية للمستأجرين
  (مصادقة عبر ترويسة `x-admin-token`).
- `GET /healthz` — فحص الصحة.

### جديد: `POST /v1/generate-speculative` (أُضيف في 2026-08-17، الالتزام `8f08900`)

فك تشفير تخميني بلا فقدان بأسلوب DSpark عبر
`open-cuda-llm::GptModel::generate_speculative`، **اختياري** (لا
يستبدل `/v1/generate` الافتراضي). يقبل `draft_id` يحدد نموذجاً من
الكتالوج تم تنزيله مسبقاً (مثل `"distilgpt2"`) كنموذج مسودة. **إفصاح
صادق حاسم**: عند التنفيذ على المعالج (CPU) في `open-cuda`، تبيّن
بالقياس أن هذا المسار **أبطأ** من `generate()` البسيط حتى مع معدل
قبول 80% — لأن عملية GEMM الساذجة على المعالج لا تكاد تحمل أي عبء
إرسال (dispatch) يمكن إزالته، فيصبح الحساب الإضافي لنموذج المسودة
على المعالج خسارة صافية. لم يُجرَ بعد التحقق من السرعة تحت
`real-vulkan` (حيث يهيمن عبء الإرسال — وهو حالة الاستخدام المقصودة
فعلياً). كما تم الإفصاح عن أن عقوبة التكرار والنماذج المضغوطة بـ MLA
غير مدعومة في هذا المسار التخميني.

## حزمة التقنيات

Rust + [RPoem](https://github.com/aon-co-jp/RPoem)
(`open-runo-poem-compat`، واجهة متوافقة مع Poem API مُنفَّذة مباشرة
فوق tokio/hyper، منذ 2026-07-31 بدلاً من حزمة
[Poem](https://github.com/poem-web/poem) الحقيقية — لا يوجد مستخرج
`Data<T>`، تُلتقط الحالة المشتركة عبر closure و`Arc::clone` عند
تسجيل المسارات) + [open-cuda](https://github.com/aon-co-jp/open-cuda).
لا اعتماد على قاعدة بيانات، ملف تنفيذي واحد مستقل.

## معمارية «الاستنساخ الظلي» (分身の術)

كما في `open-web-server`: نسخة تشغيل واحدة تتشاركها عدة نطاقات دون
الحاجة إلى تثبيت لكل نطاق (`TenantRegistry` في `src/tenants.rs`،
تسجيل وقت التشغيل دون إعادة تشغيل عبر واجهات `/admin/tenants`).
من المفترض أن تتم الإدارة من
[open-easy-web](https://github.com/aon-co-jp/open-easy-web) (التكامل
غير مُفعَّل بعد).

## مشاريع ذات صلة

- [open-cuda](https://github.com/aon-co-jp/open-cuda) — بيئة تشغيل GPU، الشريك في تهيئة SET
- [e-gov.info](https://github.com/aon-co-jp/e-gov) — أول جهة استدعاء
- [open-easy-web](https://github.com/aon-co-jp/open-easy-web) — الإدارة المتوقعة
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — المصدر المرجعي لقواعد التطوير

---

لغات أخرى: [日本語 (الأصل، مع سجل HANDOFF الكامل)](CLAUDE.md) ·
[Deutsch](CLAUDE-German.md) · [Italiano](CLAUDE-Italian.md) ·
[Français](CLAUDE-French.md) · [Русский](CLAUDE-Russian.md) ·
[Українська](CLAUDE-Ukrainian.md) · [עברית](CLAUDE-Hebrew.md) · [فارسی](CLAUDE-Persian.md)
