# Recent Codebase Updates

For project-management tracking, see `markdown/clickup_worklog_v1_to_v7.md`.

## FinMME Iterative Fixes — v4 through v10 (2026-04 / 2026-05)

### Multi-select MCQ support (`fixes_v5_multiselect`)
- Added `_is_multi_select(sample)` helper in `vision_agent.py` — checks `metadata.question_type_raw == "multiple_choice"` or keywords ("select all", "all that apply", etc.)
- **Planner prompt** (`agents/prompts/planner.txt`): added distinct multi-select MCQ guidance — "Evaluate EACH choice independently", "Compile ALL choices supported", do not stop at first match
- **Vision prompt** (`agents/prompts/vision.txt`): added `{multi_select_block}` placeholder; MCQ rules split into single-select vs multi-select paths; UNANSWERABLE override applies only when no choices are present
- **Vision agent** (`agents/vision_agent.py`): `_format_multi_select_block()` generates a bold warning block; Task `expected_output` updated conditionally for multi-select
- **Vision tool** (`tools/vision_qa_tool.py`): `VisionQAInput` has `multi_select: bool`; `_build_prompt()` has three branches for multi-select / single MCQ / open
- **Verifier agent** (`agents/verifier_agent.py`): passes `multi_select_note` into task description; verifier tool called with `multi_select=True/False`
- **Verifier tool** (`tools/verifier_tool.py`): two separate prompt templates `_VERIFIER_PROMPT_SINGLE` and `_VERIFIER_PROMPT_MULTI`; multi prompt enforces per-choice step-by-step verification and never collapsing to single letter
- Result: multiple_choice accuracy 30.2% → 53.5% (+23.3 pp)

### Verifier confidence gate (`fixes_v6_conf_gate` / `fixes_v7_g3flash_conf_gate`)
- **Selective verifier**: `vision_min_conf` computed from `choice_analysis` confidence scores; if ≥ 0.95 a strong reluctance hint is added to verifier task; if ≥ 0.85 a soft hint
- **Confidence gate** in `runner/run_generate_meps.py`: if verifier returns `verdict=revised` with `confidence < 0.75`, revision is downgraded to confirmed and vision answer retained; logged as `verifier_revision_gated`
- **Bug (v6)**: `"confidence"` added to `VERIFIER_REQUIRED_KEYS` caused `parse_strict` to return `{}` for 244/250 samples (model rarely returns this field) → regression −8.8 pp
- **Repair script** (`scripts/repair_meps_v6.py`): re-parsed `verifier.raw_text` for all 250 v6 MEPs using corrected required_keys `["verdict", "answer", "reasoning"]`; fixed 239 MEPs in-place
- **Fix (v7)**: confidence default changed from `0.0` → `0.75` (neutral, passes gate) in both `verifier_agent.py` and `run_generate_meps.py`; gate now only fires on explicit model output `< 0.75`
- **Repair script** (`scripts/repair_meps_v7.py`): un-gates revisions in v7 MEPs generated before the fix where `confidence` was absent from raw output
- Result: v7 gate fired 1 time (correctly), 66 revisions passed through; judge accuracy 69.6% (+0.2 pp vs v5)

### Pipeline additions (`fixes_v4_g3flash`)
- **Forced-choice retry**: if VisionAgent returns UNANSWERABLE and MCQ choices exist, vision is re-called with "FORCED CHOICE" prepend; reduced UNANSWERABLE rate from 38 → 5
- **Caption → verifier**: `verified_caption` from sample metadata injected into verifier task description
- **MCQ-aware planner**: added eliminate-wrong-options guidance and verify-best-match step
- **Pie/donut legend grounding**: `_LEGEND_GROUNDING_CHART_TYPES` extended to include `"pie"` and `"donut"`
- **Model preset**: `gemini_gemini_flash` (gemini-2.5-flash) and `gemini_gemini_flash_preview` (gemini-2.5-flash-preview-04-17) added to runner config

### FinMME verifier context upgrades — v9 / v10 (2026-05)

- **`fixes_v9_g3flash_related_sents`** (`scripts/run_finmme_fixes_v9.sh`): verifier prompt now includes **`related_sentences`** from sample metadata (analyst source text) and reframes **`verified_caption`** from “background” to **cross-check the draft answer**; planner/vision/verifier remain **Gemini 3 Flash Preview** with **`gemini-2.5-flash-lite`** judge (`--run_tag fixes_v9_g3flash_related_sents`, `--out_label finmme_train_fixes_v9_g3flash_related_sents`).
  - **Full post-eval (n = 1,250):** mean `answer_accuracy` **71.28%**, exact **64.56%**; mcq **72.52%** (1,150), standard **57.00%** (100). vs Gemini-3 zero-shot on the same overlap: **+7.72 pp** mean (**+8.16 pp** exact); McNemar vs zero-shot **χ² ≈ 61.45**, **p ≈ 4.5×10⁻¹⁵**; paired **v9 vs v8** strict correctness **χ² ≈ 0.34, p ≈ 0.56** (not significant). MAX_TOKENS: **199 / 1,250** (**15.9%**); UNANSWERABLE predictions **7 / 1,250**; verifier verdicts **999 / 215 / 36** (confirmed / revised / error — slightly more confirmations / fewer revisions than v8). Latency (metrics JSONL): mean **43.7 s**, p50 **31.0 s**, **p95 87.5 s** — the **~2.4× tighter tail vs v8 (p95 209 s)** is v9's clearest unambiguous gain.
  - Earlier partial-eval read (n = 784) reported v9 at **70.60% mean** and **−1.47 pp vs v8** on that subset; that was selection bias from the fastest-finishing samples and disappears once the full 1,250 rows are scored.
- **`fixes_v10_g3flash_choice_conflict`** (`scripts/run_finmme_fixes_v10.sh`): everything in v9 plus a **choice-conflict meta-signal** — when vision `choice_analysis` marks **≥2** single-select options at confidence **≥0.95**, the verifier sees **conflicting choice labels only** (not the full `choice_analysis` dict), via `format_ambiguity_block` / `choice_analysis_ambiguity_labels` in `src/agentfinvqa/tools/verifier_tool.py` and wiring in `src/agentfinvqa/agents/verifier_agent.py`.
  - **Full post-eval (n = 1,250):** mean `answer_accuracy` **71.08%**, exact **64.24%**; mcq **72.22%** (1,150), standard **58.00%** (100). vs Gemini-3 zero-shot on the same overlap: **+7.52 pp** mean; McNemar vs zero-shot **χ² ≈ 57.4**, **p ≈ 3.6×10⁻¹⁴**; paired **v10 vs v8** strict correctness **p ≈ 0.34** (not significant). MAX_TOKENS: **183 / 1,250** (**14.6%**); UNANSWERABLE predictions **7 / 1,250**; verifier verdicts **982 / 247 / 21** (confirmed / revised / error). Latency (metrics JSONL): mean **44.2 s**, p50 **29.6 s**, p95 **222.9 s**.
- **Eval-only resume:** `scripts/submit_eval.sh` supports **`--eval_resume`** for merging existing `metrics_*.jsonl` / trace / taxonomy by `sample_id` (see `scripts/run_batch.py`).

### Color-area pre-hint (`fixes_v8_g3flash_color_area`)
- **New tool** `src/agentfinvqa/tools/color_area_tool.py`: OpenCV/HSV pixel counter that masks each labeled series color in a chart region and returns a per-label pixel breakdown plus the dominant label.
- **Gating** (`should_trigger_color_area`): the stage fires only when (a) `chart_type` is one of the supported color-area types (bar, pie, donut variants), (b) a legend map exists with ≥2 series, and (c) the question is plausibly a color-area / dominance question. All other charts skip the stage.
- **Vision-prompt injection** (`format_color_area_block_for_vision`): when the OpenCV result is clean, the per-label pixel ranking and dominant-label hint are formatted as a context block and injected into the vision prompt; when `color_ambiguity` or `low_confidence` is set, the hint is suppressed so the vision agent isn't misled by a weak mask.
- **MEP schema** (`mep/schema.py`): new `MEPColorArea` dataclass capturing `triggered`, `breakdown`, `largest`, `total_pixels_matched`, `low_confidence`, `color_ambiguity`, `parse_error`, and `tool_trace`; wired into the `MEP` dataclass as `color_area: Optional[MEPColorArea] = None`.
- **Runner integration** (`runner/run_generate_meps.py`): color-area stage runs after legend grounding and before vision; the result is threaded into `build_vision_task_description` via the `color_area=` kwarg.
- **Vision agent** (`agents/vision_agent.py`): `build_vision_task_description` now accepts `color_area: Optional[MEPColorArea]` and emits the formatted block (or nothing when suppressed).
- **Tests** (`tests/agentfinvqa/test_color_area_tool.py`): unit suite covering the HSV-color helper, OpenCV-imread mocking, dominant-label selection, ambiguity flagging, and the `format_color_area_block_for_vision` formatter.
- **Runner script** (`scripts/run_finmme_fixes_v8_g3flash.sh`): one-shot wrapper that calls `submit_pipeline.sh` with `--config gemini_gemini`, Gemini 3 Flash Preview models, `--resume`, and writes outputs under `output/final/finmme_fixes_v8_g3flash_color_area/`.
- **Results — 250-ID snapshot (early eval, still valid for paired v7 comparison)**:
  - Mean `answer_accuracy` **0.694** (v7: **0.696**, Δ **−0.2 pp** on the same 250 IDs — within noise)
  - Exact (≥ 0.999) **0.628** (same as v7 on that slice)
  - Verifier verdicts: confirmed **187** / revised **59** / error **4** / gated **0** — revision rate drops vs v7 on the same IDs
  - Color-area triggered **12/250** (4.8%); **10** `color_ambiguity` / **2** `low_confidence` suppressions — only ~2 samples received a usable hint, so the 250-ID headline stayed flat
  - MAX_TOKENS **16/250**; UNANSWERABLE **2/250**; legend compliance retries **5/250**
  - Latency p50 **29.8 s** (v7: 32.3 s), mean **46.9 s** (v7: 41.5 s)
- **Results — 1,250-ID scale-up (May 2026 post-eval)** — `metrics_finmme_train_fixes_v8_g3flash_color_area.jsonl`:
  - Mean `answer_accuracy` **0.7124** (exact **65.12%**); by type: **mcq 0.7248** (n = 1,150), **standard 0.5700** (n = 100)
  - `vision_parse_ok` **~0.930**, `has_errors` **~3.9%** (reliability improved vs the first 250-ID tail)
  - Verifier verdicts: confirmed **986** / revised **234** / error **30**
  - Color-area triggered **62/1,250** (**5.0%**); **52** ambiguity / **3** low-confidence suppressions (MEP-derived counts)
  - MAX_TOKENS hits **184/1,250** (**14.7%**); UNANSWERABLE predictions **14/1,250** (**1.1%**); legend compliance retries **165/1,250** (**13.2%**)
  - Latency mean **43.9 s**, p50 **30.1 s**, p95 **209.4 s**
- **Fair same-model baseline (Gemini-3 zero-shot)** — `output/baselines/metrics_finmme_train_zeroshot_structured_gemini_gemini_3_flash_preview.jsonl` is the **full FinMME train** run (**11,099** rows). On the **1,250-ID overlap**, all three g3flash agents beat zero-shot by ~7.5–7.7 pp mean acc.: v8 **+7.68 pp** (McNemar χ² ≈ 68.2, p ≈ 1.1×10⁻¹⁶), v9 **+7.72 pp** (χ² ≈ 61.5, p ≈ 4.5×10⁻¹⁵), v10 **+7.52 pp** (χ² ≈ 57.4, p ≈ 3.6×10⁻¹⁴). Pairwise between agents nothing is significant (v9 vs v8 p = 0.56, v10 vs v8 p = 0.34, v9 vs v10 p = 0.75). The legacy **+10.0 pp on 250 IDs** (strict exact v7 vs repaired zero-shot snapshot) is superseded for the main paper claim — see `results.md` §8b and `markdown/camera_ready_metrics.md`.
- **Artifact hygiene:** `scripts/run_batch.py` now **sorts** `metrics_*.jsonl`, `trace_metrics_*.jsonl`, and `taxonomy_*.jsonl` by `sample_id` before flush so reruns are diff-stable (results buffered in memory until workers finish — interrupted `--eval_only` runs may write nothing).

### Notebook (`notebooks/results_analysis.ipynb`)
- Section 14 expanded to 9-run comparison covering all runs through v8
- Per-type breakdown (single_choice / multiple_choice / numerical), revision gated tracking, confidence field availability
- Section 15 paper comparison table (Table 3 style from FinMME ACL 2025 paper) vs Gemini Flash 2.0, Qwen2.5-VL 72B, GPT-4o, Claude 3.5 Sonnet, GPT-4o Mini
- Added fair 250-ID comparison cell for `fixes_v7_g3flash_conf_gate` vs zero-shot baselines (`gemini-2.5-flash` and `gemini-3-flash-preview`) with coverage-aware matched IDs
- Added Section 16: failure taxonomy by question type (v5→v7), including v7 heatmap and trend plots for top failure categories
- Section 14 / 15 / 16 cells extended to also load and report `fixes_v8_g3flash_color_area`, including per-type accuracy, MAX_TOKENS / UNANSWERABLE / verifier-verdict / latency rows, and a `color_area_triggered` activation breakdown

### Zero-shot parser repair (Gemini 3 Flash Preview)
- Root cause: `baselines/run_zeroshot.py` structured parser only accepted strict JSON, so fenced/truncated-but-recoverable responses were marked `parse_ok=false` with empty `predicted`.
- Implemented robust extraction in `run_zeroshot.py`:
  - code-fence stripping (` ```json ... ``` `)
  - truncated JSON suffix recovery
  - regex fallback for `"answer": "..."` extraction
  - last-resort `Answer: ...` fallback
- Mirrored same extraction improvements in `baselines/fix_zeroshot_scores.py`.
- Repaired existing file in place:
  - `output/baselines/metrics_finmme_train_zeroshot_structured_gemini_gemini_3_flash_preview.jsonl`
  - Reparsed rows: **141**
  - Exact accuracy moved from artificially low (~29.2%) to **52.8%** on 250 rows
  - Backup written as `.jsonl.bak`

### v7 verifier effectiveness snapshot
- v7 verifier verdict counts: `confirmed=180`, `revised=66`, `error=4`
- Exact accuracy by verdict:
  - `confirmed`: **68.33%**
  - `revised`: **50.00%**
  - `error`: **25.00%**
- Revised subset (`n=66`) remains harder than non-revised subset (`n=184`, 67.39% exact), so revision quality is still a key improvement target.

---

## Langfuse Integration
- Added `_Instrumentor` protocol plus `_build_instrumentor` helper so Google/OpenAI telemetry hooks are instantiated only when their packages are installed; prevents assigning `None` to imported classes and keeps imports at module top for Ruff E402.
- `get_client` now runs from a cached `_client` and `_initialised` flag exactly as before, but the instrumentation guards wrap `instrument()` calls in `contextlib.suppress` so failures never crash the service.
- Switched `register_dataset` to accept `Sequence[PerceivedSample]`, iterate with an explicit `for` loop, and print success counts; the CLI now uses a typed `DatasetLoaderConfig` (loader/display/default image dir) so both dataset CLI and runner share one configuration source.
- Introduced `DatasetConfig` TypedDict and `DatasetLoader` aliases inside the MEP runner, then reused them for CLI argument validation and to compute dataset slugs.
- Extracted `_run_verifier_stage` from `process_sample` to encapsulate parsing, defaulting, and error tracking for Pass 2.5 output; reduces branching and keeps error messages consistent with the previous inlined logic.
- Centralized dataset registration and Langfuse prompt pushes at runner start, so a single `get_client()` call now drives dataset creation, prompt versioning, and on-the-fly score logging without duplicated env handling between CLI utilities.
- Added clearer log lines (dataset name, skip counts, Langfuse enablement) around runner startup so long-running pipeline executions expose their configuration at a glance.

## Vision Agent Prompting
- Split `build_vision_task_description` into `_format_choice_blocks`, `_format_context_block`, `_format_plan_steps`, and `_format_ocr_block` so each conditional block can be unit-tested independently and Ruff no longer reports PLR0912.
- Each formatter mirrors the existing prompt structure (e.g., choice labels lines, context prefix, OCR axis formatting), so the generated prompt text is identical while the main builder simply fills the template.
- The refactor keeps `VISION_PROMPT_PATH` templating untouched, guaranteeing that only formatting helpers change behavior; this makes prompt diffs easy to audit alongside prompt text stored in `agents/prompts/vision.txt`.

## Evaluation & Taxonomy Pipeline
- `_parse_choice_labels` now stores lookups in `maybe_label` before appending, eliminating assignments of `str | None` into a list of `str`; this appeases mypy and clarifies the branch logic.
- Gemini helper functions in `error_taxonomy.py` and `eval_topk.py` now construct `genai.types.Part.from_bytes(...)` directly and only cast the `client.models` attribute to `Any`, removing redundant casts that mypy flagged while keeping compatibility with the google-genai stubs.
- The Streamlit dashboard’s `sidebar_path_picker` docstring now starts with an imperative verb (“Select…”) matching Ruff D401 requirements.
- The taxonomy classifier now shares the same prompt string but routes missing-image cases through explicit OpenAI/Gemini fallbacks, improving error messaging and making it simple to reason about when a multimodal call will actually occur.
- Top-K evaluation continues to record Langfuse scores when available, but safer casting ensures those writes degrade gracefully if Langfuse credentials are absent.

## Static Analysis & Tooling Hygiene
- Added module docstring to `langfuse_integration/client.py`, docstrings to `DatasetLoaderConfig`/`DatasetConfig`, and removed blank lines per Ruff D1xx/D2xx suggestions.
- Updated typing imports (e.g., using `collections.abc.Callable`, `TypedDict`, `Protocol`) to make mypy happy under `from __future__ import annotations`.
- Verified all changes with `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check ...` and `... uv run mypy`, ensuring the pre-commit hooks that originally failed now succeed locally.
- Leveraged Ruff’s PLR checks as a forcing function to split logic-heavy helpers; the resulting structure also simplifies future unit test authoring because each helper has a single responsibility.
- Documented the new behavior in `updates.md` so future reviewers can see at a glance why typing and linting changes were necessary (e.g., compatibility with google-genai release cadence).
