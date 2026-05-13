# Camera-Ready Metrics (FinMME)

This file captures the paper-safe numbers to cite from the current workspace state.

Primary artifacts:

- **Agent v8 (scale-up):** `output/final/finmme_fixes_v8_g3flash_color_area/metrics_finmme_train_fixes_v8_g3flash_color_area.jsonl` (**n = 1,250** train `sample_id`s)
- **Agent v9 (verifier related sentences + caption cross-check):** `output/final/finmme_fixes_v9_g3flash_related_sents/metrics_finmme_train_fixes_v9_g3flash_related_sents.jsonl` (**n = 1,250** — same IDs as v8)
- **Agent v10 (verifier choice-conflict meta-signal):** `output/final/finmme_fixes_v10_g3flash_choice_conflict/metrics_finmme_train_fixes_v10_g3flash_choice_conflict.jsonl` (**n = 1,250** — same IDs as v8)
- **Agent v7 (ablation ladder):** `output/final/finmme_fixes_v7_g3flash_conf_gate/*` (**n = 250**)
- **Zero-shot Gemini-3 Flash Preview (structured):** `output/baselines/metrics_finmme_train_zeroshot_structured_gemini_gemini_3_flash_preview.jsonl` (**n = 11,099** FinMME train rows — full train split coverage after the April 2026 re-run)
- Earlier ladder runs (v4, v5): `output/final/finmme_fixes_v4_g3flash/*`, `output/final/finmme_v5_multiselect/*`

Unless stated otherwise, **accuracy** means mean **`answer_accuracy`** (rule-based scorer with numeric tolerance + MCQ partial credit). **Exact** means the fraction of rows with **`answer_accuracy ≥ 0.999`**.

## 1) Headline — fair same-model gain (matched IDs, **n = 1,250**)

All **1,250** agent IDs (v8, v9, v10) appear in the full Gemini-3 zero-shot train file — join on `sample_id`.

| System | Mean `answer_accuracy` | Exact (≥ 0.999) |
|--------|----------------------:|----------------:|
| Zero-shot `gemini-3-flash-preview` (structured) | **63.56%** | **56.40%** |
| Agent `fixes_v8_g3flash_color_area` | **71.24%** | **65.12%** |
| Agent `fixes_v9_g3flash_related_sents` | **71.28%** | **64.56%** |
| Agent `fixes_v10_g3flash_choice_conflict` | **71.08%** | **64.24%** |
| **Δ (v8 − zero-shot)** | **+7.68 pp** | **+8.72 pp** |
| **Δ (v9 − zero-shot)** | **+7.72 pp** | **+8.16 pp** |
| **Δ (v10 − zero-shot)** | **+7.52 pp** | **+7.84 pp** |

Paired **McNemar** test (binary “strict correct” = `answer_accuracy == 1` vs `< 1`, scipy `chi2.cdf`):

- **v8 vs zero-shot**: **χ² = 68.21**, **p ≈ 1.1 × 10⁻¹⁶**
- **v9 vs zero-shot**: **χ² = 61.45**, **p ≈ 4.5 × 10⁻¹⁵**
- **v10 vs zero-shot**: **χ² = 57.37**, **p ≈ 3.6 × 10⁻¹⁴**
- **Pairwise between agents (all not significant)**: v9 vs v8 χ² = 0.34, p = 0.56; v10 vs v8 χ² = 0.90, p = 0.34; v9 vs v10 χ² = 0.10, p = 0.75.

**By `question_type` on the same overlap (v8 / v9 / v10 vs zero-shot):**
- mcq (n = 1,150): **+8.09 / +8.13 / +7.83 pp** mean
- standard (n = 100): **+3.00 / +3.00 / +4.00 pp** — treat the standard delta as **exploratory** (wide CI).

## 2) Ablations on the **250-ID** ladder (historical / controlled slice)

| Comparison | Mean `answer_accuracy` | Exact |
|---|---:|---:|
| Zero-shot G3 (structured) on a **250-ID** snapshot after parser repair | *(see `*.bak` / notebook §14)* | **52.8%** |
| Agent `fixes_v7_g3flash_conf_gate` (same 250 IDs) | **69.6%** | **62.8%** |
| Δ (v7 agent − zero-shot) on that snapshot | — | **+10.0 pp** exact |

Use **§1** for the main paper claim; keep **§2** only if you need to cite the original ablation-era strict-exact snapshot.

## 3) Verifier effectiveness

**v7 (n = 250)** — accuracy by `verifier_verdict` (exact):

| Verdict | Count | Exact |
|---|---:|---:|
| `confirmed` | 180 | **68.33%** |
| `revised` | 66 | **50.00%** |
| `error` | 4 | **25.00%** |

**v8 (n = 1,250)** — exact on subset:

| Verdict | Count | Exact |
|---|---:|---:|
| `confirmed` | 986 | **68.15%** |
| `revised` | 234 | **55.56%** |
| `error` | 30 | *(rare slice — cite with care)* |

**v9 (n = 1,250)** — exact on subset:

| Verdict | Count | Exact |
|---|---:|---:|
| `confirmed` | 999 | **67.97%** |
| `revised` | 215 | **52.09%** |
| `error` | 36 | *(rare slice — cite with care)* |

**v10 (n = 1,250)** — exact on subset:

| Verdict | Count | Exact |
|---|---:|---:|
| `confirmed` | 982 | **69.04%** |
| `revised` | 247 | **47.77%** |
| `error` | 21 | *(rare slice — cite with care)* |

## 4) Most defensible novelty lift (multi-select)

From v4 to v5 on FinMME `multiple_choice` (`n=86`):

- `fixes_v4_g3flash`: **30.23%**
- `fixes_v5_multiselect`: **53.49%**
- **Gain: `+23.26 pp`**

## 5) Failure taxonomy snapshot (v7, n = 250)

Top failure categories from `taxonomy_finmme_train_fixes_v7_g3flash_conf_gate.jsonl`:

| Failure Type | Count | Share of Failures (n=93) |
|---|---:|---:|
| `question_misunderstanding` | 25 | **26.9%** |
| `legend_confusion` | 18 | **19.4%** |
| `extraction_error` | 17 | **18.3%** |

## 6) Runtime context

**v7** (from trace metrics on 250 IDs): mean **~32 s**, median **~18.7 s**, p95 **~147 s**.

**v8 @ 1,250** (from metrics JSONL): mean **43.9 s**, median **30.1 s**, p95 **209.4 s** — agent wall-clock remains **several×** a single zero-shot call (see results memo for a head-to-head latency table).

**v9 @ 1,250** (from metrics JSONL): mean **43.7 s**, median **31.0 s**, **p95 87.5 s** — **~2.4× tighter tail than v8/v10**; adding `related_sentences` + caption-as-cross-check gives the verifier enough textual grounding to converge without long retry chains. This is v9's clearest unambiguous gain at full scale even though headline accuracy is tied with v8.

**v10 @ 1,250** (from metrics JSONL): mean **44.2 s**, median **29.6 s**, p95 **222.9 s** — same order of magnitude as v8.

## 7) Important caveats (must keep in paper text)

1. **Two different FinMME train sample counts** — the iterative ladder (v1–v7) is documented on **250** IDs; **`fixes_v8`**, **`fixes_v9`**, and **`fixes_v10`** are all evaluated on the same **1,250** train IDs. Compare systems only after **`sample_id` join**.
2. **Zero-shot train file is full-split (11,099 rows)** — the headline fair comparison uses the **1,250-ID overlap** with v8/v9/v10, not the zero-shot mean over all 11k (that full mean is **~61.5%**, dominated by much harder **standard** questions than in the 1,250-ID slice).
3. **Gemini-3 zero-shot file was parser-repaired** (141 rows recovered on an early snapshot) before the headline **52.8%** exact-on-250 number stabilized; the authoritative baseline file is the 11k-row JSONL above.
4. **Mean vs exact** — do not mix them in the same delta sentence without labeling columns.

## 8) Suggested one-line claim

On **1,250** FinMME train questions where both systems have predictions, our Gemini-3-based agent (**`fixes_v8_g3flash_color_area`**) achieves **71.24%** mean `answer_accuracy` vs **63.56%** for a **fair same-model** Gemini-3 Flash Preview zero-shot baseline (**+7.68 pp**; McNemar **p ≪ 0.001**), with the clearest lift on **MCQ** (**+8.09 pp** on 1,150). Adding analyst `related_sentences` + caption-as-cross-check (**v9**, **71.28%**, **+7.72 pp**) or a choice-conflict meta-signal (**v10**, **71.08%**, **+7.52 pp**) leaves headline accuracy statistically tied with v8 (pairwise McNemar p ≥ 0.34), but **v9 cuts p95 latency by ~2.4×** (87.5 s vs v8's 209.4 s) — a real operational gain. The strongest **isolated** pipeline ablation remains multi-select MCQ support (**+23.3 pp** on `multiple_choice` from v4 → v5).
