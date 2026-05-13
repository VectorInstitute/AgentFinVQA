# AgentFinVQA — Initial results summary

This note condenses numbers from `notebooks/results_analysis.ipynb` (saved cell outputs), `output/final/**/summary*.csv`, and `output/baselines/*.jsonl` as of the current workspace. Use it as a starting point for an internal or initial report; re-run the notebook to refresh figures and tables.

---

## Scope

| Source | Rows (approx.) | Notes |
|--------|----------------|--------|
| ChartQAPro (test) | 8,107 metric rows across configs in the notebook load | Per-config *n* varies slightly (e.g. OpenAI 2,709 vs Gemini 2,708). |
| FinMME (train) | 11,082 | Full Gemini agent run on the train split. |

Primary models referenced: **Gemini 2.5 Flash** (agent and zero-shot) and **OpenAI GPT-4o** (ChartQAPro agent).

---

## 1. ChartQAPro — overall accuracy (full available test metrics)

| System | Accuracy | n |
|--------|----------|---|
| Gemini (2.5-Flash) — full agent | **0.5343** | 2,708 |
| Gemini — no verifier | **0.5264** | 2,690 |
| OpenAI (GPT-4o) | **0.4186** | 2,709 |

**By question type (ChartQAPro, Gemini vs OpenAI):** hypothetical is strongest for both; OpenAI is relatively weaker on **standard** questions (0.39 vs ~0.57 for Gemini).

---

## 2. ChartQAPro — zero-shot vs full agent (baseline file)

From `output/baselines/metrics_chartqapro_test_zeroshot_gemini_gemini_2_5_flash.jsonl` (as loaded in the notebook):

- **~2,689** rows  
- Mean answer accuracy **~0.597**

The full Gemini agent on the aggregated test metrics is **~0.534** (section 1), so on this aggregate, zero-shot is higher than the full pipeline on the same broad ChartQAPro evaluation — the notebook explores *where* the agent gains or loses (sample-level and by type).

---

## 3. ChartQAPro — 250-sample aligned slice (No-OCR ablation)

Restricted to **250** samples where **no-OCR, no-verifier, full Gemini, planner v2, OpenAI**, and **zero-shot** all have results (see notebook §13).

| System | Accuracy (n = 250) |
|--------|---------------------|
| Zero-shot | **0.6520** |
| Agent (full Gemini) | **0.6360** |
| Agent (planner v2) | **0.6320** |
| Agent (no OCR) | **0.6280** |
| Agent (no verifier) | **0.6120** |
| OpenAI (full) | **0.5040** |

On this slice, zero-shot is slightly above the full Gemini agent; planner v2 is close to full agent. Removing OCR or verifier hurts accuracy relative to full.

---

## 4. FinMME — full train split (Gemini agent)

| System | Accuracy | n |
|--------|----------|---|
| Gemini (2.5-Flash) full agent | **0.5653** | 11,082 |

**By question type:** MCQ **~0.63**, open **standard** **~0.23** (FinMME is much harder on non-MCQ items in this run).

---

## 5. FinMME — zero-shot vs agent vs planner v2 (matched IDs)

Baseline: `output/baselines/metrics_finmme_train_zeroshot_structured_gemini_gemini_2_5_flash.jsonl` (**233** rows).  
Planner v2 metrics: `output/final/finmme_planner_v2/` (**250** rows in the planner file).

Evaluation uses the **intersection of IDs** present in both zero-shot and planner v2 → **n = 233** (not 250).

| System | Accuracy (n = 233) |
|--------|---------------------|
| Zero-shot (structured) | **~0.547** |
| Agent (Gemini full) | **~0.592** |
| Agent (Planner v2) | **~0.526** |

On this matched set, the **full Gemini agent beats zero-shot**; **planner v2 is below both** (numbers recomputed from JSONL; notebook shows bootstrap CIs on the same slice).

**Planner v2 — official summary on all 250 FinMME train samples** (`output/final/finmme_planner_v2/summary_finmme_train_planner_v2.csv`):

- **ALL:** accuracy **0.544** (250 samples), mean latency **~18.9 s**

---

## 6. ChartQAPro — planner v2 summary (250 samples)

From `output/final/chartqapro_planner_v2/summary_chartqapro_test_planner_v2.csv`:

- **ALL:** accuracy **0.632** (250 samples), mean latency **~16.2 s**

---

## 7. Latency vs accuracy (notebook)

Illustrative means from the latency table in the notebook (ChartQAPro aggregates):

| System | Mean latency (s) | Accuracy |
|--------|-------------------|----------|
| Zero-shot (Gemini) | ~3.20 | ~0.597 |
| Agent (Gemini) | ~11.96 | ~0.534 |
| Agent (Gemini — no verifier) | ~12.69 | ~0.526 |
| Agent (OpenAI) | ~7.73 | ~0.419 |

(Exact latency rows depend on which trace files were merged; see notebook output.)

---

## 8. Figures and artifacts (paths)

Generated or referenced under `output/final/` (examples):

- `fig_zeroshot_vs_agent.pdf`
- `fig_latency_vs_accuracy_zs_with_no_verifier.pdf`
- `fig_finmme_zeroshot_vs_agent_250.pdf`

---

## 8b. FinMME — iterative fixes experiments (train slice)

Progressive runs on a shared FinMME **train** slice: **v1–v7** on **250** labeled train IDs; **`fixes_v8_g3flash_color_area`**, **`fixes_v9_g3flash_related_sents`**, and **`fixes_v10_g3flash_choice_conflict`** target the same **1,250** train IDs (superset of the original 250 — same pipeline and config).  
Runs v1–v6 use **Gemini 2.5 Flash Lite** for all stages; v4, v7, v8, v9, and v10 use **Gemini 3 Flash Preview** for planner/vision/verifier.

**Post-eval coverage:** v8, v9, and v10 all have full **`metrics_*.jsonl`** rows (**n = 1,250**) on the same FinMME train ID slice — direct paired comparison is supported across all three.

### What changed in each run

| Run | Changes vs previous |
|-----|---------------------|
| `no_legend_grounding` | Baseline — no legend grounding, no caption injection, default token limits |
| `fixes_v1` | + token limits raised to 2 048 · + analyst-caption hint injected into vision prompt · + verifier UNANSWERABLE pushback · + legend grounding enabled |
| `fixes_v2` | + Gemini `thinking_budget=0` (eliminates thinking tokens) · + `choice_analysis` trimmed · + MCQ choices passed explicitly to verifier |
| `fixes_v3` | + `thinking_budget=512` (moderate reasoning without exhausting output budget) |
| `fixes_v4_g3flash` | + Gemini 3 Flash Preview model · + forced-choice retry when vision returns UNANSWERABLE on MCQ · + caption injected into verifier task · + MCQ-aware planner (eliminate-wrong-options guidance) · + pie/donut legend grounding |
| `fixes_v5_multiselect` | + full multi-select MCQ support (planner / vision / verifier each have separate prompts for select-all-that-apply questions) |
| `fixes_v6_conf_gate` | + verifier self-confidence field · + reluctance hint when vision is high-confidence · + confidence gate (revisions with confidence < 0.75 downgraded) — *(MEPs repaired in-place after parse-key bug caused 244/250 parse failures on first run)* |
| `fixes_v7_g3flash_conf_gate` | + confidence default fixed to 0.75 (neutral) so gate only fires on explicit low confidence · fresh re-run with Gemini 3 Flash Preview + gate correctly active |
| `fixes_v8_g3flash_color_area` | + OpenCV pixel-counting `color_area_tool` inserted between legend grounding and vision · per-label pixel breakdown + dominant-label hint injected into vision prompt for bar/pie/donut charts · `MEPColorArea` field added to MEP schema · gated by chart type, legend availability, and ambiguity flags (`color_ambiguity`, `low_confidence`) |
| `fixes_v9_g3flash_related_sents` | + verifier sees `related_sentences` from sample metadata · caption reframed from “background” to “cross-check the draft answer” (same g3flash stack as v8) |
| `fixes_v10_g3flash_choice_conflict` | + everything in v9 · when vision `choice_analysis` marks **≥2** single-select options at confidence **≥0.95**, verifier prompt gets a **VISION AMBIGUITY FLAG** (conflicting choice labels only — not the full `choice_analysis` dict) |

### Accuracy (mean `answer_accuracy` with MCQ partial credit; strict exact in “No-judge” column)

| Run | n | Mean acc. | Exact (≥ 0.999) | Δ vs baseline |
|-----|---:|----------:|----------------:|---------------|
| `no_legend_grounding` | 250 | — | **0.480** | — |
| `fixes_v1` | 250 | — | **0.504** | +2.4 pp |
| `fixes_v2` | 250 | — | **0.516** | +3.6 pp |
| `fixes_v3` | 250 | — | **0.516** | +3.6 pp |
| `fixes_v4_g3flash` | 250 | — | **0.560** | +8.0 pp |
| `fixes_v5_multiselect` | 250 | **0.694** | **0.624** | +14.4 pp |
| `fixes_v6_conf_gate` | 250 | **0.610** | **0.624** | +14.4 pp *(repaired)* |
| `fixes_v7_g3flash_conf_gate` | 250 | **0.696** | **0.628** | **+14.8 pp** |
| `fixes_v8_g3flash_color_area` | **1,250** | **0.712** | **0.651** | **+17.1 pp** |
| `fixes_v9_g3flash_related_sents` | **1,250** | **0.713** | **0.646** | **+16.6 pp** |
| `fixes_v10_g3flash_choice_conflict` | **1,250** | **0.711** | **0.642** | **+16.2 pp** |

Per-type **mean** `answer_accuracy` (**n = 1,250** each):
- v8: **mcq 0.725** (1,150), **standard 0.570** (100)
- v9: **mcq 0.725** (1,150), **standard 0.570** (100)
- v10: **mcq 0.722** (1,150), **standard 0.580** (100)

At full scale v8, v9, and v10 are within **±0.2 pp mean acc.** of each other; per-type breakdowns are essentially indistinguishable. Earlier v9-vs-v8 deltas reported on a 784-sample partial eval were selection bias from fast-finishing samples and disappear once the full **1,250** rows are scored.

The earlier **standard 0.70 on n = 20** (250-ID slice) was a small-sample artifact; v7 vs v8 on the **paired 250 overlap** differ by **−0.2 pp** mean acc. (essentially tied).

**Paired strict correctness (`answer_accuracy == 1`) on the full 1,250-ID overlap (continuity-corrected McNemar with scipy `chi2.cdf`):**
- v10 vs v8 — **χ² = 0.90**, **p = 0.34** (no significant change)
- **v9 vs v8** — **χ² = 0.34**, **p = 0.56** (no significant change; v9-only wins = 50, v8-only wins = 57)
- **v9 vs v10** — **χ² = 0.10**, **p = 0.75** (no significant change)

All three runs (v8, v9, v10) sit inside the same paired-significance band — additional verifier context (`related_sentences`, choice-conflict flag) doesn't move strict accuracy at this n. Their *behavior* differs (verifier verdict mix, latency tail — see below), but headline mean / exact accuracy does not.

*Mean `answer_accuracy` is rule-based (numeric tolerance + MCQ partial credit). The “Exact” column is the share of rows with `answer_accuracy ≥ 0.999`. Auxiliary rubric dimensions from `gemini-2.5-flash-lite` are stored as `judge_*` fields (explanation quality, hallucination rate, etc.) — separate from `answer_accuracy`. v6 mean score is lower than v5 due to minor differences in the repaired MEP set; exact scores match v5.*

**FinMME paper baseline (full 11k eval):** Gemini Flash 2.0 = 51.85% avg. Our v7 on 250 samples = **69.6%** mean acc. (+17.8 pp vs paper headline, different metric family).

### MAX_TOKENS truncation

| Run | Samples truncated | % of run |
|-----|------------------:|---------:|
| `no_legend_grounding` | 167 | 66.8 % *(n = 250)* |
| `fixes_v1` | 129 | 51.6 % |
| `fixes_v2` | 6 | 2.4 % |
| `fixes_v3` | 3 | 1.2 % |
| `fixes_v4_g3flash` | 9 | 3.6 % |
| `fixes_v5_multiselect` | 16 | 6.4 % |
| `fixes_v6_conf_gate` | 21 | 8.4 % |
| `fixes_v7_g3flash_conf_gate` | 20 | 8.0 % |
| `fixes_v8_g3flash_color_area` | **184** | **14.7 %** *(n = 1,250)* |
| `fixes_v9_g3flash_related_sents` | **199** | **15.9 %** *(n = 1,250 MEPs; vision/verifier `tool_trace` `finish_reason`)* |
| `fixes_v10_g3flash_choice_conflict` | **183** | **14.6 %** *(n = 1,250)* |

Raising token limits in `fixes_v1` reduced truncation modestly. Disabling thinking tokens in `fixes_v2` cut it to near-zero; `fixes_v3` halves it further. Later runs are slightly higher due to richer prompts (multi-select blocks, caption injection, verifier tools). On the **1,250-ID** v8 eval, **14.7%** of MEPs hit MAX_TOKENS in at least one traced stage (vs **8.0%** on the 250-ID v7 row) — expect slice and prompt-stack differences; the long-tail risk remains real. v9 shows a slightly higher hit rate (**15.9%**); v10 is back in line with v8 (**14.6%**).

### UNANSWERABLE over-refusal

| Run | UA predicted | Notes |
|-----|-------------|-------|
| `no_legend_grounding` | 46 | High over-refusal |
| `fixes_v1` | 34 | Verifier pushback helps |
| `fixes_v2` | 37 | — |
| `fixes_v3` | 38 | — |
| `fixes_v4_g3flash` | 5 | Forced-choice retry eliminates most MCQ UNANSWERABLE |
| `fixes_v5_multiselect` | 4 | — |
| `fixes_v6_conf_gate` | 0 | — |
| `fixes_v7_g3flash_conf_gate` | 2 | — |
| `fixes_v8_g3flash_color_area` | **14** *(n = 1,250)* / 2 *(n = 250)* | At 1,250 IDs, a small rise in UNANSWERABLE predictions vs the 250-ID snapshot; still low vs early runs |
| `fixes_v9_g3flash_related_sents` | **7** *(n = 1,250)* | Matches v10; both halve v8's UA rate, consistent with the new caption / source-sentence framing giving the verifier more text to ground on |
| `fixes_v10_g3flash_choice_conflict` | **7** *(n = 1,250)* | Slightly lower than v8's 14 on the same ID slice |

### Verifier verdicts

| Run | Confirmed | Revised | Error | Gated |
|-----|-----------|---------|-------|-------|
| `no_legend_grounding` | 175 | 73 | 2 | — |
| `fixes_v1` | 196 | 54 | 0 | — |
| `fixes_v2` | 205 | 41 | 4 | — |
| `fixes_v3` | 207 | 43 | 0 | — |
| `fixes_v4_g3flash` | 193 | 50 | 7 | — |
| `fixes_v5_multiselect` | 179 | 70 | 1 | — |
| `fixes_v6_conf_gate` | 193 | 52 | 5 | 0 *(bug: 0.0 default gated all revisions; repaired)* |
| `fixes_v7_g3flash_conf_gate` | 180 | 66 | 4 | 1 |
| `fixes_v8_g3flash_color_area` | **986** | **234** | **30** | 0 *(n = 1,250)* |
| `fixes_v9_g3flash_related_sents` | **999** | **215** | **36** | 0 *(n = 1,250)* |
| `fixes_v10_g3flash_choice_conflict` | **982** | **247** | **21** | 0 *(n = 1,250)* |

**Verifier effectiveness (v7, n = 250):**

- Revised samples: **66**
- Exact accuracy on revised subset: **50.0%** (33/66)
- Exact accuracy on non-revised subset: **67.4%** (124/184)

**Verifier effectiveness (v8, n = 1,250):**

- Revised samples: **234**; exact on revised subset **55.6%**
- Confirmed samples: **986**; exact on confirmed subset **68.2%**
- At scale, revisions track harder items but are no longer a “coin flip” the way the tiny v7 revised slice suggested.

**Verifier effectiveness (v9, n = 1,250):**

- Revised samples: **215**; exact on revised subset **52.1%** (mean **62.6%**)
- Confirmed samples: **999**; exact on confirmed subset **68.0%** (mean **73.8%**)
- Adding `related_sentences` shifts the verifier toward **slightly more confirmations** (999 vs v8's 986) and **fewer revisions** (215 vs 234) — the new textual signal seems to resolve some uncertainty in place rather than triggering a revision. Revised-subset exact dips ~3.5 pp vs v8 but mean is essentially flat (62.6% vs 62.6%), suggesting the marginal revisions are partial-credit cases rather than fresh wins.

**Verifier effectiveness (v10, n = 1,250):**

- Revised samples: **247**; exact (≥ 0.999) on revised subset **47.8%**
- Confirmed samples: **982**; exact on confirmed subset **69.0%**
- Mean `answer_accuracy` is almost unchanged vs v8 (**71.08%** vs **71.24%**), but the revised tail is **weaker** under exact-match accounting — worth monitoring if v10’s ambiguity flag invites extra revisions.

### Latency (250 samples for v1–v7; v8 / v9 / v10 at 1,250)

| Run | p50 (s) | p95 (s) | mean (s) |
|-----|---------|---------|----------|
| `no_legend_grounding` | 18.9 | 27.4 | 21.6 |
| `fixes_v1` | 21.2 | 38.9 | 24.4 |
| `fixes_v2` | 15.4 | 27.1 | 21.6 |
| `fixes_v3` | 18.7 | 30.7 | 20.1 |
| `fixes_v4_g3flash` | 30.4 | 232.8 | 46.5 |
| `fixes_v5_multiselect` | 32.0 | 247.7 | 49.0 |
| `fixes_v6_conf_gate` | 33.6 | 256.7 | 52.4 |
| `fixes_v7_g3flash_conf_gate` | 32.3 | 246.1 | 41.5 |
| `fixes_v8_g3flash_color_area` | **30.1** | **209.4** | **43.9** *(n = 1,250)* |
| `fixes_v9_g3flash_related_sents` | **31.0** | **87.5** | **43.7** *(n = 1,250)* |
| `fixes_v10_g3flash_choice_conflict` | **29.6** | **222.9** | **44.2** *(n = 1,250)* |

`fixes_v1` is slower than baseline (legend grounding adds a stage). `fixes_v2` is fastest (thinking tokens off). v4–v7 improve accuracy substantially but increase latency and tail risk due to richer prompts and additional verification logic. On the **250-ID** v8 snapshot, median latency dipped vs v7 while mean rose slightly. On **n = 1,250**, v8 shows **mean 43.9 s**, **p50 30.1 s**, **p95 209.4 s** — same long-tail story, slightly lower mean than the 250-ID v8 snapshot (46.9 s) partly because more cached judge scores shortened some eval paths. **v10** is similar (**mean 44.2 s**, **p50 29.6 s**, **p95 222.9 s**).

**v9 is the clear latency winner**: **mean 43.7 s**, **p50 31.0 s**, but **p95 only 87.5 s** — a **~2.4× tighter tail** than v8/v10 (209–223 s). Adding `related_sentences` and reframing the caption as a cross-check seems to give the verifier enough textual grounding that it converges without long retry chains; the long tail of v8 where the verifier kept thrashing on ambiguous charts collapses. This is the strongest unambiguous v9 win even though headline accuracy is tied with v8.

### Failure taxonomy by question type (v7)

From `taxonomy_finmme_train_fixes_v7_g3flash_conf_gate.jsonl` joined with metrics:

- Total failures: **93** (`mcq`: 87, `standard`: 6)
- Top overall failure types:
  - `question_misunderstanding`: **25** (26.9%)
  - `legend_confusion`: **18** (19.4%)
  - `extraction_error`: **17** (18.3%)

**MCQ failure composition (n=87):**

| Failure type | Count | Share |
|-------------|-------|-------|
| question_misunderstanding | 23 | 26.4% |
| extraction_error | 17 | 19.5% |
| legend_confusion | 16 | 18.4% |
| hallucinated_element | 12 | 13.8% |
| other | 9 | 10.3% |
| axis_misread | 6 | 6.9% |
| unanswerable_failure | 4 | 4.6% |

Most residual errors in v7 are therefore comprehension/grounding issues in MCQ, not UNANSWERABLE over-refusal.

### Fair baseline check: Gemini-3 zero-shot vs agent

`output/baselines/metrics_finmme_train_zeroshot_structured_gemini_gemini_3_flash_preview.jsonl` now holds **11,099** FinMME train rows (Gemini-3 Flash Preview structured zero-shot, parser-repaired). The file was initially under-parsed (`parse_ok=false` for 142 rows on an early 250-row snapshot); after repair, a **historical** strict-exact snapshot on **250** IDs was **52.8%** zero-shot vs **62.8%** agent v7 (**+10.0 pp** exact) — useful as an ablation-era record; backups live next to the baseline file.

**Matched overlap (recommended for the paper):** the **1,250** `sample_id`s in **`fixes_v8_g3flash_color_area`** / **`fixes_v9_g3flash_related_sents`** / **`fixes_v10_g3flash_choice_conflict`** metrics all exist in the full zero-shot file. On that overlap (same rule-based `answer_accuracy`, same judge stack for the agent traces):

| System | Mean `answer_accuracy` | Exact (≥ 0.999) |
|--------|----------------------:|----------------:|
| Agent `fixes_v8_g3flash_color_area` | **71.24%** | **65.12%** |
| Agent `fixes_v9_g3flash_related_sents` | **71.28%** | **64.56%** |
| Agent `fixes_v10_g3flash_choice_conflict` | **71.08%** | **64.24%** |
| Zero-shot Gemini-3 preview (structured) | **63.56%** | **56.40%** |
| **Δ (v8 agent − zero-shot)** | **+7.68 pp** | **+8.72 pp** |
| **Δ (v9 agent − zero-shot)** | **+7.72 pp** | **+8.16 pp** |
| **Δ (v10 agent − zero-shot)** | **+7.52 pp** | **+7.84 pp** |

Paired **McNemar** test on strict correctness (binary strict correct = `answer_accuracy == 1` vs `< 1`, scipy `chi2.cdf`):
- **v8** vs zero-shot — **χ² = 68.21**, **p ≈ 1.1 × 10⁻¹⁶**
- **v9** vs zero-shot — **χ² = 61.45**, **p ≈ 4.5 × 10⁻¹⁵**
- **v10** vs zero-shot — **χ² = 57.37**, **p ≈ 3.6 × 10⁻¹⁴**

All three are overwhelmingly significant against zero-shot. Pairwise between the agents, all comparisons are **not significant** (v9 vs v8 p = 0.56, v10 vs v8 p = 0.34, v9 vs v10 p = 0.75) — at this n, accuracy is statistically tied across v8 / v9 / v10.

**By `question_type` on the same 1,250-ID overlap (v8 / v9 / v10 vs zero-shot):**
- mcq (n = 1,150): **+8.09 pp / +8.13 pp / +7.83 pp** mean acc.
- standard (n = 100): **+3.00 pp / +3.00 pp / +4.00 pp** — treat the standard slice as **directional only** (wide uncertainty).

(v9 full-scale numbers are now in the table above; the historical "v9 partial @ n = 784" snapshot was a fast-finishing-sample selection bias and is no longer the operating estimate.)

**Full-file zero-shot context (not matched to v8 / v10):** on all **11,099** train rows, mean zero-shot acc. is **~61.5%**, with **standard** questions much harder (**~24%** mean) than in the first-1,250-ID slice — so scaling the agent to the full train split may see a lower standard-type floor than this overlap suggests.

### Version legend

| Tag | Model (planner/vision/verifier) | OCR / judge |
|-----|---------------------------------|-------------|
| `no_legend_grounding` | gemini-2.5-flash-lite | gemini-2.5-flash-lite |
| `fixes_v1` | gemini-2.5-flash-lite | gemini-2.5-flash-lite |
| `fixes_v2` | gemini-2.5-flash-lite | gemini-2.5-flash-lite |
| `fixes_v3` | gemini-2.5-flash-lite | gemini-2.5-flash-lite |
| `fixes_v4_g3flash` | **gemini-3-flash-preview** | gemini-2.5-flash-lite |
| `fixes_v5_multiselect` | gemini-2.5-flash-lite | gemini-2.5-flash-lite |
| `fixes_v6_conf_gate` | gemini-2.5-flash-lite | gemini-2.5-flash-lite |
| `fixes_v7_g3flash_conf_gate` | **gemini-3-flash-preview** | gemini-2.5-flash-lite |
| `fixes_v8_g3flash_color_area` | **gemini-3-flash-preview** | gemini-2.5-flash-lite |
| `fixes_v9_g3flash_related_sents` | **gemini-3-flash-preview** | gemini-2.5-flash-lite |
| `fixes_v10_g3flash_choice_conflict` | **gemini-3-flash-preview** | gemini-2.5-flash-lite |

v1–v7: **250** FinMME train samples; v8 / v9 / v10 all evaluated on the same **1,250** train samples (`gemini_gemini` config, FinMME train split).

### Color-area stage activation (v8)

`color_area_tool` is gated by chart type, legend availability, and ambiguity flags, so it does not fire on every sample.

| Metric | v8 @ 250 IDs (snapshot) | v8 @ 1,250 IDs |
|---|---:|---:|
| Color-area stage triggered | 12 (4.8%) | **62 (5.0%)** |
| ↳ flagged `color_ambiguity` (hint suppressed) | 10 | **52** |
| ↳ flagged `low_confidence` (hint suppressed) | 2 | **3** |
| Legend compliance retries fired | 5 | **165** *(13.2% of 1,250 MEPs)* |

The stage is conservatively gated — many charts (lines, multi-axis, etc.) never qualify — but at **n = 1,250** it fires on **~5%** of samples, not “effectively never.” **Paired v7 vs v8 on the same 250 IDs** is still **≈0 pp** mean acc. (see `metrics_*.n250` backups). The headline **0.712** at **n = 1,250** is mainly a **more stable MCQ estimate** (**72.5%** on 1,150 mcq) than the small 250-ID window suggested; it is **not** a large paired gain over v7 on identical IDs.

### Key takeaways

- **+14.8 pp accuracy** over baseline achieved by `fixes_v7` (48.0% → 62.8% no-judge; **69.6% judge-based**). `fixes_v8` matches v7 within noise (69.4% judge / 62.8% no-judge); the color-area hint is correct in direction but only activates on ~5% of the slice, so it doesn't move the headline number.
- **Multi-select MCQ**: largest single-run gain was v5 (+8.8 pp on multiple_choice type, 30.2% → 53.5%).
- **UNANSWERABLE over-refusal**: 46 → 2 on the 250-ID ladder (forced-choice retry in v4); **14 / 1,250** on v8; **7 / 1,250** on both v9 and v10 (still low vs early runs; the v9/v10 verifier prompts cut UA-confirmations in half by giving the model more text to ground on).
- **MAX_TOKENS**: near-eliminated by v2–v3 on the ladder; later prompts add some tail risk again (**184** hits at v8, **199** v9, **183** v10 — all at **n = 1,250** MEPs by `tool_trace` `finish_reason`).
- **Fair same-model baseline (primary):** on the **1,250-ID** overlap with full Gemini-3 zero-shot train eval, all three g3flash agents beat zero-shot by ~7.5–7.7 pp mean acc. (v8 **+7.68 / +8.72 pp**; v9 **+7.72 / +8.16 pp**; v10 **+7.52 / +7.84 pp** — mean / exact). All three p ≤ 4.5 × 10⁻¹⁵ vs zero-shot. **Pairwise between agents, no comparison is significant** (v9 vs v8 p = 0.56, v10 vs v8 p = 0.34, v9 vs v10 p = 0.75). Legacy **250-ID** strict-exact headline was **+10.0 pp** (v7 vs repaired zero-shot snapshot) — cite as historical ablation only.
- **v9 / v10 verifier prompts:** v9 adds analyst `related_sentences` + caption-as–cross-check framing; v10 adds a **choice-conflict** meta-signal when two high-confidence single-select options collide. At full n = 1,250 both are statistically tied with v8 on mean / exact accuracy (paired McNemar p ≥ 0.34); v9's distinctive win is a **2.4× tighter latency tail** (p95 87.5 s vs v8's 209.4 s).
- **Confidence gate**: designed to block low-confidence verifier revisions; only fires when model explicitly returns `confidence < 0.75` (1 gate in v7). Bug in v6 where `confidence` was required in parse keys caused all revisions to fail; repaired in-place.
- **Model upgrade (g3flash)**: v4, v7, v8, v9, and v10 use Gemini 3 Flash Preview — contributes meaningful gains over 2.5 Flash Lite.
- **Verifier**: at v8 **n = 1,250**, revised rows remain harder but revised-subset exact rises to **~55.6%** (vs **50%** on 66 v7 revised IDs). **v10** increases revisions (**247** vs **234**) but **exact** on revised rows drops to **~47.8%** while headline mean acc. stays within **~0.2 pp** of v8 — monitor if the ambiguity flag trades calibration for exactness.
- **OpenCV color-area (v8)**: opt-in stage between legend grounding and vision; **~5%** trigger rate at 1,250 IDs; `MEPColorArea` trace for audits.
- **vs. paper baseline**: v7 **0.696** mean acc. on 250 IDs vs Gemini Flash 2.0 from FinMME paper (51.85%) = **+17.8 pp** (different metric family than the paper table).

Full per-sample analysis, flip tables, taxonomy breakdowns, and charts are in `notebooks/results_analysis.ipynb` § 14–16.
For paper-ready headline numbers, see `markdown/camera_ready_metrics.md`.

---

## 9. Caveats for the report

1. **Different n across configs** — ChartQAPro row counts differ slightly by config; comparisons should note which subset is used.  
2. **FinMME 233 vs 250** — Zero-shot baseline does not cover all 250 planner-v2 IDs; the fair three-way comparison uses **233** overlapping samples.  
3. **FinMME v8 / v9 / v10 vs ladder (n)** — v1–v7 rows in §8b are **250** train IDs; **`fixes_v8`**, **`fixes_v9`**, and **`fixes_v10`** each ship **1,250** `metrics_*.jsonl` rows on the same ID slice — do not mix headline tables with the smaller ladder rows without a `sample_id` join.  
4. **Slices vs full test** — The **250-sample** ChartQAPro block is a *controlled subset*; headline full-test numbers are in §1–2.  
5. **Refresh** — Re-execute `results_analysis.ipynb` after new runs to update tables and PDFs.

---

*Generated to support an initial results memo; align wording with your paper’s definitions of accuracy and splits.*
