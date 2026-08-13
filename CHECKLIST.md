# GCQ — Paper Checklist (ODI 2026, deadline Aug 29 AoE)

Living checklist for the whole paper. Ticked items are done and verified.

## Scoping & story
- [x] Story locked: blind spot → warning sign → measure the gap → one training-free fix
- [x] Scope cut to one model (Qwen3-VL-2B), one intervention (sensitivity-guided mixed precision), one protocol
- [x] GCL fine-tune + grounding-calibrated PTQ moved to deferred/future work
- [x] VQA-preservation tension made structural (training-free = nothing can be forgotten)

## Writing
- [x] Full proposal document written and adversarially reviewed (`GCQ-Training-Free.md`)
- [x] Abstract finalized (spatial-understanding-centered, ~260 words, deployment + future-research stakes)
- [x] Evaluation-pipeline subsection added (prompt → JSON box → IoU; objects: RefCOCO/+/g referring expressions, ODinW OOD categories)
- [x] Compute/feasibility updated to real numbers (~60–75 GPU-h, 4-week solo plan)
- [x] LaTeX paper started: `paper/main.tex` — all sections ported, compiles with or without official style file
- [ ] Download official `neurips_2026.sty` and compile with it
- [ ] Fill bibliography TODOs (author lists for recent preprints)
- [x] Preliminary results inserted: real Table 1 + findings paragraph + Figures 1-2 (paper compiles, 7 draft pages pre-compression)
- [x] Figure 2 done (ahead of schedule — was camera-ready item)
- [ ] Compress to ≤5 pages under the official style
- [ ] Final proofread pass (terminology: grounding/spatial precision consistent; all §-refs valid)

## Experiments (from plan-4-weeks.md)
- [x] Week 0a: environment verified — cluster `spring-2026-pyt` env (Python 3.12, torch 2.9.1+cu128, transformers 5.15.0 with native `qwen3_vl`); job owns 3 idle 46 GB A6000s (physical 4/5/6); `code/env.sh` helper; zero disk cost
- [x] Week 0b: Qwen/Qwen3-VL-2B-Instruct verified to exist, downloaded (4.0 GB → projectnb HF cache), **grounding smoke test 3/3 PASS** (JSON-array-in-code-fence output format noted for parser)
- [x] Week 0c: annotations downloaded + verified — RefCOCO/+/g via jxu124 HF mirror (UNC server is dead); split counts match published numbers exactly; **refcocog = umd split confirmed**; COCO-2014 annotations (232 MB); VQAv2 val Q+A; POPE random/popular/adversarial
- [x] Week 0d: frozen subsets built, seed 0 — eval 1k + D_dev 1k (image-disjoint, verified by assert), D_probe 512 (excludes all 6,549 cross-variant val/test images), VQA 5k; image manifest = 778 train2014 + 4,994 val2014
- [x] Week 0e: git repo initialized in ~/GCQ; code + frozen subsets committed
- [x] Week 0f: subset images fetched — all 5,772, zero failures, 880 MB (+423 VQA-probe images later)
- [x] Week 1: eval harnesses written + sanity-validated — REC (acc@0.5, GIoU, parse-fail, size-stratified; sanity 32 samples: **84.4% acc, 0 parse failures** — plausible vs published 2B numbers) and VQA/POPE (soft accuracy, F1); per-sample JSONL + one-CSV-row-per-run logging
- [x] Week 1: simulated RTN quantizer written (`quant_utils.py`) — groupwise symmetric, LLM blocks only, vision tower/embeddings/lm_head untouched; supports per-module promotion overrides (ready for Week-3 allocation)
- [x] Week 1: **throughput bug found and fixed** — Qwen3-VL's vision patch-embed Conv3d (kernel==stride) hit a catastrophic cuDNN path (~4.5 s/image); replaced with its exact GEMM equivalent (`gcq_patches.py`): 140× prefill speedup, task metrics reproduce bit-for-bit-level (84.38% acc, ΔGIoU 0.0003, 0 parse fail); patch applied identically to ALL configs so comparisons are unaffected — document in paper's eval note
- [x] Week 1: REC baselines DONE (n=1000 each) — BF16 **88.9%**, W8-RTN 88.7% (lossless ✓), **W4-RTN 84.5%** (−4.4 pts, R=95.0%); replicated on D_dev: 89.0→83.9 (−5.1 pts) — consistent across independent 1k samples; grounding is measurably damaged at W4 even before VQA comparison
- [x] Week 1: D_dev selection-set baselines done (Week-3 prerequisite)
- [x] Week 1: VQA + POPE chains DONE — VQA BF16 81.1 / W4 78.0 / W3 1.3 (collapse) / floor 36.8; POPE BF16 89.4 / W4 88.8 / W3 = exact 50% floor / floor 50.0. **Full W4-RTN floor-corrected retention gradient: POPE 98.5% > grounding 94.9% > VQA 92.8%** — capabilities degrade in proportion, VQA most affected; no grounding-first collapse under weight-only RTN
- [x] Week 1: REC W3-RTN = global model collapse (gibberish; VQA confirms) — RTN unusable below W4; GPTQ arm needed for meaningful W3
- [x] Week 2: **first complete grounding-gap datapoint (W4-RTN)** — floor-corrected retention: grounding 94.9% vs VQA 92.8% → **no grounding-first asymmetry at W4-RTN**; pre-registered contrast framing engaged ("quantization ≠ pruning"); caveats: RTN-only, large objects only, W4A8/GPTQ untested
- [ ] Week 2: GPTQ arm of the grid (toolchain install staged) + W4A8 stress condition
- [x] Week 3: **sensitivity map COMPLETE (56/56 groups, 512-sample probe)** — strong structure: localization sensitivity concentrates in a contiguous **mid-network attention band (layers 10–17)**; H2's "non-uniform" half confirmed; Figure 2 data ready
- [x] Week 3: greedy allocation at B=4.25 — 7 attention groups (layers 10,12–17), 88.1M params, exactly 4.250 avg bits; 3 matched-budget random controls generated
- [x] Week 3: **GCQ vs controls — CORE CLAIM ESTABLISHED**: GCQ B=4.25 = 86.6% (+2.1 over uniform W4) vs random-promotion mean 84.87% (+0.37, 3 seeds) at IDENTICAL 88.1M-param budget — ~5–6× the random effect; B=4.5 = 87.3% (64% of loss recovered, monotonic budget curve); **VQA constraint passed with margin (78.4 vs baseline 78.0 — GCQ slightly HELPS VQA)**
- [x] Week 3: H3 verdict (honest): partial recovery — 48%/64% at +6%/+12.5% LLM memory; 90% bar not reached training-free at ≤4.5 bits → GCL fine-tune remains the pre-registered escalation for the full paper
- [x] Week 3: **A1 ρ RESULT — Spearman ρ(grounding-sens, VQA-sens) = 0.001 across 56 groups; top-7 overlap = 0/7** — the two capability maps are statistically independent (grounding: mid-network attention band; VQA: early network). "Localization sensitivity ≠ general sensitivity" is empirical fact; H2 confirmed at maximum strength
- [x] Week 3: **A2 allocation-policy comparison COMPLETE** at matched 88.1M-param budget: grounding-driven 86.6 > VQA-driven 85.1 > random 84.9 > uniform 84.5 — the capability you profile is the capability you protect; VQA-driven ≈ random for grounding, as ρ=0.001 predicted
- [x] Week 3: symmetric check — VQA-driven-on-VQA 78.60 (+0.65) vs grounding-driven-on-VQA 78.41 (+0.46): each capability is best protected by its own map; full cross-matrix consistent with ρ=0.001
- [ ] Week 3: proxy validation via promote-one decoded REC (RUNNING — 9 modules spanning s_m range, 500-sample decoded each; allocation-level validation already implicit in GCQ-vs-random)
- [ ] Week 2/4: GPTQ/AWQ arm of the grid (blocked on gptqmodel import shim) + W4A8 stress
- [x] Week 4: Figures 1+2 rendered from real data (CVD-safe Okabe-Ito, visually inspected, collision-fixed) and wired into main.tex — includes the bonus finding that layer-15 attention is grounding-critical but VQA-negative
- [ ] Week 4: ODinW-13 held-out slice (download + eval, touched once)
- [ ] Week 4: full testA/testB runs for headline configs (BF16, W4, GCQ 4.25/4.5)

## Rigor gates (must all hold before submission)
- [x] Novelty claims hedged ("first *systematic*", "to our knowledge") and GWQ (2411.00850) cited — verified against arXiv
- [x] Data hygiene specified: D_probe/D_dev/test disjoint; RefCOCOg umd split; cross-variant image exclusion
- [x] Profiling direction correct (promote-one-from-fully-quantized, not demote-one-from-BF16)
- [x] Floor-corrected retention R′ defined; GIoU never reported as a ratio
- [x] VQA constraint pre-registered (1.5 pts, paired test, fixed ~5k items)
- [x] Kill criteria pre-registered for H1/H2/proxy/method-ceiling
- [ ] Re-run scooping searches within 48h of submission ("grounding-aware quantization VLM", "RefCOCO quantized VLM", etc.)
- [ ] Anonymization pass (PDF metadata included); double-blind check of appendix

## Submission mechanics
- [ ] OpenReview account + ODI 2026 venue located (link in workshop CFP)
- [ ] LLM-usage disclosure section present (drafted in main.tex — verify against final NeurIPS policy wording)
- [ ] Submit by Aug 29, 2026, 23:59 AoE — target Aug 27 to leave slack
- [ ] Post-submission: archive exact submitted PDF + code state (tag `submission-odi2026`)

## Post-deadline (Weeks 3–4 → camera-ready / full paper)
- [ ] Complete results version of the paper (all figures/tables real)
- [ ] Code + frozen subsets + checkpoint released; model card includes grounding numbers
- [ ] Notification Sept 29 → if accepted: poster + possible 15-min oral prep
