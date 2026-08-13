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
- [ ] Insert preliminary results: Figure 1 + partial Table 1 (needs Week-1/2 runs)
- [ ] Figure 2 (sensitivity map) — optional for submission, required for camera-ready
- [ ] Compress to ≤5 pages under the official style
- [ ] Final proofread pass (terminology: grounding/spatial precision consistent; all §-refs valid)

## Experiments (from plan-4-weeks.md)
- [x] Week 0a: environment verified — cluster `spring-2026-pyt` env (Python 3.12, torch 2.9.1+cu128, transformers 5.15.0 with native `qwen3_vl`); job owns 3 idle 46 GB A6000s (physical 4/5/6); `code/env.sh` helper; zero disk cost
- [x] Week 0b: Qwen/Qwen3-VL-2B-Instruct verified to exist, downloaded (4.0 GB → projectnb HF cache), **grounding smoke test 3/3 PASS** (JSON-array-in-code-fence output format noted for parser)
- [x] Week 0c: annotations downloaded + verified — RefCOCO/+/g via jxu124 HF mirror (UNC server is dead); split counts match published numbers exactly; **refcocog = umd split confirmed**; COCO-2014 annotations (232 MB); VQAv2 val Q+A; POPE random/popular/adversarial
- [x] Week 0d: frozen subsets built, seed 0 — eval 1k + D_dev 1k (image-disjoint, verified by assert), D_probe 512 (excludes all 6,549 cross-variant val/test images), VQA 5k; image manifest = 778 train2014 + 4,994 val2014
- [x] Week 0e: git repo initialized in ~/GCQ; code + frozen subsets committed
- [ ] Week 0f: subset images fetched (5,772 targeted images ≈ 1 GB instead of 19 GB of COCO zips — in progress)
- [ ] Week 0g: ODinW slice (deferred to Week 4 per plan)
- [x] Week 0f: subset images fetched — all 5,772, zero failures, 880 MB
- [x] Week 1: eval harnesses written + sanity-validated — REC (acc@0.5, GIoU, parse-fail, size-stratified; sanity 32 samples: **84.4% acc, 0 parse failures** — plausible vs published 2B numbers) and VQA/POPE (soft accuracy, F1); per-sample JSONL + one-CSV-row-per-run logging
- [x] Week 1: simulated RTN quantizer written (`quant_utils.py`) — groupwise symmetric, LLM blocks only, vision tower/embeddings/lm_head untouched; supports per-module promotion overrides (ready for Week-3 allocation)
- [ ] Week 1: BF16 baselines + image-blind floor runs (RUNNING — 3 GPU chains in parallel)
- [ ] Week 1: first W4 quantized eval → first grounding-gap datapoint (RUNNING — W4-RTN queued in same chains; GPTQ checkpoint to follow)
- [ ] Week 2: full measurement grid {RTN,GPTQ,AWQ} × {W8,W4,W3} on frozen subsets
- [ ] Week 2: floor-corrected retention + Figure 1
- [ ] Week 3: sensitivity profiling (56 promote-one configs, coordinate-KL proxy)
- [ ] Week 3: proxy validation (Spearman ρ vs decoded REC on 8–10 modules; kill criterion)
- [ ] Week 3: greedy allocation at B=4.25 + GCQ checkpoint + paired VQA/POPE constraint check
- [ ] Week 4: equal-memory controls (random ×3, VQA-driven, uniform)
- [ ] Week 4: A1 Spearman ρ (grounding vs VQA sensitivity — the scientific heart)
- [ ] Week 4: held-out ODinW-13 (touched once, at the end)
- [ ] Week 4: full testA/testB runs for headline configs only

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
