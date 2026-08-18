# GCQ — Paper Checklist (results-first revision)

Living checklist for the whole paper. Ticked items are done and verified.

## Results-improvement sprint — reopened 2026-08-14

The goal is **higher quantized-model grounding accuracy**, not a larger BF16→W4
drop.  Keep one coherent story: quantization erodes fine spatial precision;
quantizer-matched GCQ protects the responsible weights; a small recovery adapter
is added only if it closes substantially more of the remaining gap without
hurting general capability.

### Paper-level success gates
- [x] Preserve the training-free GCQ result as the base contribution; treat recovery training as a clearly labeled second stage
- [ ] Final training-free GCQ beats uniform W4 at matched actual bytes on REC, GIoU, and precise-IoU AUC, with image-clustered 95% CIs excluding zero
- [ ] Final method keeps VQAv2 and POPE within 1.5 absolute points of its corresponding untrained quantized base
- [ ] Lead size claim uses all predeclared relative-area quartiles on a credible full split, never the post-hoc 17→8 ODinW count alone
- [ ] Evaluate the frozen final configuration once on untouched RefCOCO+ and corrected ODinW; do not tune on those results

### Validity repairs — required before new headline numbers
- [x] Fix ODinW sampling so `--images N` returns exactly N images (or `--images 0` runs all), and log image/query/GT counts
- [x] Remove the 20-prediction truncation, flag generation truncation, and use order-independent maximum-cardinality matching at IoU≥0.5
- [x] Add outcome-independent relative target-area quartiles with per-quartile scores/counts; use these instead of absolute-pixel bins in the final analysis
- [x] Implement paired image-clustered bootstrap intervals (10,000 resamples); historical full-split gains remain significant for REC, GIoU, and precise-IoU AUC
- [x] Keep REC@0.5 and mean GIoU primary; add mean accuracy over IoU 0.50:0.05:0.95 as the precision-sensitive secondary metric
- [x] Historical precise-IoU reanalysis: GCQ−W4 is +3.01 points on testA (95% CI +2.43,+3.60) and +2.88 on testB (+2.20,+3.58); label this post-hoc until untouched confirmation
- [x] Replace approximate VQAv2 scoring with official answer normalization and leave-one-annotator-out soft accuracy; require explicit yes/no POPE outputs and report malformed generations
- [x] Use one exact IoU-threshold tuple for evaluation and bootstraps, and retain full-precision IoU/GIoU in new JSONL logs
- [ ] Correct the paper's historical “500 ODinW images” and “touched once” language after the rerun

### GCQ method upgrade — one intervention, better aligned
- [ ] Generate exact GPTQ W4 and W8 candidates for every decoder projection from the same Hessian and W4-prefix activations; reuse those tensors in profiling and final checkpoints
- [ ] Build a training-only, evaluation-disjoint profiling set that contains genuine small/medium/large boxes and both referring-expression and unambiguous category-grounding prompts
- [ ] Profile `q/k/v/o` and `gate/up/down` projections separately; charge exact parameter/checkpoint bytes
- [ ] Restrict the cheap proxy to the four numeric coordinate spans, then rerank a fixed top-24 shortlist with decoded paired GIoU
- [ ] Replace one-shot ratio-greedy allocation with width-4 conditional beam/sequential selection; stop at non-positive marginal gain and report actual bits used
- [ ] Compare at matched memory against uniform W4, original coarse GCQ, random, VQA-driven, and a clearly labeled MABA-style additive allocator
- [x] Cite MABA (CVPR Findings 2026) and narrow novelty to grounding-conditioned, coordinate-specific protection and object-scale/held-out grounding evaluation

### Recovery training — implemented; corrected evaluation running
- [x] Implement deterministic 8k training data: 4.5k RefCOCO-family REC + 1.5k size-balanced COCO grounding + 2k caption replay
- [x] Reserve a separate 1k recovery-dev set; exclude all 6,549 RefCOCO-family eval images and all 499 allocation-probe images
- [x] Download and validate all 7,000 selected COCO train images; freeze manifest hashes
- [x] Implement LoRA on the frozen quantize-dequantize base (r=16, α=32, dropout=.05; all seven LLM projections; 17,432,576 trainable parameters)
- [x] Implement assistant-only coordinate-weighted CE: γ=5 on the four numeric spans, weight 1 elsewhere, normalized per example by total weight
- [x] Add adapter-safe evaluation (base → RTN/GCQ → adapter), quantization/data hashes, four independent 40GB+ GPU launchers (A6000-compatible), and automated pass/fail summary
- [x] Pass one-step 40GB+ Ampere smoke: zero-init parity, no base gradients, nonzero LoRA gradients, save/reload, and decoded generation
- [x] Run the matched four-arm pilot, seed 0: all four W4/GCQ × CE/CWCE arms completed 500/500 steps with verified gradients, hashes, and adapter tensors (training job 7184200)
- [x] Harden evaluation: pin base revision `89644892…`, rerun fresh BF16/W4/GCQ baselines, reject stale/duplicate outputs, and validate every completed adapter before launch
- [x] Make the 750 referring-expression development examples primary and report the 250 category-grounding examples separately; add paired image-clustered 10k-resample CIs and the factorial interaction diagnostic
- [x] Complete corrected single-hardware evaluation array 7184512 and write `pilot_summary.json` (partial mixed-A100/L40S array 7184503 was stopped and archived before use)
- [x] Score the step-500 pilot gate: 11/12 checks pass and GCQ+CWCE reaches 82.93 REC / 0.7438 GIoU / 0.7071 precise-IoU on the primary set, but the overall gate fails only because VQA drops 77.37→75.01, beyond the frozen 1.5-point margin
- [x] Complete L40S checkpoint array 7184550 over saved steps 100/200/300/400 on recovery-dev plus the first 1k frozen VQA rows; all tasks exited 0 and every checkpoint passed all six frozen grounding/VQA screens
- [x] Freeze step 300 by the predeclared rule and evaluate it once: grounding improves 80.53→83.73 REC (+3.20, paired 95% CI [1.47, 5.07]) and POPE improves 88.77→89.52 (+0.76, CI [0.33, 1.21]), but untouched VQA drops 77.65→75.20 (−2.45, CI [−3.20, −1.71]); retain the failed gate rather than selecting another checkpoint on the holdout
- [x] Freeze the replacement 12k recovery manifest: the unchanged 6,000 grounding rows plus 6,000 official VQAv2-train short-answer rows on 6,000 unique images, with no caption rows or recovery-dev overlap (`8bf3b6a…aec4a`)
- [x] Train the single frozen GCQ+CWCE 50/50-replay arm at 5e-5 (job 7184634): 750/750 steps in 40.3 minutes, all six predeclared checkpoints pass tensor/config/manifest validation
- [ ] Evaluate only predeclared steps 200/300/400/500/600/750 on recovery-dev plus the now-exposed VQA5k (homogeneous-L40S array 7184903 submitted; no partial scores inspected), then freeze at most one winner by the locked gates
- [x] Freeze a new 5k VQAv2-val confirmation set before retraining predictions: 5,000 questions / 4,571 images, disjoint by image from the exposed VQA5k and POPE (`416aea5…e21038`)
- [x] Freeze untouched full RefCOCO+ testA/testB expressions/results before checkpoint selection: 5,726/4,889 expressions on 750/750 images (`542fbbf…57ff83`, `fafda8c…44462`); all images validated and no recovery/profiling overlap (the COCO image partitions overlap earlier RefCOCO evaluation, so do not call the images globally unseen)
- [ ] Evaluate exactly one frozen final candidate on that confirmation set and require the paired image-bootstrap CI lower endpoint to remain above −1.5 points; if it fails, do not try a second candidate on the set
- [ ] If the gate passes, repeat the winning method and strongest control at seeds 1–2, add a BF16+CWCE ceiling, and require image-clustered paired CIs to exclude zero on untouched RefCOCO+
- [ ] Report the adapter separately: current PEFT artifacts are 69,788,264-byte FP32 checkpoints (9.90% of nominal W4 decoder storage); claim ~34.9 MB BF16 (4.96%) only after a non-overwriting export, BF16 runtime-dtype assertion, and output-parity check; never merge it for headline bit-budget claims
- [ ] If CWCE fails or forgets general ability, test BF16-teacher coordinate-weighted KL as the single escalation, with an unweighted-KL matched control

### Confirmatory experiment set — no extra benchmark sprawl
- [ ] Run BF16, uniform W4, and frozen final GCQ on full RefCOCO/+/g official splits; report REC, GIoU, precise-IoU AUC, parse failures, and quartiles
- [ ] Run only BF16, uniform W4, and frozen final GCQ on corrected full ODinW; bootstrap by image and stratify by dataset
- [ ] Run VQAv2 and POPE for every final/trained row used in a main table; test the 1.5-point non-inferiority constraint
- [ ] Keep W3/W4A8 as stress diagnostics only; do not target or cherry-pick a score in the 70s

### Explicitly deferred unless a core gate fails
- [ ] Grounding-aware calibration pilot only if the upgraded allocator still recovers <75% of the W4 loss
- [ ] Resolution/upscaling pilot only on disjoint development images if the BF16 fine-object baseline remains weak; apply the frozen setting equally to all models and report token cost
- [ ] Second model, packed-kernel latency, broad VQA suites, tiling, and multi-sample decoding remain out of the main study until the single-model causal result is complete

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
- [x] Official `neurips_2026.sty` acquired (GitHub mirror of official template; environ/trimspaces deps vendored from CTAN); paper compiles with ZERO errors under official style; main body currently ~5.8 pages → compression pass next
- [x] Bibliography complete — all 18 entries with real authors/titles verified via arXiv API (incl. correcting "LiteLVLM" to its actual title)
- [x] Preliminary results inserted: real Table 1 + findings paragraph + Figures 1-2 (paper compiles, 7 draft pages pre-compression)
- [x] Figure 2 done (ahead of schedule — was camera-ready item)
- [x] Compressed to exactly 5 pages under official neurips_2026.sty (page 6 = refs/disclosure/appendix; 0 errors); science content fully preserved
- [x] Proofread pass: 0 LaTeX errors/warnings, all labels referenced (table/figure cross-refs added), no stale "planned"/TODO text, terminology consistent

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
- [x] Week 3: **sensitivity map COMPLETE (56/56 groups, 512-sample probe)** — strong structure: localization sensitivity concentrates in a contiguous **mid-network attention band (layers 10–17)**; H2's "non-uniform" half confirmed; Figure 2 data ready
- [x] Week 3: greedy allocation at B=4.25 — 7 attention groups (layers 10,12–17), 88.1M params, exactly 4.250 avg bits; 3 matched-budget random controls generated
- [x] Week 3: **GCQ vs controls — CORE CLAIM ESTABLISHED**: GCQ B=4.25 = 86.6% (+2.1 over uniform W4) vs random-promotion mean 84.87% (+0.37, 3 seeds) at IDENTICAL 88.1M-param budget — ~5–6× the random effect; B=4.5 = 87.3% (64% of loss recovered, monotonic budget curve); **VQA constraint passed with margin (78.4 vs baseline 78.0 — GCQ slightly HELPS VQA)**
- [x] Week 3: H3 verdict (honest): partial recovery — 48%/64% at +6%/+12.5% LLM memory; 90% bar not reached training-free at ≤4.5 bits → GCL fine-tune remains the pre-registered escalation for the full paper
- [x] Week 3: **A1 ρ RESULT — Spearman ρ(grounding-sens, VQA-sens) = 0.001 across 56 groups; top-7 overlap = 0/7** — the two capability maps are statistically independent (grounding: mid-network attention band; VQA: early network). "Localization sensitivity ≠ general sensitivity" is empirical fact; H2 confirmed at maximum strength
- [x] Week 3: **A2 allocation-policy comparison COMPLETE** at matched 88.1M-param budget: grounding-driven 86.6 > VQA-driven 85.1 > random 84.9 > uniform 84.5 — the capability you profile is the capability you protect; VQA-driven ≈ random for grounding, as ρ=0.001 predicted
- [x] Week 3: symmetric check — VQA-driven-on-VQA 78.60 (+0.65) vs grounding-driven-on-VQA 78.41 (+0.46): each capability is best protected by its own map; full cross-matrix consistent with ρ=0.001
- [x] Week 3: proxy validation — honest verdict: per-module decoded effects (~0.3 pts) are below decoded noise at n=500 (rank ρ uninformative over 9 modules); proxy validity rests on the allocation-level control comparison (pre-registered acceptance rule); limitation updated in main.tex
- [x] **GPTQ arm COMPLETE (own validated implementation)**: W8 86.2 (lossless, matched-500) / W4 85.4 (beats RTN +0.9; proportional degradation REPLICATES: grounding 95.9% vs VQA 93.2% floor-corrected) / **W3: grounding at ABSOLUTE ZERO (100% parse fail) while VQA retains partial varied function (17.7%, not yes-bias — verified) → grounding dies FIRST at 3-bit, the original hypothesis confirmed in its precise late-stage form**; GCQ-on-GPTQ 84.2 < uniform 85.4 — promotion gain does not transfer to GPTQ (its error compensation already absorbs it; honest finding)
- [x] Anomaly investigation CLOSED (3-check protocol): promote path bit-exact at both endpoints; maps are quantizer-specific (ρ=0.082); **GPTQ-specific allocation: 85.5 acc / 0.768 GIoU / 0% parsefail — repairs the mismatch (+1.3) and recovers 44% of GIoU loss**; refined recipe: profile on the deployed quantizer
- [x] W4A8 stress: grounding 92.9% vs VQA 91.8% floor-corrected — proportional again; measurement table complete across all conditions
- [x] **Historical v1 ODinW size result (reopened above)**: W4 recall loss small −53% / medium −26% / large −0%; this result motivated the new size analysis but is not final because the sampler returned 376 rather than 500 images and the smallest bin had only 92 boxes
- [x] Improvement: budget curve extended — B=4.25/4.5/4.75/5.0 → 86.6/87.3/87.4/86.7: training-free recovery SATURATES at ~65% by B≈4.5–4.75; residual damage needs the fine-tune escalation (precisely motivates the sequel)
- [x] Improvement: full-split paired bootstrap CIs — testA +1.84 [1.29,2.39], testB +1.51 [0.80,2.20] — core effect solidly significant
- [x] Week 4: Figures 1+2 rendered from real data (CVD-safe Okabe-Ito, visually inspected, collision-fixed) and wired into main.tex — includes the bonus finding that layer-15 attention is grounding-critical but VQA-negative
- [x] Week 4: **ODinW-13 held-out — GENERALIZATION CONFIRMED**: BF16 F1 0.654 / W4 0.582 (−11% rel.) / GCQ-4.25 0.626 (**61% of OOD loss recovered, zero ODinW in selection**); also refines the asymmetry story: OOD multi-object detection degrades 2× more than in-domain REC (11% vs 5%)
- [x] Week 4: **full testA/testB CONFIRMATIONS COMPLETE** — testA (n=5,657): 90.7/86.9/88.7/89.2 (BF16/W4/GCQ4.25/GCQ4.5, 49–60% recovery); testB (n=5,095): 84.7/79.9/81.4/81.9 (32–43% recovery); replicates subset numbers; harder split recovers less (consistent story)

## Rigor gates (must all hold before submission)
- [x] Novelty claims updated for MABA (CVPR Findings 2026): do not claim first VLM mixed precision, first gradient-guided allocation, or broad first capability-aware PTQ
- [x] Data hygiene specified: D_probe/D_dev/test disjoint; RefCOCOg umd split; cross-variant image exclusion
- [x] Profiling direction correct (promote-one-from-fully-quantized, not demote-one-from-BF16)
- [x] Floor-corrected retention R′ defined; GIoU never reported as a ratio
- [x] VQA constraint pre-registered (1.5 pts, paired test, fixed ~5k items)
- [x] Kill criteria pre-registered for H1/H2/proxy/method-ceiling
- [ ] Scooping re-check updated after finding MABA; retain only the narrower grounding-conditioned/coordinate-specific/object-scale claim and rerun before submission (novelty text now narrowed in main.tex 2026-08-17; the literature re-search itself still needs to run right before submission)
- [x] Anonymization audit clean: empty PDF author metadata, \author{Anonymous}, no identifying strings in PDF or tex (one false positive: "generic")

## Submission mechanics
- [ ] OpenReview account + ODI 2026 venue located (link in workshop CFP)
- [ ] LLM-usage disclosure section present (drafted in main.tex — verify against final NeurIPS policy wording)
- [ ] Submit by Aug 29, 2026, 23:59 AoE — target Aug 27 to leave slack
- [ ] Post-submission: archive exact submitted PDF + code state (tag `submission-odi2026`)

## Post-deadline (Weeks 3–4 → camera-ready / full paper)
- [x] Complete results version of the paper (all figures/tables real) — 2026-08-13: both tables fully populated with measured numbers (28 cells + allocation table), zero placeholders; W4A8 ODinW re-run after catching silent no-A8 bug; W3 ODinW rows measured (0.000 both quantizers); body exactly 5pp
- [ ] Code + frozen subsets + checkpoint released; model card includes grounding numbers
- [ ] Notification Sept 29 → if accepted: poster + possible 15-min oral prep
