# GCQ — 4-Week Solo Research Plan (2nd-year undergrad, one 24 GB GPU)

Assumptions: ~20 focused hours/week plus unattended overnight GPU jobs; PyTorch + transformers experience; total GPU budget ≈ 60–75 hours. The ODI 2026 deadline (**Aug 29, AoE**) falls at the end of Week 2 — the plan is built so the submission goes in with preliminary results, and Weeks 3–4 complete the study for the camera-ready / full paper.

**Golden rule for a solo project: subset-first, always.** Every experiment runs on frozen 1k–5k subsets; full test splits are touched exactly once, in Week 4, for the 3–4 headline configs only.

---

## Week 0 (do now, ~2 days, overlaps Week 1)

- Request/confirm GPU allocation. Download: Qwen3-VL-2B-Instruct, RefCOCO/+/g annotations (RefCOCOg **umd** split), COCO train2014/val2014 images, VQAv2 val, POPE, a 2–3-dataset ODinW-13 slice.
- Make a git repo. Everything that defines an experiment (subset indices, seeds, prompts, configs) gets committed.

## Week 1 — Eval harness + the first gap datapoint

*This is the make-or-break week; the harness is the real work of the whole project.*

- **Days 1–2:** Load BF16 Qwen3-VL-2B, run 10 grounding prompts ("Locate the {expression}, output bbox_2d in JSON"), parse the JSON, map [0,1000]→pixels, compute IoU. Eyeball all 10 against the images — do not proceed until boxes visibly land on the right objects.
- **Days 3–4:** Build the harness: (a) REC eval — frozen 1k-sample RefCOCO-val eval subset (seed 0), disjoint from a frozen 1k-sample **D_dev** selection subset; acc@IoU 0.5 + mean GIoU; unparseable outputs count as wrong and the parse-failure rate is reported. (b) VQAv2 5k-item subset + POPE, exact-match scoring.
- **Day 5:** BF16 baseline on everything + the **image-blind floor runs** (blank image, same prompts) for POPE/VQAv2/REC — needed for floor-corrected retention.
- **Days 6–7:** First quantized model: W4 GPTQ, standard text calibration (use a community checkpoint if one exists; else GPTQModel). Same evals.
- **Done when:** one table — BF16 vs. W4 on {REC, GIoU, VQAv2, POPE}. This is your first grounding-gap datapoint.
- **Fallback (important):** if GPTQModel fights you on the VLM, use *simulated* quantization — in-place groupwise RTN quant-dequant of the LLM linear weights (~20 lines of PyTorch). Scientifically valid for the measurement study and all profiling; only the final released checkpoint needs real GPTQ.

## Week 2 — The measurement grid + SUBMIT

- **Days 8–10:** The grid: {RTN, GPTQ, AWQ} × {W8, W4, W3}, all on the frozen subsets (9 configs × ~1–2 h; run overnight). W4A8 is a stretch goal, not a requirement.
- **Day 11:** Compute floor-corrected retention R′; make **Figure 1** (grounding retention vs. VQA retention across bit-widths). Pre-registered decision point: if the W4 gap is <2 points, the headline moves to W3 — already in the grid, no new runs.
- **Days 12–14:** Re-run the scooping searches (incl. "RefCOCO quantized VLM"), port the proposal doc to the NeurIPS LaTeX template, insert the preliminary table + Figure 1, anonymize (check the PDF metadata too), **submit on OpenReview by Aug 29 AoE**.
- **Done when:** submission is in, with real preliminary numbers no other proposals-track submission is likely to have.

## Week 3 — Sensitivity map + the allocation

- **Days 15–17:** Profiling: from the fully-W4 model, promote one module at a time to 8-bit (56 configs) and measure the coordinate-token KL reduction vs. the BF16 teacher on the 512-sample D_probe (teacher-forced; precompute and cache the teacher's coordinate-position logits once). A few GPU-hours total.
- **Day 18:** **Proxy validation (kill criterion):** pick 8–10 modules spanning the s_m range, run decoded REC (500-sample subset) for each promotion, compute Spearman ρ against s_m. If ρ < ~0.5, switch to direct decoded-REC profiling of the top-15 modules only.
- **Day 19:** **Figure 2** (sensitivity by depth, attention vs. MLP); greedy allocation at B = 4.25; build the mixed 4/8-bit checkpoint (GPTQModel per-module overrides, or simulated).
- **Days 20–21:** Evaluate the GCQ model on all subsets; run the paired VQA/POPE constraint check against the uniform-W4 baseline.
- **Done when:** you know whether grounding-driven promotion beats uniform W4 at +6% memory.

## Week 4 — Controls, held-out, final numbers

- **Days 22–24:** Equal-memory controls: random promotion (3 draws — this doubles as your variance estimate), VQA-sensitivity-driven allocation (same greedy procedure, VQA answer-token KL, same probe size), uniform W4. Compute A1's Spearman ρ between grounding and VQA sensitivity — this is the scientific heart.
- **Days 25–26:** Held-out ODinW-13 zero-shot on the final configs (first and only time it is touched). Full RefCOCO/+/g testA/testB overnight runs for the 3–4 headline configs only.
- **Days 27–28:** Write results into the paper (camera-ready / full-paper draft), push code + the frozen subsets + the quantized checkpoint with grounding numbers on its model card.

---

## Rigor checklist (non-negotiable, all cheap)

1. Frozen, committed subsets and seeds; selection data (D_dev) never overlaps eval subsets; reported val numbers exclude D_dev.
2. ODinW untouched until Week 4 — it is the only proof the method didn't overfit RefCOCO phrasing.
3. All headline numbers from *decoded* outputs; the KL proxy is only ever a search signal, validated before trusted.
4. Floor-corrected retention + absolute deltas; report parse-failure rates.
5. Controls at *equal memory* (random + VQA-driven), or the allocation claim is unsupported.
6. VQA/POPE constraint checked as a paired comparison on identical items, tolerance (1.5 pts) fixed before running.
7. Lab notebook: every run gets one CSV row (config hash → all metrics). Future-you writes the paper from this file.

## If you fall behind (pre-planned cuts, in order)

1. Drop AWQ (keep RTN + GPTQ). 2. Drop W8 (keep BF16/W4/W3). 3. Drop B = 4.5 (keep 4.25). 4. Shrink proxy validation to 6 modules. Never cut: the controls (random/VQA-driven), ODinW, or the floor runs — those are the difference between "workshop-rigorous" and "not believable."
