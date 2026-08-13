# GCQ: Measuring and Closing the Grounding Gap in Quantized Edge VLMs

*Working document for the ODI 2026 Proposals track (NeurIPS 2026 workshop; 5-page main paper, references and appendix excluded, double-blind). This italic line is document metadata — drop it when porting to the LaTeX template.*

**Abstract.** The field quantizes vision-language models (VLMs) for edge deployment and validates the result on VQA, OCR, and reasoning benchmarks — grounding and detection go unmeasured. We propose what is, to our knowledge, the first systematic measurement of localization degradation under low-bit VLM quantization; two minimal interventions to repair it; and an evaluation protocol that puts detection, not VQA, in the selection loop. The hard part is not recovering grounding — a grounding-only fine-tune would simply trade one collapsed capability for another — but recovering it while VQA, hallucination, and chat behavior stay within 1–2 points of the quantized baseline. Every intervention here is selected under that constraint.

## 1. Summary

Every major VLM post-training-quantization (PTQ) paper — MBQ (arXiv 2412.19509, CVPR 2025), Q-VLM (2410.08119, NeurIPS 2024), VLMQ (2508.03351), LUQ (2509.23729), SPEED-Q (2511.08914) — validates on VQA, OCR, chart, and reasoning benchmarks. With one incidental exception — GWQ (2411.00850) includes a single RefCOCO table to validate its general outlier-preserving method — none reports any localization metric, and none analyzes how localization retention compares to VQA retention or makes grounding a compression target. Meanwhile, the token-pruning literature has already shown that localization is the *first* capability to collapse under compression, with retention asymmetries as extreme as 95% VQA vs. 47% visual grounding at the same budget (Nüwa, 2602.02951; FEATHER, 2412.13180, ICCV 2025). Quantization is the dominant edge-compression technique, and its effect on detection is an unguarded blind spot. Since the point of putting a VLM on a device is usually *perception* — find the object, point at the thing, read the scene — practitioners may be shipping broken perception behind healthy VQA numbers on the model card.

This proposal has three parts, deliberately minimal:

1. **A measurement study (the centerpiece).** Quantize Qwen3-VL-2B-Instruct across bit-widths and measure the "grounding gap": grounding retention vs. VQA retention, side by side, for the first time to our knowledge. (Throughout, *grounding* means referring-expression localization — emitting a correct box for a phrase; the held-out ODinW evaluation probes its transfer toward open-vocabulary detection.)
2. **Two minimal interventions.** (a) *Grounding-calibrated PTQ*: swap the usual text-corpus calibration set for ~512 grounding samples — a one-line change. (b) *Grounding-consistency loss (GCL) recovery*: a short QLoRA fine-tune of the quantized model with a distillation loss up-weighted on coordinate tokens — a five-line change.
3. **An evaluation-protocol claim.** Every selection decision in the compression pipeline is made on grounding metrics with VQA as a constraint, inverting standard practice — and compressed-VLM releases should report grounding on the model card.

We call the combined recipe **GCQ** (grounding-consistent quantization). One model, two interventions, one protocol. No new architecture, no new coordinate head, no RL, no learned components. What makes the problem non-trivial is the double bind: the recovery must restore grounding *without* becoming specialization — a model that localizes well but answers, describes, and refuses hallucination worse is not a repaired edge VLM, it is a different broken one. Grounding is therefore the objective and general capability the constraint, everywhere in the pipeline. The measurement study is the minimum publishable unit; the interventions are the constructive answer; the protocol is the community-practice contribution.

## 2. Motivation: the measurement blind spot

**The asymmetry is documented — for pruning.** FEATHER (2412.13180) showed that visual-token pruning leaves most benchmarks intact while localization accuracy decays roughly linearly to zero, and that popular benchmarks fail to detect this because they require minimal grounding. Nüwa (2602.02951) quantified the asymmetry at 95% VQA retention vs. 47.2% grounding retention under 88.9% token reduction; LiteLVLM (2605.13178) and FocusUI (2601.03928) report the same fragility for pixel/UI grounding (figures from recent preprints; re-verified during the final-week scooping check, §7). The mechanism is intuitive: emitting a correct box requires a preserved global spatial reference frame and fine positional detail; answering "what color is the dog" does not. The fragility extends to training: Mini-InternVL (2410.16261) reported that even ordinary LoRA fine-tuning disproportionately hurts grounding — so recovery training after compression is not automatically safe either.

**For quantization, nobody has looked systematically.** The VLM PTQ literature cited above evaluates exclusively on VQA/OCR/chart/doc/reasoning. MBQ frames quantization sensitivity as a *modality* issue (vision vs. language tokens), not a *capability* issue. Direct evidence is two isolated datapoints pointing in opposite directions: GWQ's single incidental table shows near-lossless RefCOCO grounding at ~4.6 average bits *when the top-1% outlier weights stay FP16*, while an applied vehicle-damage paper (2608.02470) observed that a 4-bit GPTQ Qwen-VL 2B classifies damage correctly at 87.3% but "fails consistently" at localizing it. Nobody has mapped where between those two datapoints the truth lies. Separately, an empirical Qwen3 quantization study (2505.02214) shows Qwen3 LLMs degrade more than Llama-3 at ≤3 bits and that the smallest models suffer most — exactly the regime edge deployment lives in.

**Why this fits ODI.** This is not primarily a compression paper; it is a claim that the field's *evaluation practice* for on-device multimodal models is blind to the capability those deployments exist to provide. A measurement protocol that exposes the gap — and a recipe that closes it — is immediately useful to anyone shipping a quantized VLM.

## 3. Why Qwen3-VL-2B

Qwen3-VL (technical report 2511.21631) trains grounding as a headline capability (COCO, Objects365, OpenImages, RefCOCO/+/g) and emits coordinates as plain text in a normalized [0, 1000] system, typically `{"bbox_2d": [x1, y1, x2, y2], "label": "..."}`. Coordinates are ordinary tokens in the output string, which makes "up-weight the loss on coordinate tokens" trivial to implement. The 2B Instruct model is edge-realistic — ~1.5 GB of weights at W4 (4-bit weights; we write WxAy for x-bit weights with y-bit activations), fitting Jetson Orin Nano 8GB-class devices (footprint noted for context; on-device latency benchmarking is deferred, §9) — and cheap to iterate on. The ecosystem (GPTQModel/AutoAWQ, transformers, ms-swift with grounding-format converters) means every baseline is downloadable. Known low-bit fragility of the Qwen3 family (2505.02214) means there is real headroom for a method to matter. Following 2607.08029, the vision tower is kept at BF16/INT8 throughout; all aggressive quantization targets the LLM.

## 4. Method

### 4.1 Setup and data hygiene

Student = Qwen3-VL-2B-Instruct with quantized LLM weights; teacher = the same model at BF16. **D_probe**: ~512 RefCOCO-train samples in the chat template (image + "Locate the {expression}, output bbox_2d in JSON" + ground-truth answer), used only as the calibration set for Intervention 1 (§4.3). **D_dev**: 1,000 RefCOCO-val samples, used *only* for selection — never for training or calibration. Reported RefCOCO-val numbers exclude the 1,000 D_dev samples, and the ODinW-13 evaluation set is never touched by training, calibration, or selection, so D_probe, D_dev, and all reported test data are strictly disjoint. Because the three RefCOCO variants draw on overlapping COCO images, two further rules apply: RefCOCOg uses the umd split (the google split leaks images between train and val), and the recovery-training mix and D_probe are filtered at the *image level* against the val/test splits of all three variants (MDETR-style, removing ~10% of training images) so fine-tuning never sees a test image.

We write **C** for the set of coordinate-token positions in a teacher-forced target: the tokens covering the digits, commas, and brackets inside `bbox_2d` arrays, identified by a regex over the target string and mapped to token positions — a mapping that remains well-defined if the tokenizer merges multi-digit spans into single tokens.

### 4.2 The protocol: detection in the loop

Every choice a compression pipeline normally makes by perplexity or VQA — calibration set, group size, loss weights, checkpoint — is made here by D_dev grounding metrics: referring-expression comprehension (REC) accuracy@IoU 0.5 plus mean generalized IoU (GIoU) against ground-truth boxes. General capability is tracked as a *constraint* — VQA and hallucination scores must stay within a pre-registered tolerance (1.5 points) of the same-method text-calibrated quantized baseline, checked by a paired test on a fixed ~5k-item VQAv2 subset and POPE (sized so the confidence interval is well inside the tolerance) — rather than as the objective. This inversion is itself one of the paper's claims: the field's selection signal has been blind to detection, and any group compressing a perception-oriented VLM can adopt this protocol without adopting anything else in the paper.

### 4.3 Intervention 1: grounding-calibrated PTQ

Run standard GPTQ/AWQ unchanged, but replace the text-corpus calibration set with D_probe (~512 grounding samples). Hypothesis: calibration statistics computed on grounding-style activations (image tokens + coordinate emission) preserve the activation ranges that matter for localization better than text calibration does. This is a one-line change to existing pipelines, evaluated as a **three-arm ablation** so a win is attributable to grounding specifically rather than to the mere presence of image tokens: (a) text-only calibration (the status quo), (b) multimodal non-grounding calibration — the *same images* with captioning prompts at matched count and length — and (c) grounding calibration.

### 4.4 Intervention 2: GCL recovery

After quantization, run a short parameter-efficient recovery fine-tune with LoRA (r=16, α=32) whose forward pass reproduces the deployed computation exactly: the quantized checkpoint's weights are dequantized to *frozen* BF16 — for weight-only quantization this is numerically what deployment kernels compute before the matmul — and the adapters train on top, so they correct exactly the deployed base's error. (This avoids the common trap of "QLoRA" in the bitsandbytes-NF4 sense, which would re-quantize the original weights differently at load and silently discard the grounding calibration.) Teacher-forced on ground-truth answers, the loss is

L = L_CE(ground truth) + λ · Σ_t w_t · KL( p_teacher(·|x, y_<t) ‖ p_student(·|x, y_<t) )

where **w_t = γ for t ∈ C and w_t = 1 otherwise**. We fix λ = 1 and sweep γ ∈ {1, 3, 5, 10} at W4A16 only; γ = 1 is the unweighted-distillation control, so the sweep doubles as the key ablation.

Two design notes, both in service of simplicity. First, *no differentiable-IoU machinery*: token-level cross-entropy on coordinate digits correlates imperfectly with geometric error (2512.10554), but rather than backpropagating through decoded boxes, geometric quality (mean GIoU on D_dev) is used purely as the *selection* metric for checkpoints and hyperparameters — the loss stays plain weighted KL + CE, and geometry steers via selection. (If weighted KL proves insufficient, the one permitted escalation is an expected-coordinate regression term; explicitly a fallback, not the plan.) Second, *training data*: 30–50k samples — RefCOCO/+/g train (majority) plus 20–30% general instruction data (a LLaVA-mix subset); 1–2 epochs. The general-instruct mix is not padding: it is the guard against the recovery collapsing into grounding specialization, and any checkpoint that violates the §4.2 VQA/POPE constraint is rejected regardless of its grounding score — the γ sweep (ablation A3, §5) asks precisely whether γ > 1 buys grounding without paying for it in general capability. We ship quantized base + LoRA adapters (~10–40 MB) by default and also report the merge-then-requantize variant to check for regression.

### 4.5 What we deliberately do not do

No new architecture or coordinate head, no tokenizer changes, no RL, no multi-teacher distillation, no mixed-precision bit-allocation machinery, no novel activation-quantization scheme (W4A8 is used off-the-shelf as a stress condition only). One model, one loss, one protocol. If a component does not survive its ablation, it gets cut, not patched.

## 5. Experimental plan

**Quantization settings.** BF16 reference; W8A16; W4A16; W3A16 — each via RTN, GPTQ, and AWQ where applicable — and W4A8 as a single harder stress condition.

**Benchmarks.** Grounding (primary): RefCOCO/+/g REC accuracy@0.5 (val/testA/testB, with val excluding D_dev; parse JSON, map [0,1000] to pixels) and mean GIoU; a held-out ODinW-13 (Object Detection in the Wild) zero-shot subset as out-of-distribution grounding, never touched by training, calibration, or selection. General (constraint): VQAv2 subset + POPE. Headline quantity: retention ratios **R = metric_quantized / metric_BF16, computed on accuracy-type metrics only** — REC accuracy for R_grounding, VQAv2 accuracy for R_VQA. Because benchmarks have different chance floors (POPE's binary floor is 50%, VQA has a strong language-prior floor, REC's floor is ≈0), the headline figure uses floor-corrected retention **R′ = (S_quantized − S_floor)/(S_BF16 − S_floor)**, with each floor measured empirically by an image-blind pass (blank image, same prompts); raw R and absolute point deltas are reported alongside. GIoU is reported as absolute values and deltas only, since a ratio of a metric that can sit near zero or go negative is not meaningful.

**Figure 1 (planned):** the grounding-gap curves — R_grounding vs. R_VQA across bit-widths, per quantization method. **Table 1 (planned):** main results grid, method × bit-width × {REC acc@0.5, mean GIoU, VQAv2, POPE, ODinW}.

**Baselines and matched controls.** BF16; RTN/GPTQ/AWQ with standard text calibration (the field's status quo); grounding-calibrated PTQ alone (isolates Intervention 1); plain-CE LoRA recovery without distillation, run at *both* matched optimizer steps and matched compute — these differ because the GCL teacher pass costs ~1.3–1.5× per step (teacher logits on the fixed teacher-forced targets are precomputed once to shrink this); headline comparisons use matched compute, with matched steps as the mechanism-isolating secondary; unweighted-KL distillation (γ = 1) at matched steps; full GCL. The headline GCL-vs-plain-CE comparison is run with 3 seeds and reported as mean ± sd; differences below seed variance are not claimed.

**Ablations.** A1: text-calib vs. grounding-calib PTQ, nothing else changed. A2: GCL vs. plain-CE QLoRA vs. γ=1 distillation at matched steps. A3: γ ∈ {1, 3, 5, 10} at W4A16 — does γ > 1 help grounding without hurting VQA? A4: generalization — held-out ODinW-13 after RefCOCO-heavy training (guards against phrasing overfit).

## 6. Hypotheses and pre-registered kill criteria

- **H1 (measurement).** At W4 — and clearly at W3 and W4A8 — grounding retention trails VQA retention by a meaningful margin (target: >10 percentage points of floor-corrected retention). Support: the pruning asymmetry (2602.02951, 2412.13180), the 4-bit localization-failure anecdote (2608.02470), Qwen3's documented low-bit fragility (2505.02214). GWQ's near-lossless RefCOCO at ~4.6 bits with FP16 outliers retained (2411.00850) tempers expectations at moderate bits — if the gap is real, it should open at uniform W4 and below, which is exactly where our grid concentrates.
- **H2 (method).** At the most aggressive setting where H1's gap manifests (expected W4A16; otherwise W3A16 or W4A8), grounding-calibrated PTQ + GCL recovers ≥90% of the BF16 grounding lost by the standard text-calibrated baseline, within the pre-registered 1.5-point VQA tolerance of that baseline.

**Kill criteria (pre-registered honesty).** If grounding is essentially lossless at W4A16, the headline gap claim and both interventions move to the most aggressive settings already in the grid (W3A16, W4A8). If grounding is lossless *everywhere practical*, the paper pivots to a clean negative/contrast result — "weight quantization spares grounding; token pruning destroys it" — which is still a publishable measurement study and directly useful to practitioners. If GCL fails to beat plain-CE recovery at matched compute, we report the null. The measurement study (H1, weeks 1–4) is the minimum publishable unit on its own; a useful paper exists even if the interventions under-deliver.

**Known limitations, stated up front.** (i) The coordinate-weighted teacher-forced KL in the GCL objective may understate free-decoding damage (errors compound autoregressively); it therefore appears only as a training loss — all measurement, checkpoint/hyperparameter selection, and headline numbers use decoded REC accuracy and mean GIoU, never the training loss. (ii) The novelty claim — that no published work *systematically* measures grounding degradation under low-bit quantization of a generative VLM or uses coordinate-weighted distillation for quantization recovery — is *to our knowledge*, a negative claim from a finite search; the closest lines are GWQ's single incidental RefCOCO table for method validation (2411.00850 — one model family, one bit setting, no retention analysis, nothing grounding-aware), modality-balanced PTQ (2412.19509), grounding-aware *pruning* (2412.13180, 2602.02951), quantization-aware protection of visual *input* tokens for VLA policies (2509.09090), and localization distillation for CNN detector heads (2102.12252). We will re-verify immediately before submission and camera-ready. (iii) Several motivating figures come from recent preprints and are treated as provisional.

## 7. Feasibility and timeline

Everything runs on a single 24 GB GPU. The measurement phase is forward-only (decoding for evaluation) and takes days, not weeks; each GCL run on the 2B is 8–24 hours; the full project is well under 150 GPU-hours because sweeps run subset-first — 1–2k-sample REC subsets with cached image features, with only selected configurations evaluated on full splits. All data is public (RefCOCO/+/g, ODinW, LLaVA-mix, VQAv2, POPE); the ms-swift grounding converter handles [0,1000] normalization for Qwen3-VL. Team: 1–2 people; nothing requires distributed training or proprietary data.

**8-week plan.** Weeks 1–2: evaluation harness (REC parser, GIoU, VQA/POPE subsets); BF16 + RTN/GPTQ/AWQ baselines. Weeks 3–4: the grounding-gap study across all settings → Figure 1 (H1 resolved here; kill criteria applied here). Weeks 5–6: calibration ablation (A1) + GCL training (A2). Week 7: γ sweep (A3) + held-out ODinW evaluation (A4). Week 8: writing, figure polish, scooping re-check, code and checkpoint release.

## 8. Deliverables

(1) The grounding-gap curves and tables — to our knowledge the first *systematic* quantization-vs-grounding measurement for a generative VLM. (2) The two-intervention recipe with code, as thin wrappers over GPTQModel/AutoAWQ and ms-swift. (3) One released quantized Qwen3-VL-2B checkpoint *with grounding numbers on the model card* — itself a small statement about how compressed VLMs should be reported. (4) A short written recommendation: compressed-VLM releases intended for perception workloads should report REC/GIoU retention alongside VQA.

## 9. Deferred to future work

A full paper following this proposal would add: per-layer localization-sensitivity profiling and mixed-precision bit allocation; the 4B/8B models (does the gap shrink with scale?); a prompted-detection COCO AP pipeline including size-stratified AP; broader general benchmarks (GQA, MME, TextVQA); finer bit-budget sweeps and FP8/MBQ baselines; and on-device (Jetson-class) latency, memory, and throughput measurements. None of these are needed to establish the blind spot or to test the two interventions, so all are out of scope here.

## Compliance note

This submission is fully anonymized for double-blind review, including the appendix; code and checkpoint links will be released upon de-anonymization. Per the NeurIPS LLM policy: a large language model was used as a writing and editing aid in preparing this proposal; all technical content, claims, and experimental design are the authors' own, and the authors take full responsibility for the submission's content.

## Key references

- Qwen3-VL technical report — arXiv 2511.21631 (normalized [0,1000] coords; grounding data pipeline)
- FEATHER (token pruning kills localization) — arXiv 2412.13180 (ICCV 2025)
- Nüwa (95% VQA vs. 47% grounding retention) — arXiv 2602.02951
- LiteLVLM — arXiv 2605.13178; FocusUI — arXiv 2601.03928
- MBQ — arXiv 2412.19509 (CVPR 2025); Q-VLM — arXiv 2410.08119 (NeurIPS 2024); VLMQ — arXiv 2508.03351; LUQ — arXiv 2509.23729; SPEED-Q — arXiv 2511.08914
- GWQ (gradient-aware quantization; incidental RefCOCO table for quantized Qwen-VL) — arXiv 2411.00850
- Empirical Study of Qwen3 Quantization — arXiv 2505.02214
- Rethinking Small VLM Quantization — arXiv 2607.08029
- Localization Distillation — arXiv 2102.12252; SQAP-VLA — arXiv 2509.09090
- Grounding Everything in Tokens (coordinate-token CE vs. geometry) — arXiv 2512.10554
- 4-bit localization-failure anecdote — arXiv 2608.02470
- Mini-InternVL (LoRA hurts grounding) — arXiv 2410.16261

---

*Appendix plan (does not count toward the 5 pages; main paper remains self-contained): a novelty-deltas table against closest prior work; extended related work; data formatting and prompt-template details; full hyperparameter tables.*
