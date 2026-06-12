# ernie-image-mlx — Porting Spec (ERNIE-Image-Turbo → ernie-image-swift)

**Goal:** lightweight `textToImage` for lower-tier clients. Turbo first (8 steps,
guidance 1.0 = NO CFG → one DiT forward/step); full ERNIE later (same arch, ~50
steps, CFG ~4.0 — weights swap + the CFG path we already have).

## P0 facts (2026-06-12)

- **Model:** baidu/ERNIE-Image(-Turbo), Apache-2.0. 8B SINGLE-stream DiT.
  ~22 GB bf16 / 12 GB 8-bit / **6.2 GB 4-bit**. 1024² + 6 aspect buckets.
- **Components (mflux main = the MLX reference; in mflux-main/ here):**
  - ErnieTransformer: 36L, hidden 4096, 32 heads (head_dim 128), ffn 12288,
    in/out 128, patch 1, text_in 3072, **rope_theta 256**, rope_axes [32,48,48]
    (via transformer_overrides), qk_layernorm, AdaLN 6-way, single-stream.
  - ErnieMistralTextEncoder: 26L/3072, GQA 32q/8kv, vocab 131072, theta 1e6,
    features from the SECOND-TO-LAST hidden layer (mind the hidden-states
    convention trap — qwen-image-edit pitfall #26: capture TRUE tensors).
  - **VAE: Flux2VAE — identical class to Lens's** (in_ch 128 = 32-ch latents
    packed 2×2). lens-mlx-swift decoder parity-locked 120 dB → DIRECT REUSE.
    (img2img later needs the Flux2 VAE ENCODER — not yet ported.)
  - Scheduler: **FIXED sigma shift 1.3863 (= ln 4)** base==max (constant, NOT
    the dynamic-mu calculate_shift of Qwen). num_train_steps 1000,
    max_sequence_length 2048.
- **Prompt Enhancer:** optional official pre-step (use_pe=True), EXCLUDED by
  mflux — does not affect parity (gate goldens PE-off; Baidu benchmarks both).
  Engine-side optional pre-step via llm capability later; verify whether the PE
  model/template is in the open release.
- **Engine:** slots into EXISTING textToImage. Multi-package-per-capability
  SHIPPED (mlx-engine-swift 5014c0d): Lens + ERNIE-Turbo coexist; apps pick
  via PackageID ("lens-t2i" vs the new surface name) or setDefault.
- Weights: /Volumes/DEV_VOL1/VideoResearch/ernie-image-models/ERNIE-Image-Turbo

## Ladder

P1 Python sanity: mflux-main venv (.venv here) → first Turbo render (8 steps,
   1024², fixed seed) → eye check. Record load dtype + exact sigmas.
P2 Goldens: PT reference is the checkpoint's own pipeline (diffusers
   ErnieImagePipeline needs a NEWER diffusers than 0.37.1 — separate venv or
   pip-upgrade; else trust_remote_code). Capture per the oracle protocol:
   resolved config, encoder hidden (second-to-last!), packed latents, DiT
   step-0, VAE decode, full-loop reference image. Monkeypatch-capture true
   SDPA inputs for the encoder (pitfall #26).
P3 Swift (ernie-image-swift): scheduler+rope gates → DiT (single-stream block:
   new but in-family) → encoder (Qwen25VL-backbone pattern) → Flux2VAE reuse
   from lens-mlx-swift → e2e render.
P4 Engine: textToImage package (second backer; PackageID selection) → 4-bit
   quantization (the lower-tier headline) → APP-VALIDATION handoff.

## P1 — PASSED (2026-06-12)

First MLX Turbo render clean: `outputs/turbo-fox-seed42.png` (1024², 8 steps,
guidance 1.0, seed 42) — photorealistic, prompt-faithful, the documented vivid
high-contrast house style. **Load 4.0 s · generate 33.4 s** bf16 on M5 Max
(vs Lens 73 s, Qwen-Edit 800 s). Runner: `scripts_sanity.py` (the Turbo CLI has
no --model-path; the ErnieImage class takes model_path directly).
NEXT (P2): PT goldens — newer-diffusers venv for ErnieImagePipeline; oracle
protocol incl. true-SDPA-input capture on the encoder (second-to-last layer!).
