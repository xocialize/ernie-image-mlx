# ernie-image-mlx

> **Proposed README** — this repository currently has **no `README.md`** and
> **no git remote** (local-only; not on GitHub). Drafted from `PORTING-SPEC.md`
> and the code. The companion Swift port `ernie-image-swift` is the publication
> target (planned `xocialize/ernie-image-swift`).

Apple MLX porting workbench for **ERNIE-Image-Turbo** (baidu, Apache-2.0) — a
lightweight `textToImage` model for MLXEngine's lower-tier clients. This repo is
the **Python sanity + golden-capture oracle**; the parity-locked production port
lives in the separate Swift package `ernie-image-swift`.

## What it is

- **Model:** baidu/ERNIE-Image(-Turbo), Apache-2.0. 8B **single-stream** DiT.
  ~22 GB bf16 / 12 GB 8-bit / **6.2 GB 4-bit**. 1024² + 6 aspect buckets.
  Turbo = 8 steps, guidance 1.0 (no CFG → one DiT forward/step); full ERNIE
  (~50 steps, CFG ~4.0) is a later weight-swap.
- **Components:** `ErnieTransformer` (36L, hidden 4096, 32 heads, ffn 12288,
  rope_theta 256, qk_layernorm, AdaLN 6-way, single-stream) ·
  `ErnieMistralTextEncoder` (26L/3072, GQA 32q/8kv, **features from the
  second-to-last hidden layer**) · **Flux2VAE** (identical class to Lens's →
  direct reuse from `lens-mlx-swift`) · scheduler with **fixed sigma shift
  1.3863 (= ln 4)**.
- **MLX reference:** `mflux` main, vendored here as `mflux-main/` (gitignored).

## Layout

```
scripts_sanity.py            # P1 first-render sanity (ErnieImage, 8 steps, 1024²)
scripts/capture_pt_goldens.py# PT golden capture (oracle protocol)
mflux-main/                  # mflux MLX reference (gitignored)
.venv/                       # MLX venv (gitignored)
.venv-pt/                    # PyTorch oracle venv (gitignored)
outputs/                     # render outputs (gitignored)
PORTING-SPEC.md              # authoritative plan + per-phase status (P0–P4)
```

> Note: `.venv/`, `.venv-pt/`, `mflux-main/`, and `outputs/` are gitignored; the
> tracked surface is the two scripts and the spec.

## Status (per PORTING-SPEC.md, 2026-06-12)

The Python oracle and the Swift port are reported **complete through P4**:

- **P1** Python sanity: first Turbo render clean (`outputs/turbo-fox-seed42.png`,
  1024²/8 steps, load 4.0 s · generate 33.4 s bf16 on M5 Max).
- **P2** Goldens captured (encoder second-to-last + last hidden, latents,
  scheduler sigmas, dit_step0, vae_decode); PT bf16 MPS reference. Finding: the
  Prompt Enhancer (Ministral-3B) **ships** in the open release as an optional
  pre-step (excluded from goldens/footprint).
- **P3** `ernie-image-swift`: all gates green — scheduler exact, DiT step-0
  0.9999996 fp32-CPU, encoder hidden[-2] 0.9999969 bf16 (two pitfall-#26 finds:
  YaRN rope; hidden_states[-2] = after 25 layers), Flux2VAE reused 64.0 dB, e2e
  **19.5 s @1024²/8 steps bf16**.
- **P4** engine package `ErnieImagePackage` (PackageID `ernie-image-turbo`, second
  `textToImage` backer); 4-bit variant shipped (~6.8 GB disk, resident 7.4 GB).

**Remaining:** tiled VAE decode (1024² peak is the Flux2VAE conv scratch),
full-ERNIE weights + CFG path, optional PE pre-step via an `llm` capability,
HF auto-download, and publication of `xocialize/ernie-image-swift`
(weights Apache-2.0 → mlx-community candidate for the 4-bit conversion).

## Usage (Python sanity)

```bash
# MLX venv with mflux-main installed
.venv/bin/python scripts_sanity.py        # writes outputs/turbo-fox-seed42.png
```

`scripts_sanity.py` constructs `mflux ErnieImage(model_path=..., model_config=
ModelConfig.ernie_image_turbo())` and renders at 8 steps / guidance 1.0. The
Turbo CLI has no `--model-path`; the `ErnieImage` class takes `model_path`
directly. Weights expected at the local snapshot path in the spec
(`VideoResearch/ernie-image-models/ERNIE-Image-Turbo`).

## License

Model weights Apache-2.0 (baidu). Port code follows the repo's chosen license
(add one before any publish). Built on `mflux` (MLX reference) and reuses the
`lens-mlx-swift` Flux2VAE on the Swift side.
