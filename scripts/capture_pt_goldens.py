"""PT golden capture for ERNIE-Image-Turbo — oracle-capture protocol.

stage A (fp32, CPU): encoder ids + second-to-last hidden, seeded packed latents,
  resolved sigmas/timesteps, DiT step-0 (single branch — Turbo has no CFG),
  VAE bn stats + decode golden.
stage B (bf16, MPS): full 8-step loop, use_pe=False -> reference image.

PE excluded from loading (saves 7 GB; goldens are PE-off — deterministic).

Run:
  .venv-pt/bin/python scripts/capture_pt_goldens.py --stage A
  .venv-pt/bin/python scripts/capture_pt_goldens.py --stage B
"""

import argparse
import json
import pathlib

import torch

MODEL = "/Volumes/DEV_VOL1/VideoResearch/ernie-image-models/ERNIE-Image-Turbo"
OUT = pathlib.Path("/Volumes/DEV_VOL1/VideoResearch/ernie-image-models/goldens")
PROMPT = "A red fox standing in tall golden grass at sunset, photorealistic wildlife photography"
SEED = 42
STEPS = 8
GUIDANCE = 1.0
H = W = 1024

parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=["A", "B"], required=True)
args = parser.parse_args()
OUT.mkdir(parents=True, exist_ok=True)


def save_st(name, tensors):
    from safetensors.torch import save_file

    save_file(
        {k: v.contiguous().to(torch.float32).cpu() for k, v in tensors.items()},
        str(OUT / name),
    )
    print(f"[saved] {OUT / name}", flush=True)


from diffusers import ErnieImagePipeline  # noqa: E402

if args.stage == "B":
    pipe = ErnieImagePipeline.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, pe=None, pe_tokenizer=None)
    pipe.to("mps")
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    result = pipe(
        prompt=PROMPT,
        height=H, width=W,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=gen,
        use_pe=False,
    )
    result.images[0].save(OUT / "ref_turbo_bf16_mps.png")
    (OUT / "ref_config.json").write_text(json.dumps({
        "prompt": PROMPT, "seed": SEED, "steps": STEPS, "guidance": GUIDANCE,
        "height": H, "width": W, "use_pe": False, "dtype": "bfloat16", "device": "mps",
        "scheduler_config": dict(pipe.scheduler.config),
        "transformer_config": dict(pipe.transformer.config),
    }, indent=2, default=str))
    print(f"[saved] {OUT / 'ref_turbo_bf16_mps.png'} + ref_config.json", flush=True)
    raise SystemExit(0)

# ---- stage A: fp32 CPU dumps --------------------------------------------------
print("[load] pipeline fp32 cpu (no PE)", flush=True)
pipe = ErnieImagePipeline.from_pretrained(
    MODEL, torch_dtype=torch.float32, pe=None, pe_tokenizer=None)

# 1. Encoder: ids + second-to-last hidden state
ids = pipe.tokenizer(PROMPT, add_special_tokens=True, truncation=True, padding=False)["input_ids"]
input_ids = torch.tensor([ids])
with torch.no_grad():
    enc_out = pipe.text_encoder(input_ids=input_ids, output_hidden_states=True)
hidden = enc_out.hidden_states[-2][0]  # [T, 3072]
print(f"[encoder] T={len(ids)} hidden {tuple(hidden.shape)}", flush=True)
save_st("encoder.safetensors", {
    "input_ids": input_ids.to(torch.float32),
    "hidden_secondlast": hidden,
    "hidden_last": enc_out.hidden_states[-1][0],  # for convention cross-checks
})

# 2. Seeded packed latents (B, 128, H/16, W/16)
gen = torch.Generator(device="cpu").manual_seed(SEED)
lat_h, lat_w = H // 16, W // 16
latents = torch.randn((1, 128, lat_h, lat_w), generator=gen, dtype=torch.float32)
save_st("latents.safetensors", {"noise": latents})

# 3. Scheduler: resolved sigmas/timesteps for 8 steps
sigmas_in = torch.linspace(1.0, 0.0, STEPS + 1)[:-1]
pipe.scheduler.set_timesteps(sigmas=sigmas_in, device="cpu")
print("[scheduler] sigmas:", pipe.scheduler.sigmas.tolist(), flush=True)
print("[scheduler] timesteps:", pipe.scheduler.timesteps.tolist(), flush=True)
save_st("scheduler.safetensors", {
    "sigmas": pipe.scheduler.sigmas.to(torch.float32),
    "timesteps": pipe.scheduler.timesteps.to(torch.float32),
})

# 4. DiT step-0 forward (single branch)
text_bth, text_lens = pipe._pad_text(
    text_hiddens=[hidden], device="cpu", dtype=torch.float32,
    text_in_dim=pipe.transformer.config.text_in_dim)
t0 = pipe.scheduler.timesteps[0]
t_batch = torch.full((1,), t0.item(), dtype=torch.float32)
with torch.no_grad():
    pred = pipe.transformer(
        hidden_states=latents, timestep=t_batch,
        text_bth=text_bth, text_lens=text_lens, return_dict=False)[0]
print(f"[dit] pred {tuple(pred.shape)}", flush=True)
save_st("dit_step0.safetensors", {
    "pred": pred,
    "text_bth": text_bth,
    "text_lens": torch.tensor(text_lens, dtype=torch.float32)
        if not torch.is_tensor(text_lens) else text_lens.to(torch.float32),
    "timestep": t_batch,
})

# 5. VAE: bn stats + decode golden (seeded latents as stand-in final latents)
bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1)
bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + 1e-5)
denorm = latents * bn_std + bn_mean
unpatch = pipe._unpatchify_latents(denorm)
with torch.no_grad():
    decoded = pipe.vae.decode(unpatch, return_dict=False)[0]
print(f"[vae] decoded {tuple(decoded.shape)}", flush=True)
save_st("vae_decode.safetensors", {
    "latent_packed_denorm": denorm,
    "latent_unpatched": unpatch,
    "decoded": decoded,
    "bn_mean": pipe.vae.bn.running_mean,
    "bn_var": pipe.vae.bn.running_var,
})

(OUT / "goldens_meta.json").write_text(json.dumps({
    "prompt": PROMPT, "seed": SEED, "steps": STEPS, "guidance": GUIDANCE,
    "height": H, "width": W, "dtype": "float32", "device": "cpu",
    "encoder_tokens": len(ids),
    "scheduler_config": dict(pipe.scheduler.config),
    "transformer_config": dict(pipe.transformer.config),
    "text_encoder_config": {"hidden_layer": -2, "add_special_tokens": True},
}, indent=2, default=str))
print("[done] stage A", flush=True)
