"""ERNIE-Image-Turbo P1 sanity: first MLX render from the local snapshot."""
import time
from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage
from mflux.models.common.config.model_config import ModelConfig

t0 = time.time()
model = ErnieImage(
    model_path="/Volumes/DEV_VOL1/VideoResearch/ernie-image-models/ERNIE-Image-Turbo",
    model_config=ModelConfig.ernie_image_turbo(),
)
print(f"[load] {time.time()-t0:.1f}s", flush=True)
t0 = time.time()
image = model.generate_image(
    seed=42,
    prompt="A red fox standing in tall golden grass at sunset, photorealistic wildlife photography",
    num_inference_steps=8,
    height=1024,
    width=1024,
    guidance=1.0,
)
print(f"[generate] {time.time()-t0:.1f}s", flush=True)
image.save(path="outputs/turbo-fox-seed42.png", export_json_metadata=False)
print("[saved] outputs/turbo-fox-seed42.png", flush=True)
