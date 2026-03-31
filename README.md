# Modular SDXL Upscale

Tiled image upscaling for Stable Diffusion XL using [MultiDiffusion](https://arxiv.org/abs/2302.08113) latent-space blending. Built with [Modular Diffusers](https://huggingface.co/blog/modular-diffusers).

First community-contributed custom Hub block for the Modular Diffusers framework - a tiled upscaling pipeline that composes reusable SDXL blocks with MultiDiffusion and ControlNet Tile into a workflow that wasn't possible with the standard `DiffusionPipeline` API.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/akshan-main/5f8b5f23b9c231524a997c5b0a7b741f/modular_sdxl_upscale_demo.ipynb)
[![HuggingFace Hub](https://img.shields.io/badge/HuggingFace-Hub-yellow)](https://huggingface.co/akshan-main/modular-sdxl-upscale)

## What it does

- Image upscaling at any scale factor using SDXL
- MultiDiffusion: blends overlapping UNet tile predictions in latent space with cosine weights. No visible seams
- Optional ControlNet Tile conditioning for structure-preserving upscaling
- Progressive upscaling: automatically splits 4x+ into multiple 2x passes
- Auto-strength scaling per pass
- Scheduler selection (Euler, DPM++ 2M, DPM++ 2M Karras)

## Install

```bash
pip install git+https://github.com/huggingface/diffusers.git transformers accelerate safetensors
```

Requires diffusers from main (modular diffusers support).

## Usage

### From HuggingFace Hub (recommended)

```python
from diffusers import ModularPipelineBlocks, ControlNetModel
import torch

blocks = ModularPipelineBlocks.from_pretrained(
    "akshan-main/modular-sdxl-upscale",
    trust_remote_code=True,
)

pipe = blocks.init_pipeline("stabilityai/stable-diffusion-xl-base-1.0")
pipe.load_components(torch_dtype=torch.float16)

controlnet = ControlNetModel.from_pretrained(
    "xinsir/controlnet-tile-sdxl-1.0", torch_dtype=torch.float16
)
pipe.update_components(controlnet=controlnet)
pipe.to("cuda")

image = ...  # your PIL image

result = pipe(
    prompt="high quality, detailed, sharp",
    image=image,
    control_image=image,
    controlnet_conditioning_scale=1.0,
    upscale_factor=2.0,
    num_inference_steps=20,
    generator=torch.Generator("cuda").manual_seed(42),
    output="images",
)
result[0].save("upscaled.png")
```

### 4x progressive upscale

```python
result = pipe(
    prompt="high quality, detailed, sharp",
    image=image,
    control_image=image,
    controlnet_conditioning_scale=1.0,
    upscale_factor=4.0,
    progressive=True,
    generator=torch.Generator("cuda").manual_seed(42),
    output="images",
)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | required | Input image (PIL) |
| `prompt` | `""` | Text prompt |
| `upscale_factor` | `2.0` | Scale multiplier |
| `strength` | `0.3` | Denoise strength. Ignored when `auto_strength=True` |
| `num_inference_steps` | `20` | Denoising steps |
| `guidance_scale` | `7.5` | CFG scale |
| `latent_tile_size` | `64` | Tile size in latent pixels (64 = 512px) |
| `latent_overlap` | `16` | Tile overlap in latent pixels (16 = 128px) |
| `control_image` | `None` | ControlNet conditioning image |
| `controlnet_conditioning_scale` | `1.0` | ControlNet strength |
| `negative_prompt` | auto | Defaults to "blurry, low quality, artifacts, noise, jpeg compression" |
| `progressive` | `True` | Split upscale_factor > 2 into multiple 2x passes |
| `auto_strength` | `True` | Auto-scale strength per pass |
| `scheduler_name` | `None` | "Euler", "DPM++ 2M", "DPM++ 2M Karras" |
| `generator` | `None` | Torch generator for reproducibility |

## Limitations

- SDXL is trained on 1024x1024. `latent_tile_size` below 64 may produce artifacts
- 4x from inputs below 256px produces distortion. Use progressive mode
- ControlNet Tile is required for faithful upscaling, it is a very low-weight dependency, though, so not a big deal
- Not suitable for text, line art, or pixel art

## Architecture

```
MultiDiffusionUpscaleBlocks (SequentialPipelineBlocks)
  text_encoder      SDXL TextEncoderStep (reused)
  upscale           Lanczos upscale step
  input             SDXL InputStep (reused)
  set_timesteps     SDXL Img2Img SetTimestepsStep (reused)
  multidiffusion    MultiDiffusion step
                    - VAE encode full image
                    - Per timestep: UNet on each latent tile, cosine-weighted blend
                    - VAE decode full latents
```

8 SDXL blocks reused, 3 custom blocks added.

## Project structure

```
utils_tiling.py              Latent tile planning, cosine weights
input.py                     Text encoder, upscale steps
denoise.py                   MultiDiffusion step, ControlNet integration
modular_blocks.py            Block compositions
modular_pipeline.py          Pipeline class
hub_block/                   HuggingFace Hub block (consolidated single file)
```

## References

- [MultiDiffusion](https://arxiv.org/abs/2302.08113) (Bar-Tal et al., 2023)
- [Ultimate Upscale for A1111](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111)
- [Tiled Diffusion for A1111](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111)
- [Modular Diffusers](https://huggingface.co/blog/modular-diffusers)
- [Modular Diffusers contribution call](https://github.com/huggingface/diffusers/issues/13295)
- [ControlNet Tile](https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0)

