"""Block composition for Modular SDXL Upscale."""

from ...utils import logging
from ..modular_pipeline import SequentialPipelineBlocks
from ..modular_pipeline_utils import OutputParam
from ..stable_diffusion_xl.before_denoise import (
    StableDiffusionXLImg2ImgSetTimestepsStep,
    StableDiffusionXLInputStep,
)
from .denoise import UltimateSDUpscaleMultiDiffusionStep
from .input import UltimateSDUpscaleTextEncoderStep, UltimateSDUpscaleUpscaleStep


logger = logging.get_logger(__name__)


class MultiDiffusionUpscaleBlocks(SequentialPipelineBlocks):
    """Modular pipeline blocks for tiled SDXL upscaling with MultiDiffusion.

    Uses latent-space noise prediction blending across overlapping tiles for
    seamless tiled upscaling at any resolution.

    Block graph::

        [0] text_encoder     - SDXL TextEncoderStep (reused)
        [1] upscale          - Lanczos resize
        [2] input            - SDXL InputStep (reused)
        [3] set_timesteps    - SDXL Img2Img SetTimestepsStep (reused)
        [4] multidiffusion   - MultiDiffusion step

    The MultiDiffusion step handles VAE encode, tiled denoise with blending,
    and VAE decode internally, using VAE tiling for memory efficiency.
    """

    block_classes = [
        UltimateSDUpscaleTextEncoderStep,
        UltimateSDUpscaleUpscaleStep,
        StableDiffusionXLInputStep,
        StableDiffusionXLImg2ImgSetTimestepsStep,
        UltimateSDUpscaleMultiDiffusionStep,
    ]
    block_names = [
        "text_encoder",
        "upscale",
        "input",
        "set_timesteps",
        "multidiffusion",
    ]

    _workflow_map = {
        "upscale": {"image": True, "prompt": True},
        "upscale_controlnet": {"image": True, "control_image": True, "prompt": True},
    }

    @property
    def description(self):
        return (
            "MultiDiffusion upscale pipeline for Stable Diffusion XL.\n"
            "Upscales an input image and refines it using tiled denoising with "
            "latent-space noise prediction blending. Produces seamless output at "
            "any resolution without tile-boundary artifacts.\n"
            "Supports optional ControlNet Tile conditioning for improved fidelity."
        )

    @property
    def outputs(self):
        return [OutputParam.template("images")]
