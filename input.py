# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input steps for Modular SDXL Upscale: text encoding, Lanczos upscale."""

import PIL.Image
import torch

from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import InputParam, OutputParam
from ..stable_diffusion_xl.encoders import StableDiffusionXLTextEncoderStep


logger = logging.get_logger(__name__)


class UltimateSDUpscaleTextEncoderStep(StableDiffusionXLTextEncoderStep):
    """SDXL text encoder step that applies guidance scale before encoding.

    Syncs the guider's guidance_scale before prompt encoding so that
    unconditional embeddings are always produced when CFG is active.

    Also applies a default negative prompt for upscaling when the user
    does not provide one.
    """

    DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, artifacts, noise, jpeg compression"

    @property
    def inputs(self) -> list[InputParam]:
        return super().inputs + [
            InputParam(
                "guidance_scale",
                type_hint=float,
                default=7.5,
                description="Classifier-Free Guidance scale.",
            ),
            InputParam(
                "use_default_negative",
                type_hint=bool,
                default=True,
                description="Apply default negative prompt when none is provided.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        guidance_scale = getattr(block_state, "guidance_scale", 7.5)

        if hasattr(components, "guider") and components.guider is not None:
            components.guider.guidance_scale = guidance_scale

        use_default_negative = getattr(block_state, "use_default_negative", True)
        if use_default_negative:
            neg = getattr(block_state, "negative_prompt", None)
            if neg is None or neg == "":
                block_state.negative_prompt = self.DEFAULT_NEGATIVE_PROMPT
                state.set("negative_prompt", self.DEFAULT_NEGATIVE_PROMPT)

        return super().__call__(components, state)


class UltimateSDUpscaleUpscaleStep(ModularPipelineBlocks):
    """Upscales the input image using Lanczos interpolation."""

    @property
    def description(self) -> str:
        return "Upscale input image using Lanczos interpolation."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("image", type_hint=PIL.Image.Image, required=True,
                       description="Input image to upscale."),
            InputParam("upscale_factor", type_hint=float, default=2.0,
                       description="Scale multiplier."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("upscaled_image", type_hint=PIL.Image.Image),
            OutputParam("upscaled_width", type_hint=int),
            OutputParam("upscaled_height", type_hint=int),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        image = block_state.image
        upscale_factor = block_state.upscale_factor

        if not isinstance(image, PIL.Image.Image):
            raise ValueError(f"Expected PIL.Image, got {type(image)}.")

        new_width = int(image.width * upscale_factor)
        new_height = int(image.height * upscale_factor)

        block_state.upscaled_image = image.resize((new_width, new_height), PIL.Image.LANCZOS)
        block_state.upscaled_width = new_width
        block_state.upscaled_height = new_height

        logger.info(f"Upscaled {image.width}x{image.height} -> {new_width}x{new_height}")

        self.set_block_state(state, block_state)
        return components, state
