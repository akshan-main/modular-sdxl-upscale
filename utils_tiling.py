"""Tile planning and cosine blending weights for MultiDiffusion."""

from dataclasses import dataclass

import torch


@dataclass
class LatentTileSpec:
    """Tile specification in latent space.

    Attributes:
        y: Top edge in latent pixels.
        x: Left edge in latent pixels.
        h: Height in latent pixels.
        w: Width in latent pixels.
    """

    y: int
    x: int
    h: int
    w: int


def validate_tile_params(tile_size: int, overlap: int) -> None:
    if tile_size <= 0:
        raise ValueError(f"`tile_size` must be positive, got {tile_size}.")
    if overlap < 0:
        raise ValueError(f"`overlap` must be non-negative, got {overlap}.")
    if overlap >= tile_size:
        raise ValueError(
            f"`overlap` must be less than `tile_size`. "
            f"Got overlap={overlap}, tile_size={tile_size}."
        )


def plan_latent_tiles(
    latent_h: int,
    latent_w: int,
    tile_size: int = 64,
    overlap: int = 8,
) -> list[LatentTileSpec]:
    """Plan overlapping tiles in latent space for MultiDiffusion.

    Tiles overlap by ``overlap`` latent pixels. Edge tiles are clamped to
    the latent bounds.
    """
    validate_tile_params(tile_size, overlap)

    stride = tile_size - overlap
    tiles: list[LatentTileSpec] = []

    y = 0
    while y < latent_h:
        h = min(tile_size, latent_h - y)
        if h < tile_size and y > 0:
            y = max(0, latent_h - tile_size)
            h = latent_h - y

        x = 0
        while x < latent_w:
            w = min(tile_size, latent_w - x)
            if w < tile_size and x > 0:
                x = max(0, latent_w - tile_size)
                w = latent_w - x

            tiles.append(LatentTileSpec(y=y, x=x, h=h, w=w))

            if x + w >= latent_w:
                break
            x += stride

        if y + h >= latent_h:
            break
        y += stride

    return tiles


def make_cosine_tile_weight(
    h: int,
    w: int,
    overlap: int,
    device: torch.device,
    dtype: torch.dtype,
    is_top: bool = False,
    is_bottom: bool = False,
    is_left: bool = False,
    is_right: bool = False,
) -> torch.Tensor:
    """Boundary-aware cosine blending weight for one tile.

    Returns shape (1, 1, h, w).  Canvas-edge sides get weight 1.0 (no fade),
    interior overlap regions get a half-cosine ramp from 0 to 1.
    """
    import math

    wy = torch.ones(h, device=device, dtype=dtype)
    wx = torch.ones(w, device=device, dtype=dtype)

    ramp = min(overlap, h // 2, w // 2)
    if ramp <= 0:
        return torch.ones(1, 1, h, w, device=device, dtype=dtype)

    cos_ramp = torch.tensor(
        [0.5 * (1 - math.cos(math.pi * i / ramp)) for i in range(ramp)],
        device=device,
        dtype=dtype,
    )

    if not is_top:
        wy[:ramp] = cos_ramp
    if not is_bottom:
        wy[-ramp:] = cos_ramp.flip(0)
    if not is_left:
        wx[:ramp] = cos_ramp
    if not is_right:
        wx[-ramp:] = cos_ramp.flip(0)

    weight = wy[:, None] * wx[None, :]
    return weight.unsqueeze(0).unsqueeze(0)
