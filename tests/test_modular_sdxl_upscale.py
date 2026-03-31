"""Tests for Modular SDXL Upscale (MultiDiffusion only)."""

import unittest

import torch

import utils_tiling as utils_tiling_mod
from utils_tiling import (
    LatentTileSpec,
    make_cosine_tile_weight,
    plan_latent_tiles,
    validate_tile_params,
)


class TestTileValidation(unittest.TestCase):
    def test_zero_tile_size_raises(self):
        with self.assertRaises(ValueError):
            validate_tile_params(0, 8)

    def test_negative_overlap_raises(self):
        with self.assertRaises(ValueError):
            validate_tile_params(64, -1)

    def test_overlap_equals_tile_raises(self):
        with self.assertRaises(ValueError):
            validate_tile_params(64, 64)

    def test_valid_params(self):
        validate_tile_params(64, 16)


class TestLatentTilePlanning(unittest.TestCase):
    def test_single_tile_small_image(self):
        tiles = plan_latent_tiles(32, 32, tile_size=64, overlap=8)
        self.assertEqual(len(tiles), 1)

    def test_multiple_tiles(self):
        tiles = plan_latent_tiles(128, 128, tile_size=64, overlap=16)
        self.assertGreater(len(tiles), 1)

    def test_tiles_cover_full_area(self):
        h, w = 96, 128
        tiles = plan_latent_tiles(h, w, tile_size=64, overlap=8)
        covered = set()
        for tile in tiles:
            for y in range(tile.y, tile.y + tile.h):
                for x in range(tile.x, tile.x + tile.w):
                    covered.add((y, x))
        for y in range(h):
            for x in range(w):
                self.assertIn((y, x), covered)

    def test_no_tile_exceeds_bounds(self):
        h, w = 100, 120
        tiles = plan_latent_tiles(h, w, tile_size=64, overlap=16)
        for tile in tiles:
            self.assertLessEqual(tile.y + tile.h, h)
            self.assertLessEqual(tile.x + tile.w, w)

    def test_exact_multiple(self):
        tiles = plan_latent_tiles(128, 128, tile_size=64, overlap=0)
        self.assertEqual(len(tiles), 4)


class TestCosineWeight(unittest.TestCase):
    def test_shape(self):
        w = make_cosine_tile_weight(64, 64, 8, torch.device("cpu"), torch.float32)
        self.assertEqual(w.shape, (1, 1, 64, 64))

    def test_boundary_top_left(self):
        w = make_cosine_tile_weight(64, 64, 16, torch.device("cpu"), torch.float32,
                                     is_top=True, is_left=True)
        self.assertEqual(w[0, 0, 0, 0].item(), 1.0)

    def test_boundary_bottom_right(self):
        w = make_cosine_tile_weight(64, 64, 16, torch.device("cpu"), torch.float32,
                                     is_bottom=True, is_right=True)
        self.assertEqual(w[0, 0, -1, -1].item(), 1.0)

    def test_interior_has_ramp(self):
        w = make_cosine_tile_weight(64, 64, 16, torch.device("cpu"), torch.float32)
        self.assertLess(w[0, 0, 0, 0].item(), 1.0)

    def test_center_is_one(self):
        w = make_cosine_tile_weight(64, 64, 8, torch.device("cpu"), torch.float32)
        self.assertAlmostEqual(w[0, 0, 32, 32].item(), 1.0, places=3)

    def test_zero_overlap_all_ones(self):
        w = make_cosine_tile_weight(64, 64, 0, torch.device("cpu"), torch.float32)
        self.assertTrue(torch.allclose(w, torch.ones_like(w)))


class TestOldCodeRemoved(unittest.TestCase):
    def test_no_tile_spec(self):
        self.assertFalse(hasattr(utils_tiling_mod, "TileSpec"))

    def test_no_seam_fix(self):
        self.assertFalse(hasattr(utils_tiling_mod, "SeamFixSpec"))

    def test_no_plan_tiles_linear(self):
        self.assertFalse(hasattr(utils_tiling_mod, "plan_tiles_linear"))

    def test_no_plan_tiles_chess(self):
        self.assertFalse(hasattr(utils_tiling_mod, "plan_tiles_chess"))


if __name__ == "__main__":
    unittest.main()
