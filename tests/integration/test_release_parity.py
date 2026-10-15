from pathlib import Path

import numpy as np
import torch
from PIL import Image

from beltwatch.inference.neural import save_inference_checkpoint
from beltwatch.inference.preprocessing import PreprocessConfig, load_rgb
from beltwatch.models.registry import ModelConfig, build_model
from beltwatch.release import parity
from beltwatch.release.bundle import build_release
from tests.integration.conftest import POLICY


def test_parity_catches_a_bundle_with_another_temperature(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = ModelConfig(pretrained=False)
    model_dir = tmp_path / "run"
    model_dir.mkdir()
    save_inference_checkpoint(
        model_dir / "best.pt", build_model(config), config, PreprocessConfig(long_side=96), {}
    )
    build_release(
        tmp_path / "r",
        version="v1",
        kind="unet",
        model_path=model_dir / "best.pt",
        review_policy=POLICY,
        temperature=1.5,
    )
    frames = tmp_path / "frames"
    frames.mkdir()
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (54, 96, 3), dtype=np.uint8)).save(frames / "a.png")
    assert load_rgb(frames / "a.png").shape == (54, 96, 3)

    common = ["--release", str(tmp_path / "r" / "v1"), "--model", "unet"]
    common += ["--model-dir", str(model_dir), "--images", str(frames)]
    assert parity.main([*common, "--temperature", "1.5"]) == 0
    assert parity.main([*common, "--temperature", "1.0"]) == 1
