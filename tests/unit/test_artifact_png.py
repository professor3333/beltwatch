import io

import numpy as np
import pytest
from PIL import Image

from beltwatch.jobs import pipeline


@pytest.mark.parametrize("shape", [(6, 9), (6, 9, 3)])
def test_artifact_png_is_lossless(shape: tuple[int, ...]) -> None:
    array = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)

    data = pipeline.encode_png(array)

    assert data.startswith(b"\x89PNG")
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(data))), array)
