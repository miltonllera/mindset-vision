"""unit tests for shape extraction and cropping module."""

from pathlib import Path
import numpy as np
from PIL import Image
from mindset.shapes.cropping import ShapeExtractor, SegmentationResult


def test_shape_extractor_text_prompt():
    """test extraction with text prompt."""
    sample_img = Path("mindset/assets/linedrawings/cropped/banana/banana.png")
    if not sample_img.exists():
        return

    extractor = ShapeExtractor()
    res = extractor.extract(sample_img, prompt="banana")

    assert isinstance(res, SegmentationResult)
    assert res.mask.ndim == 2
    assert res.cropped_image.size == (298, 216) or res.cropped_image.size[0] > 0
    assert len(res.contour) > 0
    assert res.polygon is not None


def test_shape_extractor_box_prompt():
    """test extraction with box prompt."""
    sample_img = Path("mindset/assets/linedrawings/cropped/banana/banana.png")
    if not sample_img.exists():
        return

    extractor = ShapeExtractor()
    res = extractor.extract(sample_img, prompt=[10, 10, 200, 150])

    assert isinstance(res, SegmentationResult)
    assert res.mask.ndim == 2
    assert len(res.contour) > 0


def test_shape_extractor_highlight_mask_prompt():
    """test extraction with highlighted region mask prompt."""
    sample_img = Path("mindset/assets/linedrawings/cropped/banana/banana.png")
    if not sample_img.exists():
        return

    highlight_mask = Image.new("L", (298, 216), 0)
    extractor = ShapeExtractor()
    res = extractor.extract(sample_img, prompt=highlight_mask)

    assert isinstance(res, SegmentationResult)
    assert res.mask.ndim == 2
