"""mindset shape base classes and utilities."""

from mindset.shapes.manipulation import BaseDrawManipulatedObject
from mindset.shapes.cropping import SegmentationResult, ShapeExtractor

__all__ = ["ShapeExtractor", "SegmentationResult", "BaseDrawManipulatedObject"]
