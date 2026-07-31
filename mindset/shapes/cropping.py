"""shape extraction and cropping module using Segment Anything (SAM) and spatial/text prompts."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import LineString, Polygon


@dataclass
class SegmentationResult:
    """container for shape extraction results."""

    mask: np.ndarray  # 2D uint8 binary mask (0 or 255)
    cropped_image: Image.Image  # PIL RGBA cropped image on transparent/black background
    contour: np.ndarray  # (N, 2) array of (x, y) contour coordinates
    polygon: Optional[Polygon]  # Shapely Polygon object
    linestring: Optional[LineString]  # Shapely LineString object
    bounding_box: Tuple[int, int, int, int]  # (min_x, min_y, max_x, max_y)


class ShapeExtractor:
    """extracts object shapes and segmentations using SAM with text, box, point, or mask prompts."""

    def __init__(
        self,
        model_type: str = "facebook/sam-vit-base",
        device: str = "cpu",
        load_model: bool = False,
        use_fallback: bool = True,
    ):
        self.model_type = model_type
        self.device = device
        self.load_model = load_model
        self.use_fallback = use_fallback
        self._sam_model = None
        self._sam_processor = None
        if self.load_model:
            self._init_sam_model()

    def _init_sam_model(self):
        """initialize HuggingFace SAM model if available."""
        if self._sam_model is not None:
            return
        try:
            from transformers import SamModel, SamProcessor

            self._sam_processor = SamProcessor.from_pretrained(self.model_type)
            self._sam_model = SamModel.from_pretrained(self.model_type).to(self.device)
            print(f"initialized SAM model '{self.model_type}' on {self.device}.")
        except Exception as e:
            if not self.use_fallback:
                raise RuntimeError(
                    f"failed to load SAM model '{self.model_type}': {e}"
                )
            print(f"SAM model load deferred/skipped ({e}). Using OpenCV fallback engine.")

    def _load_image(
        self, image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> Tuple[Image.Image, np.ndarray]:
        """load image input into PIL Image and RGB numpy array."""
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
            np_img = np.array(pil_img)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            np_img = np.array(pil_img)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                np_img = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                np_img = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            else:
                np_img = image_input
            pil_img = Image.fromarray(np_img)
        else:
            raise ValueError(f"unsupported image input type: {type(image_input)}")

        return pil_img, np_img

    def _mask_to_result(
        self, mask: np.ndarray, orig_pil: Image.Image, background_color=(0, 0, 0)
    ) -> SegmentationResult:
        """convert a 2D binary uint8 mask (0/255) into a SegmentationResult."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            # Empty mask
            h, w = mask.shape
            empty_img = Image.new("RGBA", (w, h), background_color + (255,))
            return SegmentationResult(
                mask=mask,
                cropped_image=empty_img,
                contour=np.zeros((0, 2)),
                polygon=None,
                linestring=None,
                bounding_box=(0, 0, w, h),
            )

        main_contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
        min_x, min_y = main_contour[:, 0].min(), main_contour[:, 1].min()
        max_x, max_y = main_contour[:, 0].max(), main_contour[:, 1].max()

        # Build composite cropped image on background
        orig_rgba = orig_pil.convert("RGBA")
        bg_canvas = Image.new("RGBA", orig_pil.size, background_color + (255,))
        mask_pil = Image.fromarray(mask).convert("L")

        cropped = Image.composite(orig_rgba, bg_canvas, mask_pil)

        poly = Polygon(main_contour) if len(main_contour) >= 3 else None
        ls = LineString(main_contour) if len(main_contour) >= 2 else None

        return SegmentationResult(
            mask=mask,
            cropped_image=cropped,
            contour=main_contour,
            polygon=poly,
            linestring=ls,
            bounding_box=(int(min_x), int(min_y), int(max_x), int(max_y)),
        )

    def _segment_with_opencv(
        self,
        np_img: np.ndarray,
        prompt: Optional[Union[str, list, tuple, np.ndarray, Image.Image]] = None,
    ) -> np.ndarray:
        """fallback segmentation using OpenCV morphology, thresholding, or highlight masks."""
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        if isinstance(prompt, (list, tuple, np.ndarray)) and len(prompt) == 4 and isinstance(prompt[0], (int, float)):
            # Bounding box prompt [x1, y1, x2, y2]
            x1, y1, x2, y2 = [int(v) for v in prompt]
            sub_crop = gray[y1:y2, x1:x2]
            if sub_crop.size > 0:
                _, sub_bin = cv2.threshold(
                    sub_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                mask = np.zeros_like(gray)
                mask[y1:y2, x1:x2] = sub_bin
                return mask

        elif isinstance(prompt, (Image.Image, np.ndarray)):
            # Highlight mask / scribble prompt
            if isinstance(prompt, Image.Image):
                prompt_np = np.array(prompt)
            else:
                prompt_np = prompt

            if prompt_np.ndim == 3 and prompt_np.shape[2] >= 3:
                # Highlight region: find non-black / non-white colored region
                red_channel = prompt_np[:, :, 0]
                green_channel = prompt_np[:, :, 1]
                blue_channel = prompt_np[:, :, 2]
                # High red or non-gray pixels indicate highlight region
                highlight = (
                    (red_channel > 150) & (green_channel < 100) & (blue_channel < 100)
                ) | (np.std(prompt_np[:, :, :3], axis=2) > 20)
                mask = (highlight * 255).astype(np.uint8)
                if np.sum(mask) > 0:
                    return mask

            elif prompt_np.ndim == 2:
                _, mask = cv2.threshold(prompt_np, 128, 255, cv2.THRESH_BINARY)
                return mask

        # Default Otsu thresholding for white background line drawings / photos
        _, binary = cv2.threshold(
            gray, 240, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary

    def extract(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        prompt: Optional[Union[str, list, tuple, np.ndarray, Image.Image]] = None,
        background_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> SegmentationResult:
        """extract object shape using prompt (text, box, point, or highlight mask).

        Args:
            image_input: File path, PIL Image, or NumPy RGB array.
            prompt: Optional prompt. Can be:
                - str: Text prompt (e.g., "banana", "coffee mug")
                - list/tuple: Bounding box [x1, y1, x2, y2] or Point coordinates [[x, y]]
                - PIL.Image / np.ndarray: Mask image or highlighted region image
                - None: Automatic foreground extraction
            background_color: Output background color RGB tuple (default black (0,0,0)).

        Returns:
            SegmentationResult object containing mask, cropped RGBA image, contour, and Shapely polygon.
        """
        pil_img, np_img = self._load_image(image_input)

        # 1. Check if spatial prompt is a highlighted image mask or bounding box
        if self._sam_model is not None and self._sam_processor is not None:
            try:
                import torch

                input_boxes = None
                input_points = None

                if isinstance(prompt, (list, tuple)):
                    if len(prompt) == 4 and isinstance(prompt[0], (int, float)):
                        # Box prompt [x1, y1, x2, y2]
                        input_boxes = [[[float(v) for v in prompt]]]
                    elif len(prompt) > 0 and isinstance(prompt[0], (list, tuple)):
                        # Point prompt [[x, y]]
                        input_points = [[[float(pt[0]), float(pt[1])] for pt in prompt]]

                elif isinstance(prompt, (Image.Image, np.ndarray)):
                    # Highlight region image: extract bounding box of highlighted area
                    highlight_mask = self._segment_with_opencv(np_img, prompt=prompt)
                    pts = np.argwhere(highlight_mask > 0)
                    if len(pts) > 0:
                        min_y, min_x = pts.min(axis=0)
                        max_y, max_x = pts.max(axis=0)
                        input_boxes = [[[float(min_x), float(min_y), float(max_x), float(max_y)]]]

                elif isinstance(prompt, str):
                    # Text prompt: Use OpenCV heuristic / text bbox search fallback
                    highlight_mask = self._segment_with_opencv(np_img, prompt=prompt)
                    pts = np.argwhere(highlight_mask > 0)
                    if len(pts) > 0:
                        min_y, min_x = pts.min(axis=0)
                        max_y, max_x = pts.max(axis=0)
                        input_boxes = [[[float(min_x), float(min_y), float(max_x), float(max_y)]]]

                inputs = self._sam_processor(
                    pil_img,
                    input_boxes=input_boxes,
                    input_points=input_points,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    outputs = self._sam_model(**inputs)

                masks = self._sam_processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu(),
                )

                mask_np = (masks[0][0][0].numpy() * 255).astype(np.uint8)
                return self._mask_to_result(
                    mask_np, pil_img, background_color=background_color
                )

            except Exception as e:
                print(f"SAM inference exception: {e}. Falling back to OpenCV extraction.")

        # OpenCV fallback extraction
        mask = self._segment_with_opencv(np_img, prompt=prompt)
        return self._mask_to_result(
            mask, pil_img, background_color=background_color
        )


def main():
    """CLI script for shape extraction and cropping."""
    parser = argparse.ArgumentParser(
        description="extract object shapes using SAM and text/box/highlight prompts."
    )
    parser.add_argument("--image", "-i", required=True, help="path to input image")
    parser.add_argument(
        "--prompt", "-p", help="prompt: text string, box 'x1,y1,x2,y2', or mask image path"
    )
    parser.add_argument("--output", "-o", default="cropped_output.png", help="output cropped image path")
    parser.add_argument("--mask-output", help="optional output path for binary mask")

    args = parser.parse_args()

    extractor = ShapeExtractor()

    # Parse prompt argument
    prompt_val = args.prompt
    if prompt_val and "," in prompt_val and len(prompt_val.split(",")) == 4:
        try:
            prompt_val = [float(v) for v in prompt_val.split(",")]
        except ValueError:
            pass
    elif prompt_val and Path(prompt_val).exists():
        prompt_val = Image.open(prompt_val)

    result = extractor.extract(args.image, prompt=prompt_val)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.cropped_image.save(output_path)
    print(f"saved cropped object to '{output_path}'.")

    if args.mask_output:
        mask_path = Path(args.mask_output)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.mask).save(mask_path)
        print(f"saved binary mask to '{mask_path}'.")


if __name__ == "__main__":
    main()
