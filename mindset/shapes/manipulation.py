"""base drawing class for texture and outline object manipulations using Shapely and PyCairo."""

import math
from pathlib import Path

import cairo
import cv2
import numpy as np
from PIL import Image
from shapely.geometry import LineString, Polygon

from mindset.drawing.base import DrawStimuli
from mindset.utils import apply_antialiasing


class BaseDrawManipulatedObject(DrawStimuli):
    """base class for drawing PyCairo and Shapely object manipulations."""

    def __init__(
        self,
        obj_longest_side=200,
        linedrawing_input_folder="mindset/assets/linedrawings/cropped/",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side
        self.linedrawing_input_folder = Path(linedrawing_input_folder)
        self._asset_contour_cache = {}

    def _normalize_color(self, color):
        """convert RGB tuple (0-255) to Cairo float RGB (0.0-1.0)."""
        if isinstance(color, (list, tuple)):
            return (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
        return (1.0, 1.0, 1.0)

    def load_asset_vector_contour(self, asset_spec):
        """load an asset image (by path or category name), extract contour, and normalize to unit box [-0.5, 0.5]."""
        if not asset_spec:
            return None
        if asset_spec in self._asset_contour_cache:
            return self._asset_contour_cache[asset_spec]

        asset_path = Path(asset_spec)
        if not asset_path.exists():
            cat_dir = self.linedrawing_input_folder / asset_spec
            if cat_dir.exists() and cat_dir.is_dir():
                found = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png"))
                if found:
                    asset_path = found[0]

        if not asset_path.exists():
            print(f"warning: asset shape '{asset_spec}' not found.")
            return None

        img = cv2.imread(str(asset_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        _, binary = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
        min_x, min_y = contour[:, 0].min(), contour[:, 1].min()
        max_x, max_y = contour[:, 0].max(), contour[:, 1].max()

        w = max(1.0, max_x - min_x)
        h = max(1.0, max_y - min_y)
        max_dim = max(w, h)

        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0

        norm_contour = np.zeros_like(contour)
        norm_contour[:, 0] = (contour[:, 0] - cx) / max_dim
        norm_contour[:, 1] = (contour[:, 1] - cy) / max_dim

        self._asset_contour_cache[asset_spec] = norm_contour
        return norm_contour

    def extract_contours(self, image_path):
        """load image, segment object contour and holes, and scale/center onto canvas."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"could not load image: {image_path}")

        _, binary = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            return [], []

        all_pts = np.vstack(contours).reshape(-1, 2)
        min_x, min_y = all_pts[:, 0].min(), all_pts[:, 1].min()
        max_x, max_y = all_pts[:, 0].max(), all_pts[:, 1].max()

        w = max(1, max_x - min_x)
        h = max(1, max_y - min_y)
        scale = float(self.obj_longest_side) / float(max(w, h))

        canvas_w, canvas_h = self.canvas_size
        cx_obj = (min_x + max_x) / 2.0
        cy_obj = (min_y + max_y) / 2.0

        offset_x = (canvas_w / 2.0) - (cx_obj * scale)
        offset_y = (canvas_h / 2.0) - (cy_obj * scale)

        outer_polygons = []
        all_linestrings = []

        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for idx, c in enumerate(contours):
                if len(c) < 3:
                    continue
                pts = c.reshape(-1, 2).astype(float) * scale
                pts[:, 0] += offset_x
                pts[:, 1] += offset_y

                ls = LineString(pts)
                all_linestrings.append(ls)

                if hierarchy[idx][3] == -1:
                    outer_polygons.append(Polygon(pts))

        return outer_polygons, all_linestrings

    def _draw_path_to_cairo(self, ctx, linestring):
        """add a Shapely LineString as a path to PyCairo context."""
        coords = list(linestring.coords)
        if not coords:
            return
        ctx.move_to(coords[0][0], coords[0][1])
        for pt in coords[1:]:
            ctx.line_to(pt[0], pt[1])
        ctx.close_path()

    def _surface_to_pil(self, surface):
        """convert Cairo ARGB32 surface to PIL RGBA image."""
        canvas_w, canvas_h = self.canvas_size
        buf = surface.get_data()
        img_array = np.frombuffer(buf, np.uint8).reshape((canvas_h, canvas_w, 4))
        pil_img = Image.fromarray(img_array[:, :, [2, 1, 0, 3]], mode="RGBA")
        return apply_antialiasing(pil_img) if self.antialiasing else pil_img
