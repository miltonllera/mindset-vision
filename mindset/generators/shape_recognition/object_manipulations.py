"""object manipulations dataset generator using Shapely and PyCairo."""

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import cairo
import cv2
import numpy as np
from PIL import Image
from shapely.geometry import LineString, Polygon
from tqdm.auto import tqdm

from mindset.drawing.base import DrawStimuli
from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils import apply_antialiasing


class DrawManipulatedObject(DrawStimuli):
    """draws object manipulations (texture & outline) using PyCairo and Shapely."""

    def __init__(
        self,
        obj_longest_side=200,
        texture_mode="flat",
        texture_color=(255, 255, 255),
        texture_line_spacing=10,
        texture_line_width=2,
        texture_angle=45.0,
        texture_asset_image="",
        outline_mode="solid",
        outline_color=(255, 255, 255),
        outline_width=2,
        outline_asset_image="",
        outline_obj_distance=12.0,
        outline_obj_size=6.0,
        rotate_outline_shapes=False,
        linedrawing_input_folder="mindset/assets/linedrawings/cropped/",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side
        self.texture_mode = texture_mode
        self.texture_color = texture_color
        self.texture_line_spacing = texture_line_spacing
        self.texture_line_width = texture_line_width
        self.texture_angle = texture_angle
        self.texture_asset_image = texture_asset_image
        self.outline_mode = outline_mode
        self.outline_color = outline_color
        self.outline_width = outline_width
        self.outline_asset_image = outline_asset_image

        # Handle backward-compatible alias kwargs if provided
        if "dot_distance" in kwargs:
            outline_obj_distance = kwargs.pop("dot_distance")
        if "dash_gap" in kwargs:
            outline_obj_distance = kwargs.pop("dash_gap")

        if "dot_size" in kwargs:
            outline_obj_size = kwargs.pop("dot_size")
        if "dash_length" in kwargs:
            outline_obj_size = kwargs.pop("dash_length")
        if "oriented_line_length" in kwargs:
            outline_obj_size = kwargs.pop("oriented_line_length")
        if "asset_shape_size" in kwargs:
            outline_obj_size = kwargs.pop("asset_shape_size")

        self.outline_obj_distance = outline_obj_distance
        self.outline_obj_size = outline_obj_size
        self.rotate_outline_shapes = rotate_outline_shapes
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
            # Search in linedrawing_input_folder by category name
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

        # Center at origin (0,0) and normalize scale to unit size 1.0
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

        # Find bounding box across all contours to scale object to obj_longest_side
        all_pts = np.vstack(contours).reshape(-1, 2)
        min_x, min_y = all_pts[:, 0].min(), all_pts[:, 1].min()
        max_x, max_y = all_pts[:, 0].max(), all_pts[:, 1].max()

        w = max(1, max_x - min_x)
        h = max(1, max_y - min_y)
        scale = float(self.obj_longest_side) / float(max(w, h))

        # Center on canvas
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

                # Check hierarchy: hierarchy[idx][3] == -1 means outer contour
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

    def _render_texture(self, ctx, outer_polygons, linestrings):
        """render vector texture inside outer polygons clipped boundaries."""
        if self.texture_mode == "none":
            return

        ctx.save()
        # Set up clipping path using outer polygons
        for poly in outer_polygons:
            exterior_ls = LineString(poly.exterior.coords)
            self._draw_path_to_cairo(ctx, exterior_ls)
        ctx.clip()

        bg_r, bg_g, bg_b = self._normalize_color(self.background)
        tx_r, tx_g, tx_b = self._normalize_color(self.texture_color)

        canvas_w, canvas_h = self.canvas_size

        if self.texture_mode == "flat":
            ctx.set_source_rgb(tx_r, tx_g, tx_b)
            ctx.paint()

        elif self.texture_mode == "lines":
            ctx.set_source_rgb(bg_r, bg_g, bg_b)
            ctx.paint()

            ctx.set_source_rgb(tx_r, tx_g, tx_b)
            ctx.set_line_width(self.texture_line_width)

            rad = math.radians(self.texture_angle)
            diag = math.hypot(canvas_w, canvas_h)
            cx, cy = canvas_w / 2.0, canvas_h / 2.0

            spacing = max(1, self.texture_line_spacing)
            num_lines = int(diag / spacing) + 2

            for i in range(-num_lines, num_lines):
                offset = i * spacing
                dx = offset * math.cos(rad + math.pi / 2)
                dy = offset * math.sin(rad + math.pi / 2)

                x1 = (cx + dx) - diag * math.cos(rad)
                y1 = (cy + dy) - diag * math.sin(rad)
                x2 = (cx + dx) + diag * math.cos(rad)
                y2 = (cy + dy) + diag * math.sin(rad)

                ctx.move_to(x1, y1)
                ctx.line_to(x2, y2)
                ctx.stroke()

        elif self.texture_mode == "grid":
            ctx.set_source_rgb(bg_r, bg_g, bg_b)
            ctx.paint()

            ctx.set_source_rgb(tx_r, tx_g, tx_b)
            ctx.set_line_width(self.texture_line_width)

            spacing = max(1, self.texture_line_spacing)

            for x in range(0, canvas_w + spacing, spacing):
                ctx.move_to(x, 0)
                ctx.line_to(x, canvas_h)
                ctx.stroke()

            for y in range(0, canvas_h + spacing, spacing):
                ctx.move_to(0, y)
                ctx.line_to(canvas_w, y)
                ctx.stroke()

        elif self.texture_mode == "dots":
            ctx.set_source_rgb(bg_r, bg_g, bg_b)
            ctx.paint()

            ctx.set_source_rgb(tx_r, tx_g, tx_b)
            spacing = max(2, self.texture_line_spacing)
            radius = max(1.0, self.texture_line_width / 2.0)

            for x in range(0, canvas_w + spacing, spacing):
                for y in range(0, canvas_h + spacing, spacing):
                    ctx.arc(x, y, radius, 0, 2 * math.pi)
                    ctx.fill()

        elif self.texture_mode == "checkerboard":
            ctx.set_source_rgb(bg_r, bg_g, bg_b)
            ctx.paint()

            ctx.set_source_rgb(tx_r, tx_g, tx_b)
            spacing = max(2, self.texture_line_spacing)

            for x in range(0, canvas_w, spacing):
                for y in range(0, canvas_h, spacing):
                    if ((x // spacing) + (y // spacing)) % 2 == 0:
                        ctx.rectangle(x, y, spacing, spacing)
                        ctx.fill()

        elif self.texture_mode == "asset_shape":
            ctx.set_source_rgb(bg_r, bg_g, bg_b)
            ctx.paint()

            asset_contour = self.load_asset_vector_contour(self.texture_asset_image)
            if asset_contour is not None:
                ctx.set_source_rgb(tx_r, tx_g, tx_b)
                spacing = max(4, self.texture_line_spacing)
                stamp_sz = max(2.0, self.outline_obj_size)
                rad = math.radians(self.texture_angle)

                for x in range(0, canvas_w + spacing, spacing):
                    for y in range(0, canvas_h + spacing, spacing):
                        ctx.save()
                        ctx.translate(x, y)
                        if self.texture_angle != 0:
                            ctx.rotate(rad)
                        ctx.scale(stamp_sz, stamp_sz)

                        ctx.move_to(asset_contour[0, 0], asset_contour[0, 1])
                        for pt in asset_contour[1:]:
                            ctx.line_to(pt[0], pt[1])
                        ctx.close_path()
                        ctx.fill()
                        ctx.restore()

        ctx.restore()

    def _render_outline(self, ctx, linestrings):
        """render vector outline along Shapely LineStrings with subpixel precision."""
        if self.outline_mode == "none":
            return

        out_r, out_g, out_b = self._normalize_color(self.outline_color)
        ctx.set_source_rgb(out_r, out_g, out_b)

        if self.outline_mode == "solid":
            ctx.set_line_width(self.outline_width)
            for ls in linestrings:
                self._draw_path_to_cairo(ctx, ls)
                ctx.stroke()

        elif self.outline_mode == "dashed":
            ctx.set_line_width(self.outline_width)
            # Use outline_obj_size for dash length and outline_obj_distance for dash gap
            dash_len = max(1.0, self.outline_obj_size)
            dash_gap = max(1.0, self.outline_obj_distance)
            ctx.set_dash([dash_len, dash_gap])
            for ls in linestrings:
                self._draw_path_to_cairo(ctx, ls)
                ctx.stroke()
            ctx.set_dash([])  # reset dash pattern

        elif self.outline_mode == "dotted":
            step = max(1.0, self.outline_obj_distance)
            radius = max(0.5, self.outline_obj_size / 2.0)

            for ls in linestrings:
                length = ls.length
                if length <= 0:
                    continue
                num_dots = int(math.floor(length / step))
                for i in range(num_dots):
                    d = i * step
                    p = ls.interpolate(d)
                    ctx.arc(p.x, p.y, radius, 0, 2 * math.pi)
                    ctx.fill()

        elif self.outline_mode == "oriented_lines":
            step = max(1.0, self.outline_obj_distance)
            half_len = max(1.0, self.outline_obj_size / 2.0)
            ctx.set_line_width(self.outline_width)

            for ls in linestrings:
                length = ls.length
                if length <= 0:
                    continue
                num_samples = int(math.floor(length / step))
                for i in range(num_samples):
                    d = i * step
                    p1 = ls.interpolate(d)
                    p2 = ls.interpolate(min(d + 1.0, length))

                    dx = p2.x - p1.x
                    dy = p2.y - p1.y
                    angle = math.atan2(dy, dx) if self.rotate_outline_shapes else 0.0

                    x1 = p1.x - half_len * math.cos(angle)
                    y1 = p1.y - half_len * math.sin(angle)
                    x2 = p1.x + half_len * math.cos(angle)
                    y2 = p1.y + half_len * math.sin(angle)

                    ctx.move_to(x1, y1)
                    ctx.line_to(x2, y2)
                    ctx.stroke()

        elif self.outline_mode == "oriented_shapes":
            step = max(1.0, self.outline_obj_distance)
            half_sz = max(1.0, self.outline_obj_size / 2.0)

            for ls in linestrings:
                length = ls.length
                if length <= 0:
                    continue
                num_samples = int(math.floor(length / step))
                for i in range(num_samples):
                    d = i * step
                    p1 = ls.interpolate(d)
                    p2 = ls.interpolate(min(d + 1.0, length))

                    dx = p2.x - p1.x
                    dy = p2.y - p1.y
                    angle = math.atan2(dy, dx) if self.rotate_outline_shapes else 0.0

                    ctx.save()
                    ctx.translate(p1.x, p1.y)
                    if angle != 0:
                        ctx.rotate(angle)
                    ctx.rectangle(-half_sz, -half_sz, self.outline_obj_size, self.outline_obj_size)
                    ctx.fill()
                    ctx.restore()

        elif self.outline_mode == "asset_shape":
            asset_contour = self.load_asset_vector_contour(self.outline_asset_image)
            if asset_contour is not None:
                step = max(2.0, self.outline_obj_distance)
                stamp_sz = max(2.0, self.outline_obj_size)

                for ls in linestrings:
                    length = ls.length
                    if length <= 0:
                        continue
                    num_samples = int(math.floor(length / step))
                    for i in range(num_samples):
                        d = i * step
                        p1 = ls.interpolate(d)
                        p2 = ls.interpolate(min(d + 1.0, length))

                        dx = p2.x - p1.x
                        dy = p2.y - p1.y
                        angle = math.atan2(dy, dx) if self.rotate_outline_shapes else 0.0

                        ctx.save()
                        ctx.translate(p1.x, p1.y)
                        if angle != 0:
                            ctx.rotate(angle)
                        ctx.scale(stamp_sz, stamp_sz)

                        ctx.move_to(asset_contour[0, 0], asset_contour[0, 1])
                        for pt in asset_contour[1:]:
                            ctx.line_to(pt[0], pt[1])
                        ctx.close_path()
                        ctx.fill()
                        ctx.restore()

    def generate_image(self, image_path):
        """process input image and render texture & outline manipulation canvas."""
        outer_polygons, linestrings = self.extract_contours(image_path)

        canvas_w, canvas_h = self.canvas_size

        # Create PyCairo ImageSurface
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, canvas_w, canvas_h)
        ctx = cairo.Context(surface)

        ctx.set_antialias(cairo.ANTIALIAS_BEST)

        # Set background color
        bg_r, bg_g, bg_b = self._normalize_color(self.background)
        ctx.set_source_rgb(bg_r, bg_g, bg_b)
        ctx.paint()

        if outer_polygons or linestrings:
            # 1. Render Texture
            self._render_texture(ctx, outer_polygons, linestrings)

            # 2. Render Outline
            self._render_outline(ctx, linestrings)

        # Convert Cairo ARGB32 surface to PIL RGBA Image
        buf = surface.get_data()
        img_array = np.frombuffer(buf, np.uint8).reshape((canvas_h, canvas_w, 4))
        # Cairo ARGB32 is BGRA in memory on little-endian systems
        pil_img = Image.fromarray(img_array[:, :, [2, 1, 0, 3]], mode="RGBA")

        return apply_antialiasing(pil_img) if self.antialiasing else pil_img


# ---------------------------------------------------------------------------
# generator config and entry point
# ---------------------------------------------------------------------------


@dataclass
class ObjectManipulationsConfig(GeneratorConfig):
    """config for object manipulations dataset (texture & outline)."""

    linedrawing_input_folder: str = field(
        default="mindset/assets/linedrawings/cropped/",
        metadata={"label": "input folder with line drawings / images"},
    )
    object_longest_side: int = field(
        default=200,
        metadata={
            "min": 50,
            "max": 500,
            "step": 10,
            "label": "object longest side (px)",
        },
    )
    texture_mode: str = field(
        default="flat",
        metadata={
            "choices": [
                "flat",
                "lines",
                "grid",
                "dots",
                "checkerboard",
                "asset_shape",
                "none",
            ],
            "label": "texture manipulation mode",
        },
    )
    texture_color: list = field(
        default_factory=lambda: [255, 255, 255],
        metadata={"label": "texture color (RGB)"},
    )
    texture_line_spacing: int = field(
        default=10,
        metadata={"min": 2, "max": 50, "step": 1, "label": "texture pattern spacing"},
    )
    texture_line_width: int = field(
        default=2,
        metadata={"min": 1, "max": 20, "step": 1, "label": "texture pattern line width"},
    )
    texture_angle: float = field(
        default=45.0,
        metadata={
            "min": 0.0,
            "max": 360.0,
            "step": 5.0,
            "label": "texture pattern angle (degrees)",
        },
    )
    texture_asset_image: str = field(
        default="",
        metadata={"label": "asset category or image path for texture shape pattern"},
    )
    outline_mode: str = field(
        default="solid",
        metadata={
            "choices": [
                "solid",
                "dotted",
                "dashed",
                "oriented_lines",
                "oriented_shapes",
                "asset_shape",
                "none",
            ],
            "label": "outline manipulation mode",
        },
    )
    outline_color: list = field(
        default_factory=lambda: [255, 255, 255],
        metadata={"label": "outline color (RGB)"},
    )
    outline_width: int = field(
        default=2,
        metadata={"min": 1, "max": 20, "step": 1, "label": "outline width"},
    )
    outline_asset_image: str = field(
        default="",
        metadata={"label": "asset category or image path for outline shape stamping"},
    )
    rotate_outline_shapes: bool = field(
        default=False,
        metadata={"label": "rotate outline shapes to align with edge tangents"},
    )
    outline_obj_distance: float = field(
        default=12.0,
        metadata={
            "min": 1.0,
            "max": 50.0,
            "step": 1.0,
            "label": "distance between dots/dashes/elements along outline",
        },
    )
    outline_obj_size: float = field(
        default=6.0,
        metadata={
            "min": 1.0,
            "max": 50.0,
            "step": 0.5,
            "label": "size/length of dots/dashes/elements along outline",
        },
    )
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    output_folder: str = field(
        default="data/shape_and_object_recognition/object_manipulations",
        metadata={"label": "output folder"},
    )


@register("object_manipulations", "shape_recognition")
@generator(ObjectManipulationsConfig)
def generate_all(config: ObjectManipulationsConfig):
    """generate object manipulations dataset with vector texture and outline controls."""
    output_folder = Path(config.output_folder)
    linedrawing_input_folder = Path(config.linedrawing_input_folder)

    all_categories = [p.stem for p in linedrawing_input_folder.glob("*") if p.is_dir()]
    for cat in all_categories:
        (output_folder / cat).mkdir(exist_ok=True, parents=True)

    ds = DrawManipulatedObject(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        obj_longest_side=config.object_longest_side,
        texture_mode=config.texture_mode,
        texture_color=config.texture_color,
        texture_line_spacing=config.texture_line_spacing,
        texture_line_width=config.texture_line_width,
        texture_angle=config.texture_angle,
        texture_asset_image=config.texture_asset_image,
        outline_mode=config.outline_mode,
        outline_color=config.outline_color,
        outline_width=config.outline_width,
        outline_asset_image=config.outline_asset_image,
        rotate_outline_shapes=config.rotate_outline_shapes,
        outline_obj_distance=config.outline_obj_distance,
        outline_obj_size=config.outline_obj_size,
        linedrawing_input_folder=config.linedrawing_input_folder,
    )

    image_files = sorted(linedrawing_input_folder.rglob("*.jpg")) + sorted(
        linedrawing_input_folder.rglob("*.png")
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "Class",
                "TextureMode",
                "OutlineMode",
                "TextureAssetImage",
                "OutlineAssetImage",
                "RotateOutlineShapes",
                "BackgroundColor",
                "TextureColor",
                "OutlineColor",
                "OutlineObjDistance",
                "OutlineObjSize",
                "IterNum",
            ]
        )

        for n, img_path in enumerate(tqdm(image_files)):
            class_name = img_path.parent.stem
            image_name = img_path.stem
            img = ds.generate_image(img_path)
            path = Path(class_name) / f"{image_name}.png"
            img.save(output_folder / path)
            writer.writerow(
                [
                    path,
                    class_name,
                    config.texture_mode,
                    config.outline_mode,
                    config.texture_asset_image,
                    config.outline_asset_image,
                    config.rotate_outline_shapes,
                    ds.background,
                    config.texture_color,
                    config.outline_color,
                    config.outline_obj_distance,
                    config.outline_obj_size,
                    n,
                ]
            )

    return str(output_folder)
