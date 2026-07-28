"""manipulated outlines dataset generator using Shapely and PyCairo."""

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import cairo
from shapely.geometry import LineString
from tqdm.auto import tqdm

from mindset.generators._base import GeneratorConfig, generator, register
from mindset.shapes.manipulation import BaseDrawManipulatedObject



class DrawManipulatedOutline(BaseDrawManipulatedObject):
    """draws object outline manipulations using PyCairo and Shapely."""

    def __init__(
        self,
        outline_mode="dotted",
        outline_color=(255, 255, 255),
        outline_width=2,
        outline_asset_image="",
        outline_obj_distance=12.0,
        outline_obj_size=6.0,
        rotate_outline_shapes=False,
        fill_interior=True,
        interior_color=(255, 255, 255),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
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
        self.fill_interior = fill_interior
        self.interior_color = interior_color

    def _render_interior(self, ctx, outer_polygons):
        """fill interior silhouette with solid interior color if enabled."""
        if not self.fill_interior:
            return

        ctx.save()
        int_r, int_g, int_b = self._normalize_color(self.interior_color)
        ctx.set_source_rgb(int_r, int_g, int_b)

        for poly in outer_polygons:
            exterior_ls = LineString(poly.exterior.coords)
            self._draw_path_to_cairo(ctx, exterior_ls)
            ctx.fill()

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
            dash_len = max(1.0, self.outline_obj_size)
            dash_gap = max(1.0, self.outline_obj_distance)
            ctx.set_dash([dash_len, dash_gap])
            for ls in linestrings:
                self._draw_path_to_cairo(ctx, ls)
                ctx.stroke()
            ctx.set_dash([])

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
        """process input image and render manipulated outline canvas."""
        outer_polygons, linestrings = self.extract_contours(image_path)
        canvas_w, canvas_h = self.canvas_size

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, canvas_w, canvas_h)
        ctx = cairo.Context(surface)
        ctx.set_antialias(cairo.ANTIALIAS_BEST)

        bg_r, bg_g, bg_b = self._normalize_color(self.background)
        ctx.set_source_rgb(bg_r, bg_g, bg_b)
        ctx.paint()

        if outer_polygons or linestrings:
            self._render_interior(ctx, outer_polygons)
            self._render_outline(ctx, linestrings)

        return self._surface_to_pil(surface)


# ---------------------------------------------------------------------------
# generator config and entry point
# ---------------------------------------------------------------------------


@dataclass
class ManipulatedOutlinesConfig(GeneratorConfig):
    """config for manipulated outlines dataset."""

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
    outline_mode: str = field(
        default="dotted",
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
    fill_interior: bool = field(
        default=True,
        metadata={"label": "fill object interior with solid color silhouette"},
    )
    interior_color: list = field(
        default_factory=lambda: [255, 255, 255],
        metadata={"label": "interior silhouette color (RGB)"},
    )
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    output_folder: str = field(
        default="data/shape_and_object_recognition/manipulated_outlines",
        metadata={"label": "output folder"},
    )


@register("manipulated_outlines", "shape_recognition")
@generator(ManipulatedOutlinesConfig)
def generate_all(config: ManipulatedOutlinesConfig):
    """generate manipulated outlines dataset with vector outline controls."""
    output_folder = Path(config.output_folder)
    linedrawing_input_folder = Path(config.linedrawing_input_folder)

    all_categories = [p.stem for p in linedrawing_input_folder.glob("*") if p.is_dir()]
    for cat in all_categories:
        (output_folder / cat).mkdir(exist_ok=True, parents=True)

    ds = DrawManipulatedOutline(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        obj_longest_side=config.object_longest_side,
        outline_mode=config.outline_mode,
        outline_color=config.outline_color,
        outline_width=config.outline_width,
        outline_asset_image=config.outline_asset_image,
        rotate_outline_shapes=config.rotate_outline_shapes,
        outline_obj_distance=config.outline_obj_distance,
        outline_obj_size=config.outline_obj_size,
        fill_interior=config.fill_interior,
        interior_color=config.interior_color,
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
                "OutlineMode",
                "OutlineAssetImage",
                "RotateOutlineShapes",
                "FillInterior",
                "BackgroundColor",
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
                    config.outline_mode,
                    config.outline_asset_image,
                    config.rotate_outline_shapes,
                    config.fill_interior,
                    ds.background,
                    config.outline_color,
                    config.outline_obj_distance,
                    config.outline_obj_size,
                    n,
                ]
            )

    return str(output_folder)
