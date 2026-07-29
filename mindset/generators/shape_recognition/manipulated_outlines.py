"""manipulated outlines dataset generator using Shapely and PyCairo."""

import csv
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cairo
from shapely.geometry import LineString
from tqdm.auto import tqdm

from mindset.generators._base import GeneratorConfig, generator, register
from mindset.shapes.manipulation import BaseDrawManipulatedObject


def _to_list(val: Any) -> list[Any]:
    """ensure val is a list of options."""
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]


class DrawManipulatedOutline(BaseDrawManipulatedObject):
    """draws object outline manipulations using PyCairo and Shapely."""

    def __init__(
        self,
        outline_mode="dotted",
        outline_color=(255, 255, 255),
        outline_scale=6.0,
        outline_distance=12.0,
        outline_asset_image="",
        rotate_outline_shapes=False,
        fill_interior=True,
        interior_color=(255, 255, 255),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.outline_mode = outline_mode
        self.outline_color = outline_color
        self.outline_scale = float(outline_scale)
        self.outline_distance = float(outline_distance)
        self.outline_asset_image = outline_asset_image
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
            ctx.set_line_width(max(0.5, self.outline_scale))
            for ls in linestrings:
                self._draw_path_to_cairo(ctx, ls)
                ctx.stroke()

        elif self.outline_mode == "dashed":
            ctx.set_line_width(max(0.5, self.outline_scale))
            dash_len = max(1.0, self.outline_scale)
            dash_gap = max(1.0, self.outline_distance)
            ctx.set_dash([dash_len, dash_gap])
            for ls in linestrings:
                self._draw_path_to_cairo(ctx, ls)
                ctx.stroke()
            ctx.set_dash([])

        elif self.outline_mode == "dotted":
            step = max(1.0, self.outline_distance)
            radius = max(0.5, self.outline_scale / 2.0)

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
            step = max(1.0, self.outline_distance)
            half_len = max(1.0, self.outline_scale / 2.0)
            ctx.set_line_width(max(0.5, self.outline_scale / 3.0))

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
            step = max(1.0, self.outline_distance)
            half_sz = max(1.0, self.outline_scale / 2.0)

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
                    ctx.rectangle(-half_sz, -half_sz, self.outline_scale, self.outline_scale)
                    ctx.fill()
                    ctx.restore()

        elif self.outline_mode == "asset_shape":
            asset_contour = self.load_asset_vector_contour(self.outline_asset_image)
            if asset_contour is not None:
                step = max(1.0, self.outline_distance)
                stamp_sz = max(1.0, self.outline_scale)

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
    outline_mode: list = field(
        default_factory=lambda: ["dotted"],
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
            "label": "outline manipulation mode options",
        },
    )
    outline_color: list = field(
        default_factory=lambda: [255, 255, 255],
        metadata={"label": "outline color (RGB)"},
    )
    outline_asset_image: list = field(
        default_factory=lambda: [""],
        metadata={"label": "asset categories or image paths for outline shape stamping"},
    )
    rotate_outline_shapes: list = field(
        default_factory=lambda: [False],
        metadata={"label": "rotate outline shapes options (True/False)"},
    )
    outline_distance: list = field(
        default_factory=lambda: [12.0],
        metadata={"label": "distance between dots/dashes/elements along outline options"},
    )
    outline_scale: list = field(
        default_factory=lambda: [6.0],
        metadata={"label": "scale/size/length of dots/dashes/elements along outline options"},
    )
    fill_interior: list = field(
        default_factory=lambda: [True],
        metadata={"label": "fill object interior options (True/False)"},
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
    """generate manipulated outlines dataset across all parameter combinations."""
    output_folder = Path(config.output_folder)
    linedrawing_input_folder = Path(config.linedrawing_input_folder)

    all_categories = [p.stem for p in linedrawing_input_folder.glob("*") if p.is_dir()]
    for cat in all_categories:
        (output_folder / cat).mkdir(exist_ok=True, parents=True)

    image_files = sorted(linedrawing_input_folder.rglob("*.jpg")) + sorted(
        linedrawing_input_folder.rglob("*.png")
    )

    modes = _to_list(config.outline_mode)
    scales = [float(s) for s in _to_list(config.outline_scale)]
    distances = [float(d) for d in _to_list(config.outline_distance)]
    rotates = [bool(r) for r in _to_list(config.rotate_outline_shapes)]
    fills = [bool(f) for f in _to_list(config.fill_interior)]
    asset_images = _to_list(config.outline_asset_image)

    combinations = list(
        itertools.product(modes, scales, distances, rotates, fills, asset_images)
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
                "OutlineDistance",
                "OutlineScale",
                "IterNum",
            ]
        )

        n = 0
        for img_path in tqdm(image_files, desc="processing categories"):
            class_name = img_path.parent.stem
            image_name = img_path.stem

            for o_mode, o_scale, o_dist, o_rot, o_fill, o_asset in combinations:
                ds = DrawManipulatedOutline(
                    background=config.background_color,
                    canvas_size=config.canvas_size,
                    antialiasing=config.antialiasing,
                    obj_longest_side=config.object_longest_side,
                    outline_mode=o_mode,
                    outline_color=config.outline_color,
                    outline_asset_image=o_asset,
                    rotate_outline_shapes=o_rot,
                    outline_distance=o_dist,
                    outline_scale=o_scale,
                    fill_interior=o_fill,
                    interior_color=config.interior_color,
                    linedrawing_input_folder=config.linedrawing_input_folder,
                )

                img = ds.generate_image(img_path)

                # Construct filename reflecting the exact manipulations
                fmt_sc = int(o_scale) if o_scale == int(o_scale) else o_scale
                fmt_dist = int(o_dist) if o_dist == int(o_dist) else o_dist

                name_parts = [image_name, f"out-{o_mode}"]
                if o_mode != "solid" and o_mode != "none":
                    name_parts.append(f"sc{fmt_sc}_dist{fmt_dist}")
                elif o_mode == "solid":
                    name_parts.append(f"sc{fmt_sc}")

                if o_rot:
                    name_parts.append("rot")
                if not o_fill:
                    name_parts.append("nofill")
                if o_asset:
                    name_parts.append(f"asset-{Path(o_asset).stem}")

                filename = "_".join(name_parts) + ".png"
                path = Path(class_name) / filename

                img.save(output_folder / path)
                writer.writerow(
                    [
                        path,
                        class_name,
                        o_mode,
                        o_asset,
                        o_rot,
                        o_fill,
                        ds.background,
                        config.outline_color,
                        o_dist,
                        o_scale,
                        n,
                    ]
                )
                n += 1

    return str(output_folder)
