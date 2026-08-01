import csv
import itertools
import math
import random
from importlib import resources
from dataclasses import dataclass, field
from pathlib import Path

import cairo
from shapely.geometry import LineString
from tqdm.auto import tqdm

from mindset.generators._base import GeneratorConfig, generator, register
from mindset.shapes.manipulation import BaseDrawManipulatedObject
from mindset.utils import to_list


class DrawManipulatedOutline(BaseDrawManipulatedObject):
    """Draws object outline manipulations using PyCairo and Shapely."""

    def __init__(
        self,
        outline_mode="dotted",
        outline_color=(255, 255, 255),
        outline_scale=6.0,
        outline_distance=12.0,
        outline_asset_image="",
        outline_text="A",
        outline_font="Sans",
        random_modes=None,
        seed=None,
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
        self.outline_text = str(outline_text)
        self.outline_font = str(outline_font)
        self.rotate_outline_shapes = rotate_outline_shapes
        self.fill_interior = fill_interior
        self.interior_color = interior_color
        self.seed = seed

        if random_modes is None:
            self.random_modes = ["dotted", "oriented_lines", "oriented_shapes", "text"]
            if self.outline_asset_image:
                self.random_modes.append("asset_shape")
        else:
            self.random_modes = list(random_modes)

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

    def _render_element(self, ctx, p1, angle, mode_type, asset_contour=None):
        """render a single vector outline element at position p1."""
        stamp_sz = max(1.0, self.outline_scale)
        half_sz = stamp_sz / 2.0

        if mode_type == "dotted":
            radius = max(0.5, stamp_sz / 2.0)
            ctx.arc(p1.x, p1.y, radius, 0, 2 * math.pi)
            ctx.fill()

        elif mode_type == "oriented_lines":
            ctx.set_line_width(max(0.5, stamp_sz / 3.0))
            x1 = p1.x - half_sz * math.cos(angle)
            y1 = p1.y - half_sz * math.sin(angle)
            x2 = p1.x + half_sz * math.cos(angle)
            y2 = p1.y + half_sz * math.sin(angle)
            ctx.move_to(x1, y1)
            ctx.line_to(x2, y2)
            ctx.stroke()

        elif mode_type == "oriented_shapes":
            ctx.save()
            ctx.translate(p1.x, p1.y)
            if angle != 0:
                ctx.rotate(angle)
            ctx.rectangle(-half_sz, -half_sz, stamp_sz, stamp_sz)
            ctx.fill()
            ctx.restore()

        elif mode_type == "asset_shape":
            if asset_contour is not None:
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

        elif mode_type == "text" or mode_type == "letters":
            ctx.select_font_face(
                self.outline_font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD
            )
            text_sz = max(4.0, stamp_sz)
            ctx.set_font_size(text_sz)
            ctx.save()
            ctx.translate(p1.x, p1.y)
            if angle != 0:
                ctx.rotate(angle)
            ctx.move_to(-text_sz / 4.0, text_sz / 3.0)
            ctx.text_path(self.outline_text)
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

        elif self.outline_mode == "random":
            step = max(1.0, self.outline_distance)
            rng = random.Random(self.seed) if self.seed is not None else random
            asset_contour = self.load_asset_vector_contour(self.outline_asset_image)

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

                    sampled_mode = rng.choice(self.random_modes)
                    self._render_element(ctx, p1, angle, sampled_mode, asset_contour)

        else:
            # Single element type outline (dotted, oriented_lines, oriented_shapes, asset_shape, text)
            step = max(1.0, self.outline_distance)
            asset_contour = self.load_asset_vector_contour(self.outline_asset_image)

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

                    self._render_element(
                        ctx, p1, angle, self.outline_mode, asset_contour
                    )

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

    linedrawing_input_folder: str | None = field(
        default=None,
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
                "text",
                "letters",
                "random",
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
    outline_text: list = field(
        default_factory=lambda: ["A"],
        metadata={"label": "text/letter string options for text mode"},
    )
    outline_font: str = field(
        default="Sans",
        metadata={"label": "font family name for text mode"},
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
        default_factory=lambda: [10.0],
        metadata={"label": "scale/size/length of dots/dashes/elements along outline options"},
    )
    fill_interior: list = field(
        default_factory=lambda: [False],
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

    if config.linedrawing_input_folder is None:
        import mindset
        ROOT = Path(mindset.__file__).parent
        linedrawing_input_folder = ROOT / "assets" / "linedrawings" / "cropped"
    else:
        linedrawing_input_folder = Path(config.linedrawing_input_folder)

    all_categories = [p.stem for p in linedrawing_input_folder.glob("*") if p.is_dir()]
    for cat in all_categories:
        (output_folder / cat).mkdir(exist_ok=True, parents=True)

    image_files = sorted(linedrawing_input_folder.rglob("*.jpg")) + sorted(
        linedrawing_input_folder.rglob("*.png")
    )

    modes = to_list(config.outline_mode)
    scales = [float(s) for s in to_list(config.outline_scale)]
    distances = [float(d) for d in to_list(config.outline_distance)]
    rotates = [bool(r) for r in to_list(config.rotate_outline_shapes)]
    fills = [bool(f) for f in to_list(config.fill_interior)]
    asset_images = to_list(config.outline_asset_image)
    texts = to_list(config.outline_text)

    combinations = list(
        itertools.product(modes, scales, distances, rotates, fills, asset_images, texts)
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "IterNum",
                "Class",
                "OutlineMode",
                "OutlineDistance",
                "OutlineScale",
                "OutlineAssetImage",
                "OutlineText",
                "RotateOutlineShapes",
                "FillInterior",
                "BackgroundColor",
                "OutlineColor",
                "Path",
            ]
        )

        n = 0
        for img_path in tqdm(image_files, desc="processing categories"):
            class_name = img_path.parent.stem
            image_name = img_path.stem

            for o_mode, o_scale, o_dist, o_rot, o_fill, o_asset, o_text in combinations:
                # Handle random mode
                random_modes = [m for m in modes if m != "random"] if o_mode == "random" else None
                ds = DrawManipulatedOutline(
                    background=config.background_color,
                    canvas_size=config.canvas_size,
                    antialiasing=config.antialiasing,
                    obj_longest_side=config.object_longest_side,
                    outline_mode=o_mode,
                    outline_color=config.outline_color,
                    outline_asset_image=o_asset,
                    outline_text=o_text,
                    outline_font=config.outline_font,
                    rotate_outline_shapes=o_rot,
                    outline_distance=o_dist,
                    outline_scale=o_scale,
                    random_modes=random_modes,
                    fill_interior=o_fill,
                    interior_color=config.interior_color,
                    seed=n,
                    linedrawing_input_folder=linedrawing_input_folder,
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

                if o_mode in ["text", "letters"]:
                    name_parts.append(f"txt-{o_text}")
                elif o_asset and o_mode == "asset_shape":
                    name_parts.append(f"asset-{Path(o_asset).stem}")

                filename = "_".join(name_parts) + ".png"
                path = Path(class_name) / filename

                img.save(output_folder / path)
                writer.writerow(
                    [
                        n,
                        class_name,
                        o_mode,
                        o_dist,
                        o_scale,
                        o_asset,
                        o_text,
                        o_rot,
                        o_fill,
                        ds.background,
                        config.outline_color,
                        path,
                    ]
                )
                n += 1

    return str(output_folder)
