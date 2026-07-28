"""manipulated textures dataset generator using Shapely and PyCairo."""

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


class DrawManipulatedTexture(BaseDrawManipulatedObject):
    """draws object texture manipulations (foreground or background target region) using PyCairo and Shapely."""

    def __init__(
        self,
        target_region="foreground",
        texture_mode="lines",
        texture_color=(255, 255, 255),
        texture_line_spacing=10,
        texture_line_width=2,
        texture_angle=45.0,
        texture_asset_image="",
        asset_shape_size=12.0,
        outline_color=(255, 255, 255),
        outline_width=1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_region = target_region
        self.texture_mode = texture_mode
        self.texture_color = texture_color
        self.texture_line_spacing = texture_line_spacing
        self.texture_line_width = texture_line_width
        self.texture_angle = texture_angle
        self.texture_asset_image = texture_asset_image
        self.asset_shape_size = asset_shape_size
        self.outline_color = outline_color
        self.outline_width = outline_width

    def _render_texture(self, ctx, outer_polygons):
        """render vector texture inside polygon mask or inverted background mask."""
        if self.texture_mode == "none":
            return

        canvas_w, canvas_h = self.canvas_size

        if self.target_region == "background":
            # Fill shape interior first with solid white silhouette
            ctx.save()
            int_r, int_g, int_b = self._normalize_color((255, 255, 255))
            ctx.set_source_rgb(int_r, int_g, int_b)
            for poly in outer_polygons:
                exterior_ls = LineString(poly.exterior.coords)
                self._draw_path_to_cairo(ctx, exterior_ls)
                ctx.fill()
            ctx.restore()

            # Invert mask for background using EVEN_ODD fill rule
            ctx.save()
            ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
            ctx.rectangle(0, 0, canvas_w, canvas_h)
            for poly in outer_polygons:
                exterior_ls = LineString(poly.exterior.coords)
                self._draw_path_to_cairo(ctx, exterior_ls)
            ctx.clip()

        else:
            # Foreground mask: clip to outer polygons
            ctx.save()
            for poly in outer_polygons:
                exterior_ls = LineString(poly.exterior.coords)
                self._draw_path_to_cairo(ctx, exterior_ls)
            ctx.clip()

        bg_r, bg_g, bg_b = self._normalize_color(self.background)
        tx_r, tx_g, tx_b = self._normalize_color(self.texture_color)

        if self.texture_mode == "flat":
            ctx.set_source_rgb(tx_r, tx_g, tx_b)
            ctx.paint()

        elif self.texture_mode == "lines":
            if self.target_region == "foreground":
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
            if self.target_region == "foreground":
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
            if self.target_region == "foreground":
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
            if self.target_region == "foreground":
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
            if self.target_region == "foreground":
                ctx.set_source_rgb(bg_r, bg_g, bg_b)
                ctx.paint()

            asset_contour = self.load_asset_vector_contour(self.texture_asset_image)
            if asset_contour is not None:
                ctx.set_source_rgb(tx_r, tx_g, tx_b)
                spacing = max(4, self.texture_line_spacing)
                stamp_sz = max(2.0, self.asset_shape_size)
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

    def generate_image(self, image_path):
        """process input image and render manipulated texture canvas."""
        outer_polygons, linestrings = self.extract_contours(image_path)
        canvas_w, canvas_h = self.canvas_size

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, canvas_w, canvas_h)
        ctx = cairo.Context(surface)
        ctx.set_antialias(cairo.ANTIALIAS_BEST)

        bg_r, bg_g, bg_b = self._normalize_color(self.background)
        ctx.set_source_rgb(bg_r, bg_g, bg_b)
        ctx.paint()

        if outer_polygons or linestrings:
            self._render_texture(ctx, outer_polygons)

            # Draw optional boundary outline stroke if width > 0
            if self.outline_width > 0:
                out_r, out_g, out_b = self._normalize_color(self.outline_color)
                ctx.set_source_rgb(out_r, out_g, out_b)
                ctx.set_line_width(self.outline_width)
                for ls in linestrings:
                    self._draw_path_to_cairo(ctx, ls)
                    ctx.stroke()

        return self._surface_to_pil(surface)


# ---------------------------------------------------------------------------
# generator config and entry point
# ---------------------------------------------------------------------------


@dataclass
class ManipulatedTexturesConfig(GeneratorConfig):
    """config for manipulated textures dataset."""

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
    target_region: list = field(
        default_factory=lambda: ["foreground"],
        metadata={
            "choices": ["foreground", "background"],
            "label": "target region for texture (foreground/background options)",
        },
    )
    texture_mode: list = field(
        default_factory=lambda: ["lines"],
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
            "label": "texture manipulation mode options",
        },
    )
    texture_color: list = field(
        default_factory=lambda: [255, 255, 255],
        metadata={"label": "texture color (RGB)"},
    )
    texture_line_spacing: list = field(
        default_factory=lambda: [10],
        metadata={"label": "texture pattern spacing options"},
    )
    texture_line_width: int = field(
        default=2,
        metadata={"min": 1, "max": 20, "step": 1, "label": "texture pattern line width"},
    )
    texture_angle: list = field(
        default_factory=lambda: [45.0],
        metadata={"label": "texture pattern angle options (degrees)"},
    )
    texture_asset_image: list = field(
        default_factory=lambda: [""],
        metadata={"label": "asset categories or image paths for texture shape pattern"},
    )
    asset_shape_size: float = field(
        default=12.0,
        metadata={
            "min": 2.0,
            "max": 50.0,
            "step": 1.0,
            "label": "size of tiled asset shape (px)",
        },
    )
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    output_folder: str = field(
        default="data/shape_and_object_recognition/manipulated_textures",
        metadata={"label": "output folder"},
    )


@register("manipulated_textures", "shape_recognition")
@generator(ManipulatedTexturesConfig)
def generate_all(config: ManipulatedTexturesConfig):
    """generate manipulated textures dataset across all parameter combinations."""
    output_folder = Path(config.output_folder)
    linedrawing_input_folder = Path(config.linedrawing_input_folder)

    all_categories = [p.stem for p in linedrawing_input_folder.glob("*") if p.is_dir()]
    for cat in all_categories:
        (output_folder / cat).mkdir(exist_ok=True, parents=True)

    image_files = sorted(linedrawing_input_folder.rglob("*.jpg")) + sorted(
        linedrawing_input_folder.rglob("*.png")
    )

    modes = _to_list(config.texture_mode)
    angles = [float(a) for a in _to_list(config.texture_angle)]
    spacings = [int(s) for s in _to_list(config.texture_line_spacing)]
    target_regions = _to_list(config.target_region)
    asset_images = _to_list(config.texture_asset_image)

    combinations = list(
        itertools.product(modes, angles, spacings, target_regions, asset_images)
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "Class",
                "TargetRegion",
                "TextureMode",
                "TextureAssetImage",
                "TextureAngle",
                "TextureLineSpacing",
                "BackgroundColor",
                "TextureColor",
                "IterNum",
            ]
        )

        n = 0
        for img_path in tqdm(image_files, desc="processing categories"):
            class_name = img_path.parent.stem
            image_name = img_path.stem

            for t_mode, t_angle, t_spacing, t_region, t_asset in combinations:
                ds = DrawManipulatedTexture(
                    background=config.background_color,
                    canvas_size=config.canvas_size,
                    antialiasing=config.antialiasing,
                    obj_longest_side=config.object_longest_side,
                    target_region=t_region,
                    texture_mode=t_mode,
                    texture_color=config.texture_color,
                    texture_line_spacing=t_spacing,
                    texture_line_width=config.texture_line_width,
                    texture_angle=t_angle,
                    texture_asset_image=t_asset,
                    asset_shape_size=config.asset_shape_size,
                    linedrawing_input_folder=config.linedrawing_input_folder,
                )

                img = ds.generate_image(img_path)

                # Construct filename reflecting the exact manipulations
                name_parts = [image_name, f"tex-{t_mode}"]
                if t_mode in ["lines", "asset_shape"]:
                    name_parts.append(f"ang{int(t_angle)}")
                if t_mode in ["lines", "grid", "dots", "checkerboard", "asset_shape"]:
                    name_parts.append(f"spc{t_spacing}")

                if t_region == "background":
                    name_parts.append("bg")
                else:
                    name_parts.append("fg")

                if t_asset:
                    name_parts.append(f"asset-{Path(t_asset).stem}")

                filename = "_".join(name_parts) + ".png"
                path = Path(class_name) / filename

                img.save(output_folder / path)
                writer.writerow(
                    [
                        path,
                        class_name,
                        t_region,
                        t_mode,
                        t_asset,
                        t_angle,
                        t_spacing,
                        ds.background,
                        config.texture_color,
                        n,
                    ]
                )
                n += 1

    return str(output_folder)
