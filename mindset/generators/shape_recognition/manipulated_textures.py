import csv
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path

import cairo
from shapely.geometry import LineString
from tqdm.auto import tqdm

from mindset.generators._base import GeneratorConfig, generator, register
from mindset.shapes.manipulation import BaseDrawManipulatedObject
from mindset.utils import to_list


class DrawManipulatedTexture(BaseDrawManipulatedObject):
    """
    Draws object texture manipulations (foreground or background target region) using
    PyCairo and Shapely.
    """

    def __init__(
        self,
        target_region="foreground",
        texture_mode="lines",
        texture_color=(255, 255, 255),
        texture_spacing=10,
        texture_scale=2.0,
        texture_angle=45.0,
        texture_asset_image="",
        texture_text="A",
        texture_font="Sans",
        outline_color=(255, 255, 255),
        outline_width=0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.target_region = target_region
        self.texture_mode = texture_mode
        self.texture_color = texture_color
        self.texture_spacing = texture_spacing
        self.texture_scale = float(texture_scale)
        self.texture_angle = texture_angle
        self.texture_asset_image = texture_asset_image
        self.texture_text = str(texture_text)
        self.texture_font = str(texture_font)
        self.outline_color = outline_color
        self.outline_width = outline_width

    def _render_texture(self, ctx, outer_polygons):
        """render vector texture inside polygon mask or inverted background mask."""
        if self.texture_mode == "none":
            return

        canvas_w, canvas_h = self.canvas_size

        if self.target_region == "background":
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
            ctx.set_line_width(max(0.5, self.texture_scale))

            rad = math.radians(self.texture_angle)
            diag = math.hypot(canvas_w, canvas_h)
            cx, cy = canvas_w / 2.0, canvas_h / 2.0
            spacing = max(1, self.texture_spacing)
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

        else:
            # Grid, Dots, Checkerboard, Asset Shape, Text/Letters
            if self.target_region == "foreground":
                ctx.set_source_rgb(bg_r, bg_g, bg_b)
                ctx.paint()

            ctx.set_source_rgb(tx_r, tx_g, tx_b)

            # Apply pattern rotation around canvas center if angle != 0
            if self.texture_angle != 0:
                cx, cy = canvas_w / 2.0, canvas_h / 2.0
                ctx.translate(cx, cy)
                ctx.rotate(math.radians(self.texture_angle))
                ctx.translate(-cx, -cy)

            # Expand grid bounds to cover canvas when rotated
            diag = int(math.hypot(canvas_w, canvas_h))
            margin = (diag - min(canvas_w, canvas_h)) // 2 + 50
            min_pos = -margin
            max_w = canvas_w + margin
            max_h = canvas_h + margin

            if self.texture_mode == "grid":
                ctx.set_line_width(max(0.5, self.texture_scale))
                spacing = max(1, self.texture_spacing)

                for x in range(min_pos, max_w + spacing, spacing):
                    ctx.move_to(x, min_pos)
                    ctx.line_to(x, max_h)
                    ctx.stroke()

                for y in range(min_pos, max_h + spacing, spacing):
                    ctx.move_to(min_pos, y)
                    ctx.line_to(max_w, y)
                    ctx.stroke()

            elif self.texture_mode == "dots":
                spacing = max(2, self.texture_spacing)
                radius = max(0.5, self.texture_scale / 2.0)

                for x in range(min_pos, max_w + spacing, spacing):
                    for y in range(min_pos, max_h + spacing, spacing):
                        ctx.arc(x, y, radius, 0, 2 * math.pi)
                        ctx.fill()

            elif self.texture_mode == "checkerboard":
                spacing = max(2, self.texture_spacing)

                for x in range(min_pos, max_w, spacing):
                    for y in range(min_pos, max_h, spacing):
                        if ((x // spacing) + (y // spacing)) % 2 == 0:
                            ctx.rectangle(x, y, spacing, spacing)
                            ctx.fill()

            elif self.texture_mode == "asset_shape":
                asset_contour = self.load_asset_vector_contour(self.texture_asset_image)
                if asset_contour is not None:
                    spacing = max(4, self.texture_spacing)
                    stamp_sz = max(1.0, self.texture_scale)

                    for x in range(min_pos, max_w + spacing, spacing):
                        for y in range(min_pos, max_h + spacing, spacing):
                            ctx.save()
                            ctx.translate(x, y)
                            ctx.scale(stamp_sz, stamp_sz)

                            ctx.move_to(asset_contour[0, 0], asset_contour[0, 1])
                            for pt in asset_contour[1:]:
                                ctx.line_to(pt[0], pt[1])
                            ctx.close_path()
                            ctx.fill()
                            ctx.restore()

            elif self.texture_mode == "text" or self.texture_mode == "letters":
                ctx.select_font_face(
                    self.texture_font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD
                )
                stamp_sz = max(4.0, self.texture_scale)
                ctx.set_font_size(stamp_sz)
                spacing = max(4, self.texture_spacing)

                for x in range(min_pos, max_w + spacing, spacing):
                    for y in range(min_pos, max_h + spacing, spacing):
                        ctx.save()
                        ctx.translate(x, y)
                        ctx.move_to(-stamp_sz / 4.0, stamp_sz / 3.0)
                        ctx.text_path(self.texture_text)
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
                "text",
                "letters",
                "none",
            ],
            "label": "texture manipulation mode options",
        },
    )
    texture_color: list = field(
        default_factory=lambda: [255, 255, 255],
        metadata={"label": "texture color (RGB)"},
    )
    texture_spacing: list = field(
        default_factory=lambda: [10],
        metadata={"label": "texture pattern spacing options"},
    )
    texture_scale: list = field(
        default_factory=lambda: [12.0],
        metadata={"label": "texture element scale / font size options"},
    )
    texture_angle: list = field(
        default_factory=lambda: [0.0],
        metadata={"label": "texture pattern angle options (degrees)"},
    )
    texture_asset_image: list = field(
        default_factory=lambda: [""],
        metadata={"label": "asset categories or image paths for texture shape pattern"},
    )
    texture_text: list = field(
        default_factory=lambda: ["A"],
        metadata={"label": "text/letter string options for text mode"},
    )
    texture_font: str = field(
        default="Sans",
        metadata={"label": "font family name for text mode"},
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

    modes = to_list(config.texture_mode)
    angles = [float(a) for a in to_list(config.texture_angle)]
    spacings = [int(s) for s in to_list(config.texture_spacing)]
    scales = [float(sc) for sc in to_list(config.texture_scale)]
    target_regions = to_list(config.target_region)
    asset_images = to_list(config.texture_asset_image)
    texts = to_list(config.texture_text)

    combinations = list(
        itertools.product(modes, angles, spacings, scales, target_regions, asset_images, texts)
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "IterNum",
                "Class",
                "TargetRegion",
                "TextureMode",
                "TextureScale",
                "TextureLineSpacing",
                "TextureAssetImage",
                "TextureText",
                "TextureAngle",
                "TextureColor",
                "BackgroundColor",
                "Path",
            ]
        )

        n = 0
        for img_path in tqdm(image_files, desc="processing categories"):
            class_name = img_path.parent.stem
            image_name = img_path.stem

            for t_mode, t_angle, t_spacing, t_scale, t_region, t_asset, t_text in combinations:
                ds = DrawManipulatedTexture(
                    background=config.background_color,
                    canvas_size=config.canvas_size,
                    antialiasing=config.antialiasing,
                    obj_longest_side=config.object_longest_side,
                    target_region=t_region,
                    texture_mode=t_mode,
                    texture_color=config.texture_color,
                    texture_spacing=t_spacing,
                    texture_scale=t_scale,
                    texture_angle=t_angle,
                    texture_asset_image=t_asset,
                    texture_text=t_text,
                    texture_font=config.texture_font,
                    linedrawing_input_folder=config.linedrawing_input_folder,
                )

                img = ds.generate_image(img_path)

                # Construct filename reflecting the exact manipulations
                fmt_sc = int(t_scale) if t_scale == int(t_scale) else t_scale
                name_parts = [image_name, f"tex-{t_mode}"]

                if t_mode in [
                    "lines", "grid", "dots", "checkerboard", "asset_shape", "text", "letters"
                ]:
                    name_parts.append(f"ang{int(t_angle)}")
                    name_parts.append(f"spc{t_spacing}")
                if t_mode in ["lines", "grid", "dots", "asset_shape", "text", "letters"]:
                    name_parts.append(f"sc{fmt_sc}")

                if t_region == "background":
                    name_parts.append("bg")
                else:
                    name_parts.append("fg")

                if t_mode in ["text", "letters"]:
                    name_parts.append(f"txt-{t_text}")
                elif t_asset:
                    name_parts.append(f"asset-{Path(t_asset).stem}")

                filename = "_".join(name_parts) + ".png"
                path = Path(class_name) / filename

                img.save(output_folder / path)
                writer.writerow(
                    [
                        n,
                        class_name,
                        t_region,
                        t_mode,
                        t_scale,
                        t_spacing,
                        t_asset,
                        t_text,
                        t_angle,
                        ds.background,
                        config.texture_color,
                        path,
                    ]
                )
                n += 1

    return str(output_folder)
