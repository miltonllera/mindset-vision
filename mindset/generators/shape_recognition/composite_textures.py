import csv
import itertools
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

from mindset.generators._base import GeneratorConfig, generator, register
from mindset.generators.shape_recognition.manipulated_textures import  DrawManipulatedTexture
from mindset.utils import to_list


class DrawCompositeTexture:
    """Draws composite texture manipulations layering foreground and background textures."""

    def __init__(self, fg_params: dict, bg_params: dict):
        fg_kwargs = fg_params.copy()
        bg_kwargs = bg_params.copy()

        fg_kwargs["target_region"] = "foreground"
        bg_kwargs["target_region"] = "background"

        self.ds_fg = DrawManipulatedTexture(**fg_kwargs)
        self.ds_bg = DrawManipulatedTexture(**bg_kwargs)

    def generate_image(self, image_path: Path) -> Image.Image:
        """render foreground and background textures and composite them."""
        img_fg = self.ds_fg.generate_image(image_path)
        img_bg = self.ds_bg.generate_image(image_path)
        return Image.alpha_composite(img_bg.convert("RGBA"), img_fg.convert("RGBA"))


# ---------------------------------------------------------------------------
# generator config and entry point
# ---------------------------------------------------------------------------


@dataclass
class CompositeTexturesConfig(GeneratorConfig):
    """config for composite foreground and background textures dataset."""

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
    # Foreground Texture Controls
    fg_texture_mode: list = field(
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
            "label": "foreground texture mode options",
        },
    )
    fg_texture_color: list = field(
        default_factory=lambda: [255, 255, 255],
        metadata={"label": "foreground texture color (RGB)"},
    )
    fg_texture_spacing: list = field(
        default_factory=lambda: [10],
        metadata={"label": "foreground texture line spacing options"},
    )
    fg_texture_scale: list = field(
        default_factory=lambda: [2.0],
        metadata={"label": "foreground texture element scale options"},
    )
    fg_texture_angle: list = field(
        default_factory=lambda: [45.0],
        metadata={"label": "foreground texture angle options (degrees)"},
    )
    fg_texture_asset_image: list = field(
        default_factory=lambda: [""],
        metadata={"label": "foreground asset category or image path options"},
    )
    fg_texture_text: list = field(
        default_factory=lambda: ["A"],
        metadata={"label": "foreground text/letter options for text mode"},
    )
    fg_texture_font: str = field(
        default="Sans",
        metadata={"label": "foreground font family name"},
    )
    # Background Texture Controls
    bg_texture_mode: list = field(
        default_factory=lambda: ["dots"],
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
            "label": "background texture mode options",
        },
    )
    bg_texture_color: list = field(
        default_factory=lambda: [255, 255, 255],
        metadata={"label": "background texture color (RGB)"},
    )
    bg_texture_spacing: list = field(
        default_factory=lambda: [16],
        metadata={"label": "background texture line spacing options"},
    )
    bg_texture_scale: list = field(
        default_factory=lambda: [4.0],
        metadata={"label": "background texture element scale options"},
    )
    bg_texture_angle: list = field(
        default_factory=lambda: [0.0],
        metadata={"label": "background texture angle options (degrees)"},
    )
    bg_texture_asset_image: list = field(
        default_factory=lambda: [""],
        metadata={"label": "background asset category or image path options"},
    )
    bg_texture_text: list = field(
        default_factory=lambda: ["X"],
        metadata={"label": "background text/letter options for text mode"},
    )
    bg_texture_font: str = field(
        default="Sans",
        metadata={"label": "background font family name"},
    )
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    output_folder: str = field(
        default="data/shape_and_object_recognition/composite_textures",
        metadata={"label": "output folder"},
    )


@register("composite_textures", "shape_recognition")
@generator(CompositeTexturesConfig)
def generate_all(config: CompositeTexturesConfig):
    """generate composite textures dataset combining foreground and background texture options."""
    output_folder = Path(config.output_folder)
    linedrawing_input_folder = Path(config.linedrawing_input_folder)

    all_categories = [p.stem for p in linedrawing_input_folder.glob("*") if p.is_dir()]
    for cat in all_categories:
        (output_folder / cat).mkdir(exist_ok=True, parents=True)

    image_files = sorted(linedrawing_input_folder.rglob("*.jpg")) + sorted(
        linedrawing_input_folder.rglob("*.png")
    )

    # Prepare Foreground Combinations
    fg_modes = to_list(config.fg_texture_mode)
    fg_angles = [float(a) for a in to_list(config.fg_texture_angle)]
    fg_spacings = [int(s) for s in to_list(config.fg_texture_spacing)]
    fg_scales = [float(sc) for sc in to_list(config.fg_texture_scale)]
    fg_assets = to_list(config.fg_texture_asset_image)
    fg_texts = to_list(config.fg_texture_text)

    fg_combinations = list(
        itertools.product(
            fg_modes, fg_angles, fg_spacings, fg_scales, fg_assets, fg_texts
        )
    )

    # Prepare Background Combinations
    bg_modes = to_list(config.bg_texture_mode)
    bg_angles = [float(a) for a in to_list(config.bg_texture_angle)]
    bg_spacings = [int(s) for s in to_list(config.bg_texture_spacing)]
    bg_scales = [float(sc) for sc in to_list(config.bg_texture_scale)]
    bg_assets = to_list(config.bg_texture_asset_image)
    bg_texts = to_list(config.bg_texture_text)

    bg_combinations = list(
        itertools.product(
            bg_modes, bg_angles, bg_spacings, bg_scales, bg_assets, bg_texts
        )
    )

    all_composite_combinations = list(
        itertools.product(fg_combinations, bg_combinations)
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "IterNum",
                "Class",
                "FG_TextureMode",
                "FG_TextureScale",
                "FG_TextureSpacing",
                "FG_TextureAssetImage",
                "FG_TextureText",
                "FG_TextureAngle",
                "BG_TextureMode",
                "BG_TextureScale",
                "BG_TextureSpacing",
                "BG_TextureAssetImage",
                "BG_TextureText",
                "BG_TextureAngle",
                "BackgroundColor",
                "FG_TextureColor",
                "BG_TextureColor",
                "Path",
            ]
        )

        n = 0
        for img_path in tqdm(image_files, desc="processing categories"):
            class_name = img_path.parent.stem
            image_name = img_path.stem

            for fg_combo, bg_combo in all_composite_combinations:
                (
                    fg_mode,
                    fg_angle,
                    fg_spacing,
                    fg_scale,
                    fg_asset,
                    fg_text,
                ) = fg_combo
                (
                    bg_mode,
                    bg_angle,
                    bg_spacing,
                    bg_scale,
                    bg_asset,
                    bg_text,
                ) = bg_combo

                fg_params = {
                    "background": config.background_color,
                    "canvas_size": config.canvas_size,
                    "antialiasing": config.antialiasing,
                    "obj_longest_side": config.object_longest_side,
                    "texture_mode": fg_mode,
                    "texture_color": config.fg_texture_color,
                    "texture_spacing": fg_spacing,
                    "texture_scale": fg_scale,
                    "texture_angle": fg_angle,
                    "texture_asset_image": fg_asset,
                    "texture_text": fg_text,
                    "texture_font": config.fg_texture_font,
                    "linedrawing_input_folder": config.linedrawing_input_folder,
                }

                bg_params = {
                    "background": config.background_color,
                    "canvas_size": config.canvas_size,
                    "antialiasing": config.antialiasing,
                    "obj_longest_side": config.object_longest_side,
                    "texture_mode": bg_mode,
                    "texture_color": config.bg_texture_color,
                    "texture_spacing": bg_spacing,
                    "texture_scale": bg_scale,
                    "texture_angle": bg_angle,
                    "texture_asset_image": bg_asset,
                    "texture_text": bg_text,
                    "texture_font": config.bg_texture_font,
                    "linedrawing_input_folder": config.linedrawing_input_folder,
                }

                ds = DrawCompositeTexture(fg_params=fg_params, bg_params=bg_params)
                img = ds.generate_image(img_path)

                # Construct descriptive filename
                fg_fmt_sc = int(fg_scale) if fg_scale == int(fg_scale) else fg_scale
                bg_fmt_sc = int(bg_scale) if bg_scale == int(bg_scale) else bg_scale

                fg_tag = f"fg-{fg_mode}_ang{int(fg_angle)}_spc{fg_spacing}_sc{fg_fmt_sc}"
                if fg_mode in ["text", "letters"]:
                    fg_tag += f"_txt-{fg_text}"
                elif fg_asset:
                    fg_tag += f"_asset-{Path(fg_asset).stem}"

                bg_tag = f"bg-{bg_mode}_ang{int(bg_angle)}_spc{bg_spacing}_sc{bg_fmt_sc}"
                if bg_mode in ["text", "letters"]:
                    bg_tag += f"_txt-{bg_text}"
                elif bg_asset:
                    bg_tag += f"_asset-{Path(bg_asset).stem}"

                filename = f"{image_name}_{fg_tag}_{bg_tag}.png"
                path = Path(class_name) / filename

                img.save(output_folder / path)
                writer.writerow(
                    [
                        n,
                        class_name,
                        fg_mode,
                        fg_scale,
                        fg_spacing,
                        fg_asset,
                        fg_text,
                        fg_angle,
                        bg_mode,
                        bg_scale,
                        bg_spacing,
                        bg_asset,
                        bg_text,
                        bg_angle,
                        config.background_color,
                        config.fg_texture_color,
                        config.bg_texture_color,
                        path,
                    ]
                )
                n += 1

    return str(output_folder)
