"""nap vs mp 2d lines dataset generator."""

import csv
from dataclasses import dataclass, field
from pathlib import Path
import uuid

import cv2
import PIL.Image as Image
from tqdm.auto import tqdm

from mindset.drawing.base import (
    DrawStimuli,
    paste_linedrawing_onto_canvas,
    resize_image_keep_aspect_ratio,
)
from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils import apply_antialiasing


class DrawLinedrawings(DrawStimuli):
    """draw line drawings onto a canvas with resizing."""

    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def get_linedrawings(self, image_path):
        """load a grayscale image, resize, and paste onto canvas."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = resize_image_keep_aspect_ratio(img, self.obj_longest_side)
        canvas = paste_linedrawing_onto_canvas(
            Image.fromarray(img), self.create_canvas(), self.fill
        )
        return apply_antialiasing(canvas) if self.antialiasing else canvas


@dataclass
class NapVsMp2dConfig(GeneratorConfig):
    """config for nap vs mp 2d lines dataset."""

    object_longest_side: int = field(
        default=200,
        metadata={"min": 10, "max": 1000, "step": 10, "label": "object longest side"},
    )
    input_folder: str = field(
        default="mindset/assets/nap_vs_mp_2d/pngs", metadata={"label": "input folder"}
    )
    output_folder: str = field(
        default="data/low_mid_vision/nap_vs_mp_2d",
        metadata={"label": "output folder"},
    )
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})


@register("nap_vs_mp_2d", "low_mid_vision")
@generator(NapVsMp2dConfig)
def generate_all(config: NapVsMp2dConfig):
    """generate nap vs mp 2d lines dataset."""
    output_folder = Path(config.output_folder)
    input_folder = Path(config.input_folder)

    ref_folder = input_folder / "reference"
    mp_folder = input_folder / "MP"
    nap_folder = input_folder / "NAP"

    for cond in ["reference", "MP", "NAP"]:
        (output_folder / cond).mkdir(exist_ok=True, parents=True)

    ds = DrawLinedrawings(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        obj_longest_side=config.object_longest_side,
    )

    ref_files = sorted(
        [f for f in ref_folder.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]],
        key=lambda x: x.name,
    )

    def save_image(img, condition, sample_id, sample_name):
        unique_hex = uuid.uuid4().hex[:8]
        path = Path(condition) / f"{sample_id}_{sample_name}_{unique_hex}.png"
        img.save(output_folder / path)
        return path

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "SampleID",
                "SampleName",
                "ReferencePath",
                "MPPath",
                "NAPPath",
                "BackgroundColor",
            ]
        )

        for sample_id, ref_img_path in enumerate(tqdm(ref_files)):
            sample_name = ref_img_path.stem
            filename = ref_img_path.name

            mp_img_path = mp_folder / filename
            nap_img_path = nap_folder / filename

            if not mp_img_path.exists():
                raise FileNotFoundError(f"MP file not found for {filename}")
            if not nap_img_path.exists():
                raise FileNotFoundError(f"NAP file not found for {filename}")

            # Generate reference image
            ref_img = ds.get_linedrawings(ref_img_path)
            ref_path = save_image(ref_img, "reference", sample_id, sample_name)

            # Generate MP image
            mp_img = ds.get_linedrawings(mp_img_path)
            mp_path = save_image(mp_img, "MP", sample_id, sample_name)

            # Generate NAP image
            nap_img = ds.get_linedrawings(nap_img_path)
            nap_path = save_image(nap_img, "NAP", sample_id, sample_name)

            writer.writerow(
                [
                    sample_id,
                    sample_name,
                    ref_path,
                    mp_path,
                    nap_path,
                    ds.background,
                ]
            )

    return str(output_folder)
