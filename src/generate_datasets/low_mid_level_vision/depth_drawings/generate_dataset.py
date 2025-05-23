import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import sty
import PIL.Image as Image
import toml
import inspect

from src.utils.drawing_utils import (
    DrawStimuli,
    paste_linedrawing_onto_canvas,
    resize_image_keep_aspect_ratio,
)
from src.utils.misc import (
    add_general_args,
    delete_and_recreate_path,
    apply_antialiasing,
)

from src.utils.misc import DEFAULTS as BASE_DEFAULTS
import os

DEFAULTS = BASE_DEFAULTS.copy()
from tqdm import tqdm

from PIL import ImageOps


class DrawLinedrawings(DrawStimuli):
    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def get_linedrawings(self, image_path):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = resize_image_keep_aspect_ratio(img, self.obj_longest_side)

        canvas = paste_linedrawing_onto_canvas(
            Image.fromarray(img), self.create_canvas(), self.fill
        )

        return apply_antialiasing(canvas) if self.antialiasing else canvas


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))

DEFAULTS.update(
    {
        "object_longest_side": 200,
        "input_folder": "assets/enns_rensink_1991/pngs",
        "output_folder": f"data/{category_folder}/{name_dataset}",
        "antialiasing": False,
    }
)


def generate_all(
    object_longest_side=DEFAULTS["object_longest_side"],
    input_folder=DEFAULTS["input_folder"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    behaviour_if_present=DEFAULTS["behaviour_if_present"],
):
    loc = locals()
    args = {i: loc[i] for i in inspect.getfullargspec(generate_all)[0]}
    config = {f"{category_folder}/{name_dataset}": args}

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    if output_folder.exists() and behaviour_if_present == "skip":
        print(sty.fg.yellow + f"Dataset already exists. Skipping" + sty.rs.fg)
        return str(output_folder)

    delete_and_recreate_path(output_folder)
    toml.dump(config, open(str(output_folder / "config.toml"), "w"))

    all_categories = [i.stem for i in input_folder.glob("*")]

    [(output_folder / cat).mkdir(exist_ok=True, parents=True) for cat in all_categories]

    ds = DrawLinedrawings(
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        obj_longest_side=object_longest_side,
    )
    jpg_files = list(input_folder.rglob("*.jpg"))
    png_files = list(input_folder.rglob("*.png"))

    image_files = jpg_files + png_files
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Class", "BackgroundColor", "Id", "IterNum"])
        for n, img_path in enumerate(tqdm(image_files)):
            class_name = img_path.parent.stem
            img = ds.get_linedrawings(img_path)
            path = Path(class_name) / f"{img_path.stem}.png"
            img.save(output_folder / path)
            writer.writerow([path, class_name, ds.background, img_path.stem, n])
    return str(output_folder)


if __name__ == "__main__":
    description = "we recreated through vector graphics the stimuli appearing in Enns & Rensink (1991), using white strokes on a uniform background. \nREF:Enns, James T., and Ronald A. Rensink. 'Preattentive Recovery of Three-Dimensional Orientation from Line Drawing'. Psychological Review 98, no. 3 (1991): 335-51. https://doi.org/10.1037/0033-295X.98.3.335."

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])
    parser.set_defaults(antialiasing=DEFAULTS["antialiasing"])
    parser.add_argument(
        "--object_longest_side",
        "-objlside",
        default=DEFAULTS["object_longest_side"],
        type=int,
        help="Specify the value in pixels to which the longest side of the line drawings will be resized (keeping the aspect ratio), before pasting the image into a canvas",
    )
    parser.add_argument(
        "--input_folder",
        "-fld",
        dest="input_folder",
        help="A folder containing the Depth Perception stimuli from Enns and Rensink 1991.",
        default=DEFAULTS["input_folder"],
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
