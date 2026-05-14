"""ponzo illusion dataset generator."""

import csv
import math
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from mindset.drawing.base import DrawStimuli
from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils import apply_antialiasing


def compute_x(y, line_start, line_end):
    """compute x coordinate on a line at a given y."""
    x1, y1 = line_start
    x2, y2 = line_end
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    x = (y - b) / m
    return x


class DrawPonzo(DrawStimuli):
    """draws ponzo illusion stimuli."""

    def __init__(self, num_rail_lines=4, *args, **kwargs):
        """initialize with number of rail lines."""
        super().__init__(*args, **kwargs)
        self.num_rail_lines = num_rail_lines

    def get_random_start_end_ponts(self):
        """return random start and end points on the canvas."""
        start_point = (
            random.randint(0, self.canvas_size[0]),
            random.randint(0, self.canvas_size[1]),
        )
        end_point = (
            random.randint(0, self.canvas_size[0]),
            random.randint(0, self.canvas_size[1]),
        )
        return start_point, end_point

    def generate_illusory_images(self, same_length=False):
        """generate a ponzo illusion image with converging lines."""
        img = self.create_canvas()
        d = ImageDraw.Draw(img)
        margin = self.canvas_size[0] * 0.11

        vertical_distance_from_center = random.randint(
            int(self.canvas_size[0] * 0.11), int(self.canvas_size[0] * 0.33)
        )
        x = self.canvas_size[0] // 2 - vertical_distance_from_center
        length = self.canvas_size[0] - margin * 2
        slope = random.random() * 2 + 3.2
        x2 = x + length // 2 * math.cos(math.atan(-slope))
        y2 = self.canvas_size[1] // 2 + length // 2 * math.sin(math.atan(-slope))
        x1 = x - length // 2 * math.cos(math.atan(-slope))
        y1 = self.canvas_size[1] // 2 - length // 2 * math.sin(math.atan(-slope))
        d1_line_start = (x1, y1)
        d1_line_end = (x2, y2)
        d.line([d1_line_start, d1_line_end], fill="white", width=2)

        x = self.canvas_size[0] // 2 + vertical_distance_from_center
        x2 = x + length // 2 * math.cos(math.atan(slope))
        y2 = self.canvas_size[1] // 2 + length // 2 * math.sin(math.atan(slope))
        x1 = x - length // 2 * math.cos(math.atan(slope))
        y1 = self.canvas_size[1] // 2 - length // 2 * math.sin(math.atan(slope))
        d2_line_start = (x1, y1)
        d2_line_end = (x2, y2)

        d.line([d2_line_start, d2_line_end], fill="white", width=2)

        if self.num_rail_lines > 0:
            vertical_ranges = [
                int(
                    i * (self.canvas_size[1] - margin * 2) // (self.num_rail_lines)
                    + margin
                )
                for i in range(self.num_rail_lines + 1)
            ]
            vertical_line_position = [
                random.randint(vertical_ranges[i], vertical_ranges[i + 1] - 1)
                for i in range(len(vertical_ranges) - 1)
            ]
            for v in vertical_line_position:
                h_start = compute_x(v, d1_line_start, d1_line_end)
                h_end = compute_x(v, d2_line_start, d2_line_end)
                additional = (h_end - h_start) * 0.1
                d.line(
                    ((h_start - additional, v), (h_end + additional, v)),
                    fill="white",
                    width=2,
                )
        v_position_up = random.randint(int(margin), self.canvas_size[1] // 2)
        v_position_down = random.randint(
            self.canvas_size[1] // 2, int(self.canvas_size[1] - margin)
        )
        pos = np.array([v_position_down, v_position_up])
        red_length = random.randint(self.canvas_size[0] // 10, self.canvas_size[0] // 2)
        blue_length = (
            red_length
            if same_length
            else random.randint(self.canvas_size[0] // 10, self.canvas_size[0] // 2)
        )
        np.random.shuffle(pos)

        upper_line_color = "red" if pos[0] < pos[1] else "blue"
        d.line(
            (
                (self.canvas_size[0] // 2 - red_length // 2, pos[0]),
                (self.canvas_size[0] // 2 + red_length // 2, pos[0]),
            ),
            fill="red",
            width=2,
        )

        d.line(
            (
                (self.canvas_size[0] // 2 - blue_length // 2, pos[1]),
                (self.canvas_size[0] // 2 + blue_length // 2, pos[1]),
            ),
            fill="blue",
            width=2,
        )

        max_length = self.canvas_size[0] // 2 - self.canvas_size[0] // 10
        label = red_length - blue_length
        norm_label = label / max_length

        return img, red_length, blue_length, upper_line_color

    def generate_rnd_lines_images(
        self, colored_line_always_horizontal=False, antialias=True
    ):
        """generate an image with random lines and colored target lines."""
        img = self.create_canvas()

        d = ImageDraw.Draw(img)
        for i in range(self.num_rail_lines + 2):
            start_point, end_point = self.get_random_start_end_ponts()
            d.line([start_point, end_point], fill="white", width=2)

        if colored_line_always_horizontal:
            margin = self.canvas_size[0] * 0.11
            v_position_red = random.randint(
                int(margin), int(self.canvas_size[1] - margin)
            )
            v_position_blue = random.randint(
                int(margin), int(self.canvas_size[1] - margin)
            )
            red_length = random.randint(
                self.canvas_size[0] // 10, self.canvas_size[0] // 2
            )
            blue_length = random.randint(
                self.canvas_size[0] // 10, self.canvas_size[0] // 2
            )
            red_sp = (self.canvas_size[0] // 2 - red_length // 2, v_position_red)
            red_ep = (self.canvas_size[0] // 2 + red_length // 2, v_position_red)
            blue_sp = (self.canvas_size[0] // 2 - blue_length // 2, v_position_blue)
            blue_ep = (self.canvas_size[0] // 2 + blue_length // 2, v_position_blue)

        else:
            red_sp, red_ep = self.get_random_start_end_ponts()
            blue_sp, blue_ep = self.get_random_start_end_ponts()

        d.line([red_sp, red_ep], fill="red", width=2)
        red_length = np.linalg.norm(np.array(red_ep) - np.array(red_sp))
        d.line([blue_sp, blue_ep], fill="blue", width=2)
        blue_length = np.linalg.norm(np.array(blue_ep) - np.array(blue_sp))

        max_length = self.canvas_size[0] // 2 - self.canvas_size[0] // 10
        if antialias:
            img = img.resize(tuple(np.array(self.canvas_size) * 2)).resize(
                self.canvas_size, resample=Image.Resampling.LANCZOS
            )
        upper_line_color = ""
        return (
            apply_antialiasing(img) if self.antialiasing else img,
            red_length,
            blue_length,
            upper_line_color,
        )


@dataclass
class PonzoConfig(GeneratorConfig):
    """config for ponzo illusion dataset."""

    num_samples_scrambled: int = field(
        default=5000,
        metadata={"min": 1, "max": 50000, "step": 10, "label": "scrambled samples"},
    )
    num_samples_illusory: int = field(
        default=500,
        metadata={"min": 1, "max": 5000, "step": 10, "label": "illusory samples"},
    )
    num_rail_lines: int = field(
        default=5,
        metadata={"min": 0, "max": 20, "step": 1, "label": "number of rail lines"},
    )
    rnd_target_lines: bool = field(
        default=False, metadata={"label": "random target lines"}
    )
    output_folder: str = field(
        default="data/visual_illusions/ponzo", metadata={"label": "output folder"}
    )


@register("ponzo", "visual_illusions")
@generator(PonzoConfig)
def generate_all(config: PonzoConfig):
    """generate ponzo illusion dataset."""
    output_folder = Path(config.output_folder)
    for d in ["scrambled_lines", "ponzo_same_length", "ponzo_diff_length"]:
        (output_folder / d).mkdir(parents=True, exist_ok=True)

    ds = DrawPonzo(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        num_rail_lines=config.num_rail_lines,
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "Type",
                "BackgroundColor",
                "RedLength",
                "BlueLength",
                "UpperLineColor",
                "NumRailLines",
                "IterNum",
            ]
        )

        for i in tqdm(range(config.num_samples_scrambled)):
            img, red_l, blue_l, _ = ds.generate_rnd_lines_images(
                colored_line_always_horizontal=not config.rnd_target_lines,
                antialias=config.antialiasing,
            )
            unique_hex = uuid.uuid4().hex[:8]
            path = Path("scrambled_lines") / f"{unique_hex}.png"
            img.save(output_folder / path)
            writer.writerow(
                [
                    path,
                    "scrambled_lines",
                    ds.background,
                    red_l,
                    blue_l,
                    "",
                    config.num_rail_lines,
                    i,
                ]
            )

        for i in tqdm(range(config.num_samples_illusory)):
            for c in ["ponzo_same_length", "ponzo_diff_length"]:
                img, red_l, blue_l, upper_line_color = ds.generate_illusory_images(
                    same_length=(c == "ponzo_same_length")
                )
                unique_hex = uuid.uuid4().hex[:8]
                path = Path(c) / f"{upper_line_color}_{unique_hex}.png"
                img.save(output_folder / path)
                writer.writerow(
                    [
                        path,
                        c,
                        ds.background,
                        red_l,
                        blue_l,
                        upper_line_color,
                        config.num_rail_lines,
                        i,
                    ]
                )

    return str(output_folder)
