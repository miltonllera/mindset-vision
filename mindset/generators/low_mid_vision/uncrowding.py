"""uncrowding dataset generator."""

import csv
import math
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from mindset.drawing.base import DrawStimuli
from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils import apply_antialiasing


# Shape identifier constants
SQUARE = 1
CIRCLE = 2
HEXAGON = 3  # Flat face pointing up

SHAPE_NAMES = {
    SQUARE: "square",
    CIRCLE: "circle",
    HEXAGON: "hexagon",
}


class DrawUncrowding(DrawStimuli):
    """draws uncrowding stimuli with verniers and shape flanker grids."""

    def __init__(self, bar_width=2, bar_height=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bar_width = bar_width
        self.bar_height = bar_height

    def draw_square(self, shape_size: int) -> Image.Image:
        """draw a square outline patch."""
        patch = Image.new("RGB", (shape_size, shape_size), self.background)
        draw = ImageDraw.Draw(patch)
        margin = max(1, self.bar_width)
        draw.rectangle(
            [margin, margin, shape_size - 1 - margin, shape_size - 1 - margin],
            outline=self.line_col,
            width=self.bar_width,
        )
        return patch

    def draw_circle(self, shape_size: int) -> Image.Image:
        """draw a circle outline patch."""
        patch = Image.new("RGB", (shape_size, shape_size), self.background)
        draw = ImageDraw.Draw(patch)
        margin = max(1, self.bar_width)
        draw.ellipse(
            [margin, margin, shape_size - 1 - margin, shape_size - 1 - margin],
            outline=self.line_col,
            width=self.bar_width,
        )
        return patch

    def draw_hexagon(self, shape_size: int) -> Image.Image:
        """draw a regular hexagon outline patch with a flat top face pointing up."""
        patch = Image.new("RGB", (shape_size, shape_size), self.background)
        draw = ImageDraw.Draw(patch)
        center_x = shape_size / 2.0
        center_y = shape_size / 2.0
        radius = (shape_size / 2.0) - max(1, self.bar_width)

        # Angles for a flat top face: k * pi / 3 for k = 0..5
        # k=1 (60 deg) and k=2 (120 deg) give top-right and top-left vertices with equal y coordinates.
        vertices = []
        for k in range(6):
            angle = k * math.pi / 3.0
            x = center_x + radius * math.cos(angle)
            y = center_y - radius * math.sin(angle)
            vertices.append((x, y))

        for i in range(6):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % 6]
            draw.line([p1, p2], fill=self.line_col, width=self.bar_width)

        return patch

    def draw_shape(self, shape_id: int, shape_size: int) -> Image.Image:
        """draw a flanker shape patch by ID."""
        if shape_id == SQUARE:
            return self.draw_square(shape_size)
        elif shape_id == CIRCLE:
            return self.draw_circle(shape_size)
        elif shape_id == HEXAGON:
            return self.draw_hexagon(shape_size)
        else:
            return Image.new("RGB", (shape_size, shape_size), self.background)

    def draw_vernier(
        self, vernier_type: int, offset_pixels: int
    ) -> Image.Image:
        """
        draw a Vernier stimulus patch.

        vernier_type = 0: top line shifted to the right.
        vernier_type = 1: top line shifted to the left.
        """
        bar_h = self.bar_height
        patch_w = self.bar_width * 2 + offset_pixels + 6
        patch_h = bar_h * 2 + 4
        patch = Image.new("RGB", (patch_w, patch_h), self.background)
        draw = ImageDraw.Draw(patch)

        center_x = patch_w // 2

        if vernier_type == 0:
            # Top line to right, bottom line to left
            x_top = center_x + math.ceil(offset_pixels / 2.0)
            x_bottom = center_x - math.floor(offset_pixels / 2.0)
        else:
            # Top line to left, bottom line to right
            x_top = center_x - math.floor(offset_pixels / 2.0)
            x_bottom = center_x + math.ceil(offset_pixels / 2.0)

        y_top_start = 2
        y_top_end = y_top_start + bar_h
        y_bottom_start = y_top_end
        y_bottom_end = y_bottom_start + bar_h

        draw.line(
            [(x_top, y_top_start), (x_top, y_top_end - 1)],
            fill=self.line_col,
            width=self.bar_width,
        )
        draw.line(
            [(x_bottom, y_bottom_start), (x_bottom, y_bottom_end - 1)],
            fill=self.line_col,
            width=self.bar_width,
        )

        return patch


def sample_grid_structure(max_cols: int = 7):
    """
    sample grid rows and column configurations according to rules:
    - Max 3 rows (1 or 3 rows, centered).
    - Center row has odd number of columns from odd numbers <= max_cols.
    - If 3 rows, top & bottom rows have odd number of columns <= center row columns.
    - Added top and bottom rows share the exact same structure (horizontal axis symmetry).
    - Max 2 distinct shapes in the image.
    - Each row can be uniform (all same shape) or alternating (between 2 shapes).

    Returns:
        grid_rows: List[List[int]] shape IDs for each row.
        pattern_str: String formatted as "row_len:shape_pattern;..."
    """
    all_shapes = [SQUARE, CIRCLE, HEXAGON]

    num_distinct = random.choice([1, 2])
    selected_shapes = random.sample(all_shapes, k=num_distinct)

    possible_center_cols = [c for c in range(1, max_cols + 1, 2)]
    if not possible_center_cols:
        possible_center_cols = [1]
    c_center = random.choice(possible_center_cols)

    def generate_row_pattern(length):
        if len(selected_shapes) == 1:
            return [selected_shapes[0]] * length
        else:
            is_alternating = random.choice([True, False])
            if is_alternating:
                s1, s2 = selected_shapes[0], selected_shapes[1]
                if random.choice([True, False]):
                    s1, s2 = s2, s1
                return [s1 if i % 2 == 0 else s2 for i in range(length)]
            else:
                s_uniform = random.choice(selected_shapes)
                return [s_uniform] * length

    p_center = generate_row_pattern(c_center)

    num_rows = random.choice([1, 3])

    if num_rows == 3:
        possible_flanker_cols = [c for c in range(1, max_cols + 1, 2) if c <= c_center]
        c_flanker = random.choice(possible_flanker_cols)
        p_flanker = generate_row_pattern(c_flanker)

        # Symmetry along horizontal axis: top row and bottom row are identical
        grid_rows = [p_flanker, p_center, p_flanker]
    else:
        grid_rows = [p_center]

    pattern_parts = []
    for r_pattern in grid_rows:
        shapes_str = ",".join(str(s) for s in r_pattern)
        pattern_parts.append(f"{len(r_pattern)}:{shapes_str}")

    pattern_str = ";".join(pattern_parts)
    return grid_rows, pattern_str


def generate_uncrowding_stimulus(
    drawer: DrawUncrowding,
    grid_rows: list,
    vernier_type: int,
    vernier_offset: int,
    vernier_in_out: str,
    shape_size: int,
    canvas_size: tuple[int, int],
) -> Image.Image:
    """generate a complete uncrowding image with flanker grid and Vernier."""
    canvas = drawer.create_canvas()

    num_rows = len(grid_rows)
    max_cols = max(len(r) for r in grid_rows)

    grid_width = max_cols * shape_size
    grid_height = num_rows * shape_size

    canvas_w, canvas_h = canvas_size
    grid_x0 = (canvas_w - grid_width) // 2
    grid_y0 = (canvas_h - grid_height) // 2

    for r_idx, row_pattern in enumerate(grid_rows):
        row_len = len(row_pattern)
        row_width = row_len * shape_size
        row_x0 = (canvas_w - row_width) // 2
        row_y0 = grid_y0 + r_idx * shape_size

        for c_idx, shape_id in enumerate(row_pattern):
            shape_patch = drawer.draw_shape(shape_id, shape_size)
            px = row_x0 + c_idx * shape_size
            py = row_y0
            canvas.paste(shape_patch, (px, py))

    vernier_patch = drawer.draw_vernier(vernier_type, vernier_offset)
    vw, vh = vernier_patch.size

    grid_bbox = (
        grid_x0,
        grid_y0,
        grid_x0 + grid_width,
        grid_y0 + grid_height,
    )

    if vernier_in_out == "inside":
        center_row_idx = num_rows // 2
        center_row_len = len(grid_rows[center_row_idx])
        center_col_idx = center_row_len // 2

        center_row_width = center_row_len * shape_size
        center_row_x0 = (canvas_w - center_row_width) // 2
        center_row_y0 = grid_y0 + center_row_idx * shape_size

        center_x = center_row_x0 + center_col_idx * shape_size + shape_size // 2
        center_y = center_row_y0 + shape_size // 2

        vx = center_x - vw // 2
        vy = center_y - vh // 2

        # Blend Vernier onto central flanker
        canvas.paste(vernier_patch, (vx, vy), mask=vernier_patch.convert("L"))
    else:
        # Outside condition: sample position outside grid bounding box with clear margin
        safety_margin = 10
        min_x, min_y = safety_margin, safety_margin
        max_x = canvas_w - vw - safety_margin
        max_y = canvas_h - vh - safety_margin

        # Expanded grid bounding box including safety margin
        gx1 = grid_x0 - safety_margin
        gy1 = grid_y0 - safety_margin
        gx2 = grid_x0 + grid_width + safety_margin
        gy2 = grid_y0 + grid_height + safety_margin

        valid_position = False
        attempts = 0

        while not valid_position and attempts < 300:
            attempts += 1
            cand_x = random.randint(min_x, max(min_x, max_x))
            cand_y = random.randint(min_y, max(min_y, max_y))

            vx1, vy1 = cand_x, cand_y
            vx2, vy2 = cand_x + vw, cand_y + vh

            # Check for overlap with expanded grid bounding box
            overlap = not (vx2 < gx1 or vx1 > gx2 or vy2 < gy1 or vy1 > gy2)
            if not overlap:
                valid_position = True
                vx, vy = cand_x, cand_y

        if not valid_position:
            # Fallback to top margin above grid
            vx = max(min_x, (canvas_w - vw) // 2)
            vy = max(min_y, gy1 - vh)

        canvas.paste(vernier_patch, (vx, vy), mask=vernier_patch.convert("L"))

    img = canvas.convert("RGBA")
    if drawer.antialiasing:
        img = apply_antialiasing(img)
    return img


@dataclass
class UncrowdingConfig(GeneratorConfig):
    """config for uncrowding dataset."""

    num_samples_vernier_inside: int = field(
        default=100,
        metadata={
            "min": 1,
            "max": 10000,
            "step": 10,
            "label": "vernier inside samples",
        },
    )
    num_samples_vernier_outside: int = field(
        default=100,
        metadata={
            "min": 1,
            "max": 10000,
            "step": 10,
            "label": "vernier outside samples",
        },
    )
    min_vernier_offset: int = field(
        default=1,
        metadata={"min": 1, "max": 20, "label": "min vernier offset (px)"},
    )
    max_vernier_offset: int = field(
        default=5,
        metadata={"min": 1, "max": 30, "label": "max vernier offset (px)"},
    )
    bar_width: int = field(
        default=2,
        metadata={"min": 1, "max": 10, "label": "vernier & stroke width"},
    )
    bar_height: int = field(
        default=5,
        metadata={"min": 2, "max": 30, "label": "vernier bar height (px)"},
    )
    max_cols: int = field(
        default=7,
        metadata={"min": 1, "max": 7, "step": 2, "label": "max grid columns"},
    )
    shape_size: int = field(
        default=32,
        metadata={"min": 32, "max": 128, "label": "flanker shape size (px)"},
    )
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    output_folder: str = field(
        default="data/low_mid_vision/un_crowding",
        metadata={"label": "output folder"},
    )


@register("uncrowding", "low_mid_vision")
@generator(UncrowdingConfig)
def generate_all(config: UncrowdingConfig):
    """generate uncrowding dataset."""
    output_folder = Path(config.output_folder)

    vernier_in_out = ["outside", "inside"]
    vernier_type = [0, 1]

    for v_mode in vernier_in_out:
        for v_type in vernier_type:
            (output_folder / v_mode / str(v_type)).mkdir(exist_ok=True, parents=True)

    drawer = DrawUncrowding(
        canvas_size=config.canvas_size,
        background=(0, 0, 0),
        line_col=(255, 255, 255),
        antialiasing=config.antialiasing,
        bar_width=config.bar_width,
        bar_height=config.bar_height,
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "VernierInOut",
                "VernierType",
                "VernierOffset",
                "GridPattern",
                "BackgroundColor",
                "ShapeSize",
                "IterNum",
            ]
        )

        for v_mode in tqdm(vernier_in_out, desc="Mode"):
            num_requested = (
                config.num_samples_vernier_outside
                if v_mode == "outside"
                else config.num_samples_vernier_inside
            )

            for v_type in vernier_type:
                for n in range(num_requested // len(vernier_type)):
                    grid_rows, pattern_str = sample_grid_structure(
                        max_cols=config.max_cols
                    )
                    vernier_offset = random.randint(
                        config.min_vernier_offset, config.max_vernier_offset
                    )

                    img = generate_uncrowding_stimulus(
                        drawer=drawer,
                        grid_rows=grid_rows,
                        vernier_type=v_type,
                        vernier_offset=vernier_offset,
                        vernier_in_out=v_mode,
                        shape_size=config.shape_size,
                        canvas_size=config.canvas_size,
                    )

                    unique_hex = uuid.uuid4().hex[:8]
                    file_name = f"{v_mode}_v{v_type}_offset{vernier_offset}_{n}_{unique_hex}.png"
                    rel_path = Path(v_mode) / str(v_type) / file_name

                    img.save(output_folder / rel_path)
                    writer.writerow(
                        [
                            rel_path,
                            v_mode,
                            v_type,
                            vernier_offset,
                            pattern_str,
                            drawer.background,
                            config.shape_size,
                            n,
                        ]
                    )

    return str(output_folder)
