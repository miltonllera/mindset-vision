"""relational vs coordinate dataset generator."""

import csv
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import PIL.Image as Image
from PIL import ImageDraw
from tqdm.auto import tqdm

from mindset.drawing.base import DrawStimuli, paste_linedrawing_onto_canvas
from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils import apply_antialiasing


@dataclass
class LineNode:
    id: int
    depth: int
    parent_id: int | None
    orientation: str  # 'V' or 'H'

    # Branching point coordinates
    branch_x: float
    branch_y: float

    # Ratio along parent line (if not root)
    branch_ratio: float | None = None

    # Extensions in negative/positive directions
    # Negative perpendicular direction is left (for H) or up (for V)
    # Positive perpendicular direction is right (for H) or down (for V)
    L_neg: float = 0.0
    L_pos: float = 0.0

    # Whether this line is the designated cue line
    is_cue_line: bool = False
    # Which end has the cue ('neg' or 'pos')
    cue_end: str | None = None

    # True if L_neg is the long extension, False if L_pos is the long extension
    neg_is_long: bool = True

    def get_segment(self):
        if self.orientation == 'V':
            return (
                (self.branch_x, self.branch_y - self.L_neg),
                (self.branch_x, self.branch_y + self.L_pos)
            )
        else:
            return (
                (self.branch_x - self.L_neg, self.branch_y),
                (self.branch_x + self.L_pos, self.branch_y)
            )


def get_max_extension(x0, y0, direction, other_segments, margin=30):
    max_L = float('inf')
    for seg in other_segments:
        (x1, y1), (x2, y2) = seg
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        if direction == 'left':
            # Ray: y = y0, x decreases
            if y_min - margin <= y0 <= y_max + margin:
                if x1 == x2: # Vertical segment
                    if x_min < x0:
                        max_L = min(max_L, x0 - x_min - margin)
                else: # Horizontal segment
                    if abs(y1 - y0) < margin:
                        if x_max < x0:
                            max_L = min(max_L, x0 - x_max - margin)
        elif direction == 'right':
            # Ray: y = y0, x increases
            if y_min - margin <= y0 <= y_max + margin:
                if x1 == x2: # Vertical segment
                    if x_max > x0:
                        max_L = min(max_L, x_max - x0 - margin)
                else: # Horizontal segment
                    if abs(y1 - y0) < margin:
                        if x_min > x0:
                            max_L = min(max_L, x_min - x0 - margin)
        elif direction == 'up':
            # Ray: x = x0, y decreases
            if x_min - margin <= x0 <= x_max + margin:
                if y1 == y2: # Horizontal segment
                    if y_min < y0:
                        max_L = min(max_L, y0 - y_min - margin)
                else: # Vertical segment
                    if abs(x1 - x0) < margin:
                        if y_max < y0:
                            max_L = min(max_L, y0 - y_max - margin)
        elif direction == 'down':
            # Ray: x = x0, y increases
            if x_min - margin <= x0 <= x_max + margin:
                if y1 == y2: # Horizontal segment
                    if y_max > y0:
                        max_L = min(max_L, y_max - y0 - margin)
                else: # Vertical segment
                    if abs(x1 - x0) < margin:
                        if y_min > y0:
                            max_L = min(max_L, y_min - y0 - margin)
    return max(0.0, max_L)


def make_stimulus_triplet(root_orientation, margin=30):
    while True:
        # Generate basis tree
        # 1. Root Line
        if root_orientation == 'V':
            x_root = random.uniform(400, 600)
            y_start, y_end = 100, 900
            root = LineNode(
                id=0,
                depth=0,
                parent_id=None,
                orientation='V',
                branch_x=x_root,
                branch_y=500,
                L_neg=400, # extends up by 400
                L_pos=400, # extends down by 400
            )
        else:
            y_root = random.uniform(400, 600)
            x_start, x_end = 100, 900
            root = LineNode(
                id=0,
                depth=0,
                parent_id=None,
                orientation='H',
                branch_x=500,
                branch_y=y_root,
                L_neg=400, # extends left by 400
                L_pos=400, # extends right by 400
            )

        # 2. Level 1 Lines (branching factor 2)
        branch_pos1 = random.uniform(250, 450)
        branch_pos2 = random.uniform(550, 750)

        l2_parent_idx = random.choice([1, 2])

        l1_nodes = []
        for idx, br_pos in enumerate([branch_pos1, branch_pos2], start=1):
            neg_is_long = random.choice([True, False])
            if neg_is_long:
                L_neg_nom = random.uniform(500, 650)
                L_pos_nom = random.uniform(60, 100)
            else:
                L_neg_nom = random.uniform(60, 100)
                L_pos_nom = random.uniform(500, 650)

            if root_orientation == 'V':
                L_neg = min(L_neg_nom, x_root - 50)
                L_pos = min(L_pos_nom, 950 - x_root)
                node = LineNode(
                    id=idx,
                    depth=1,
                    parent_id=0,
                    orientation='H',
                    branch_x=x_root,
                    branch_y=br_pos,
                    branch_ratio=(br_pos - y_start) / (y_end - y_start),
                    L_neg=L_neg,
                    L_pos=L_pos,
                    neg_is_long=neg_is_long,
                )
            else:
                L_neg = min(L_neg_nom, y_root - 50)
                L_pos = min(L_pos_nom, 950 - y_root)
                node = LineNode(
                    id=idx,
                    depth=1,
                    parent_id=0,
                    orientation='V',
                    branch_x=br_pos,
                    branch_y=y_root,
                    branch_ratio=(br_pos - x_start) / (x_end - x_start),
                    L_neg=L_neg,
                    L_pos=L_pos,
                    neg_is_long=neg_is_long,
                )
            l1_nodes.append(node)

        # 3. Level 2 Line (branching from parent_node)
        parent_node = l1_nodes[l2_parent_idx - 1]
        other_l1_node = l1_nodes[2 - l2_parent_idx]

        parent_min = parent_node.branch_x - parent_node.L_neg if parent_node.orientation == 'H' \
            else parent_node.branch_y - parent_node.L_neg
        parent_max = parent_node.branch_x + parent_node.L_pos if parent_node.orientation == 'H' \
            else parent_node.branch_y + parent_node.L_pos

        if parent_node.neg_is_long:
            br_val = random.uniform(
                parent_min + 0.25 * parent_node.L_neg, parent_min + 0.75 * parent_node.L_neg
            )
        else:
            br_val = random.uniform(
                parent_max - 0.75 * parent_node.L_pos, parent_max - 0.25 * parent_node.L_pos
            )

        l2_neg_is_long = random.choice([True, False])
        if l2_neg_is_long:
            L_neg_nom = random.uniform(200, 300)
            L_pos_nom = random.uniform(40, 80)
        else:
            L_neg_nom = random.uniform(40, 80)
            L_pos_nom = random.uniform(200, 300)

        root_seg = root.get_segment()
        other_l1_seg = other_l1_node.get_segment()
        existing_segs = [root_seg, other_l1_seg]

        if root_orientation == 'V':
            L_neg_safe = get_max_extension(br_val, parent_node.branch_y, 'up', existing_segs, margin)
            L_pos_safe = get_max_extension(br_val, parent_node.branch_y, 'down', existing_segs, margin)

            L_neg = min(L_neg_nom, parent_node.branch_y - 50, L_neg_safe)
            L_pos = min(L_pos_nom, 950 - parent_node.branch_y, L_pos_safe)

            if L_neg < 35 or L_pos < 35:
                continue
            if L_pos > parent_node.branch_y - 50 or L_pos > L_neg_safe:
                continue
            if L_neg > 950 - parent_node.branch_y or L_neg > L_pos_safe:
                continue

            l2_node = LineNode(
                id=3,
                depth=2,
                parent_id=parent_node.id,
                orientation='V',
                branch_x=br_val,
                branch_y=parent_node.branch_y,
                branch_ratio=(br_val - parent_min) / (parent_max - parent_min),
                L_neg=L_neg,
                L_pos=L_pos,
                neg_is_long=l2_neg_is_long,
                is_cue_line=True,
                cue_end='neg',
            )
        else:
            L_neg_safe = get_max_extension(parent_node.branch_x, br_val, 'left', existing_segs, margin)
            L_pos_safe = get_max_extension(parent_node.branch_x, br_val, 'right', existing_segs, margin)

            L_neg = min(L_neg_nom, parent_node.branch_x - 50, L_neg_safe)
            L_pos = min(L_pos_nom, 950 - parent_node.branch_x, L_pos_safe)

            if L_neg < 35 or L_pos < 35:
                continue
            if L_pos > parent_node.branch_x - 50 or L_pos > L_neg_safe:
                continue
            if L_neg > 950 - parent_node.branch_x or L_neg > L_pos_safe:
                continue

            l2_node = LineNode(
                id=3,
                depth=2,
                parent_id=parent_node.id,
                orientation='H',
                branch_x=parent_node.branch_x,
                branch_y=br_val,
                branch_ratio=(br_val - parent_min) / (parent_max - parent_min),
                L_neg=L_neg,
                L_pos=L_pos,
                neg_is_long=l2_neg_is_long,
                is_cue_line=True,
                cue_end='neg',
            )

        # 4. Generate Coordinate Changed position (parent level 1 line slides)
        if root_orientation == 'V':
            y_branch_new = parent_node.branch_y + (l2_node.L_neg - l2_node.L_pos)

            # Check safety of shifted Level 1 line
            if not (150 <= y_branch_new <= 850):
                continue
            if abs(y_branch_new - other_l1_node.branch_y) < 100:
                continue

            parent_coord_changed = LineNode(
                id=parent_node.id,
                depth=parent_node.depth,
                parent_id=parent_node.parent_id,
                orientation=parent_node.orientation,
                branch_x=parent_node.branch_x,
                branch_y=y_branch_new,
                branch_ratio=(y_branch_new - 100) / 800,  # root goes 100 to 900
                L_neg=parent_node.L_neg,
                L_pos=parent_node.L_pos,
                neg_is_long=parent_node.neg_is_long,
            )

            l2_coord_changed = LineNode(
                id=l2_node.id,
                depth=l2_node.depth,
                parent_id=l2_node.parent_id,
                orientation=l2_node.orientation,
                branch_x=l2_node.branch_x,
                branch_y=y_branch_new,
                branch_ratio=l2_node.branch_ratio,
                L_neg=l2_node.L_neg,
                L_pos=l2_node.L_pos,
                neg_is_long=l2_node.neg_is_long,
                is_cue_line=True,
                cue_end=l2_node.cue_end,
            )
        else:
            x_branch_new = parent_node.branch_x + (l2_node.L_neg - l2_node.L_pos)

            # Check safety of shifted Level 1 line
            if not (150 <= x_branch_new <= 850):
                continue
            if abs(x_branch_new - other_l1_node.branch_x) < 100:
                continue

            parent_coord_changed = LineNode(
                id=parent_node.id,
                depth=parent_node.depth,
                parent_id=parent_node.parent_id,
                orientation=parent_node.orientation,
                branch_x=x_branch_new,
                branch_y=parent_node.branch_y,
                branch_ratio=(x_branch_new - 100) / 800,  # root goes 100 to 900
                L_neg=parent_node.L_neg,
                L_pos=parent_node.L_pos,
                neg_is_long=parent_node.neg_is_long,
            )

            l2_coord_changed = LineNode(
                id=l2_node.id,
                depth=l2_node.depth,
                parent_id=l2_node.parent_id,
                orientation=l2_node.orientation,
                branch_x=x_branch_new,
                branch_y=l2_node.branch_y,
                branch_ratio=l2_node.branch_ratio,
                L_neg=l2_node.L_neg,
                L_pos=l2_node.L_pos,
                neg_is_long=l2_node.neg_is_long,
                is_cue_line=True,
                cue_end=l2_node.cue_end,
            )

        break

    # Generate relationally flipped node
    l2_flipped = LineNode(
        id=l2_node.id,
        depth=l2_node.depth,
        parent_id=l2_node.parent_id,
        orientation=l2_node.orientation,
        branch_x=l2_node.branch_x,
        branch_y=l2_node.branch_y,
        branch_ratio=l2_node.branch_ratio,
        L_neg=l2_node.L_pos,
        L_pos=l2_node.L_neg,
        neg_is_long=not l2_node.neg_is_long,
        is_cue_line=True,
        cue_end=l2_node.cue_end,
    )

    # Pack l1 nodes for coordinate changed tree
    l1_nodes_coord = []
    for node in l1_nodes:
        if node.id == parent_node.id:
            l1_nodes_coord.append(parent_coord_changed)
        else:
            l1_nodes_coord.append(node)

    return (root, l1_nodes, l2_node), l2_flipped, (root, l1_nodes_coord, l2_coord_changed)


class DrawRelationalVsCoordinate(DrawStimuli):
    """draws programmatic relational vs coordinate stimuli."""

    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def draw_rounded_line(self, draw, p1, p2, width, fill=255):
        draw.line((p1, p2), fill=fill, width=width)
        r = width // 2
        draw.ellipse((p1[0] - r, p1[1] - r, p1[0] + r, p1[1] + r), fill=fill)
        draw.ellipse((p2[0] - r, p2[1] - r, p2[0] + r, p2[1] + r), fill=fill)

    def draw_tree_on_virtual_canvas(self, lines, line_width=20, bar_len=40, has_cue_bar=False):
        # Create a virtual canvas of 1000x1000 with black background (0)
        img = Image.new("L", (1000, 1000), 0)
        draw = ImageDraw.Draw(img)

        # Draw each line segment
        for line in lines:
            seg = line.get_segment()
            self.draw_rounded_line(draw, seg[0], seg[1], line_width)

            # Draw bars if applicable
            # Root line has bars at both ends
            if line.parent_id is None:
                # Perpendicular bars centered at ends
                if line.orientation == 'V':
                    ((x, y_start), (_, y_end)) = seg
                    self.draw_rounded_line(
                        draw, (x - bar_len, y_start), (x + bar_len, y_start), line_width
                    )
                    self.draw_rounded_line(
                        draw, (x - bar_len, y_end), (x + bar_len, y_end), line_width
                    )
                else:
                    ((x_start, y), (x_end, _)) = seg
                    self.draw_rounded_line(
                        draw, (x_start, y - bar_len), (x_start, y + bar_len), line_width
                    )
                    self.draw_rounded_line(
                        draw, (x_end, y - bar_len), (x_end, y + bar_len), line_width
                    )

            # Cue line has a bar at the designated end only if has_cue_bar is True
            if line.is_cue_line and has_cue_bar:
                if line.cue_end == 'neg':
                    xe, ye = seg[0]
                else:
                    xe, ye = seg[1]

                if line.orientation == 'V':
                    self.draw_rounded_line(draw, (xe - bar_len, ye), (xe + bar_len, ye), line_width)
                else:
                    self.draw_rounded_line(draw, (xe, ye - bar_len), (xe, ye + bar_len), line_width)

        return img

    def process_virtual_image(self, virtual_img):
        # 1. Get bounding box of white stroke
        bbox = virtual_img.getbbox()
        if bbox is None:
            bbox = (100, 100, 900, 900)

        left, upper, right, lower = bbox
        pad = 15  # a slightly wider padding to completely clear rounded caps
        left = max(0, left - pad)
        upper = max(0, upper - pad)
        right = min(virtual_img.width, right + pad)
        lower = min(virtual_img.height, lower + pad)

        cropped = virtual_img.crop((left, upper, right, lower))

        # 2. Resize keeping aspect ratio
        w, h = cropped.size
        if w > h:
            new_w = self.obj_longest_side
            new_h = int(round(h * self.obj_longest_side / w))
        else:
            new_h = self.obj_longest_side
            new_w = int(round(w * self.obj_longest_side / h))

        new_w = max(1, new_w)
        new_h = max(1, new_h)

        resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 3. Paste onto target canvas
        target_canvas = self.create_canvas()
        canvas = paste_linedrawing_onto_canvas(resized, target_canvas, self.fill)

        if self.antialiasing:
            canvas = apply_antialiasing(canvas)

        return canvas


@dataclass
class RelationalVsCoordinateConfig(GeneratorConfig):
    """config for relational vs coordinate dataset."""

    object_longest_side: int = field(
        default=200,
        metadata={"min": 10, "max": 1000, "step": 10, "label": "object longest side"},
    )
    num_samples: int = field(
        default=100,
        metadata={"min": 1, "max": 1000, "step": 1, "label": "number of samples"},
    )
    has_cue_bar: bool = field(
        default=False,
        metadata={"label": "has cue bar (forms a T at leaf end)"},
    )
    output_folder: str = field(
        default="data/low_mid_vision/rel_vs_coord_change",
        metadata={"label": "output folder"},
    )
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})


@register("relational_vs_coordinate", "low_mid_vision")
@generator(RelationalVsCoordinateConfig)
def generate_all(config: RelationalVsCoordinateConfig):
    """generate relational vs coordinate dataset."""
    output_folder = Path(config.output_folder)

    all_categories = ["basis", "coordinate_change", "relation_change"]
    for cat in all_categories:
        (output_folder / cat).mkdir(exist_ok=True, parents=True)

    ds = DrawRelationalVsCoordinate(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        obj_longest_side=config.object_longest_side,
    )

    # Programmatic line drawing parameters on virtual canvas (1000x1000)
    line_width = 20
    bar_len = 40
    margin = 35

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "SampleID",
                "RootOrientation",
                "BasisPath",
                "CoordinateChangePath",
                "RelationChangePath",
                "BasisBranchRatio",
                "CoordinateChangeRatio",
                "L2NegIsLong",
                "BackgroundColor",
            ]
        )

        for n in tqdm(range(config.num_samples)):
            # Randomly select orientation for this sample's root line
            root_orientation = random.choice(['V', 'H'])

            # Generate the shapes
            basis_tree, l2_flipped, coord_tree = make_stimulus_triplet(
                root_orientation=root_orientation,
                margin=margin
            )

            root, l1_nodes, l2_node = basis_tree
            root_c, l1_nodes_coord, l2_coord = coord_tree
            unique_hex = uuid.uuid4().hex[:8]

            # Save basis image
            basis_img = ds.draw_tree_on_virtual_canvas(
                [root, *l1_nodes, l2_node],
                line_width=line_width,
                bar_len=bar_len,
                has_cue_bar=config.has_cue_bar
            )
            basis_canvas = ds.process_virtual_image(basis_img)
            basis_path = Path("basis") / f"{n}_{unique_hex}.png"
            basis_canvas.save(output_folder / basis_path)

            # Save coordinate change image
            coord_img = ds.draw_tree_on_virtual_canvas(
                [root_c, *l1_nodes_coord, l2_coord],
                line_width=line_width,
                bar_len=bar_len,
                has_cue_bar=config.has_cue_bar
            )
            coord_canvas = ds.process_virtual_image(coord_img)
            coord_path = Path("coordinate_change") / f"{n}_{unique_hex}.png"
            coord_canvas.save(output_folder / coord_path)

            # Save relation change image
            rel_img = ds.draw_tree_on_virtual_canvas(
                [root, *l1_nodes, l2_flipped],
                line_width=line_width,
                bar_len=bar_len,
                has_cue_bar=config.has_cue_bar
            )
            rel_canvas = ds.process_virtual_image(rel_img)
            rel_path = Path("relation_change") / f"{n}_{unique_hex}.png"
            rel_canvas.save(output_folder / rel_path)

            # Record row in annotation file
            # In the coordinate change tree, the branch ratio of the Level 1 line shifts
            # but Level 2 branches at the same relative position along it.
            # We record the Level 1 branch ratios here to capture the slider coordinate change.
            parent_node = l1_nodes[l2_node.parent_id - 1]
            parent_coord = l1_nodes_coord[l2_node.parent_id - 1]

            writer.writerow(
                [
                    n,
                    root_orientation,
                    basis_path,
                    coord_path,
                    rel_path,
                    parent_node.branch_ratio,
                    parent_coord.branch_ratio,
                    l2_node.neg_is_long,
                    ds.background,
                ]
            )

    return str(output_folder)
