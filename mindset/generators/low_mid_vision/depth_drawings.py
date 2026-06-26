"""depth drawings dataset generator."""

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
from tqdm.auto import tqdm

from mindset.drawing.base import DrawStimuli
from mindset.generators._base import GeneratorConfig, generator, register


def find_ray_box_intersection(cx, cy, theta_rad, W_box, H_box):
    """
    Finds the intersection of a ray from (cx, cy) at angle theta_rad
    with the boundaries of the box [0, W_box] x [0, H_box].
    """
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    candidates = []

    if abs(cos_t) > 1e-6:
        # Intersection with x = 0
        t1 = -cx / cos_t
        if t1 > 0:
            candidates.append(t1)
        # Intersection with x = W_box
        t2 = (W_box - cx) / cos_t
        if t2 > 0:
            candidates.append(t2)

    if abs(sin_t) > 1e-6:
        # Intersection with y = 0
        t3 = -cy / sin_t
        if t3 > 0:
            candidates.append(t3)
        # Intersection with y = H_box
        t4 = (H_box - cy) / sin_t
        if t4 > 0:
            candidates.append(t4)

    if not candidates:
        return cx, cy

    t_min = min(candidates)
    return cx + t_min * cos_t, cy + t_min * sin_t


class DrawLinedrawings(DrawStimuli):
    """draw line drawings onto a canvas using 3D projection."""

    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def draw_stimulus(self, condition, angle_deg, depth_factor, back_scale=1.0):
        """
        Generates a single stimulus image for a given condition.
        Centering and scaling are automatically calculated using the bounding box of the projected shape.
        """
        # Scale factor for supersampling (if antialiasing is enabled)
        scale = 2 if self.antialiasing else 1
        C_w = self.canvas_size[0] * scale
        C_h = self.canvas_size[1] * scale

        # Calculate scaled dimensions based on target obj_longest_side
        s_longest = self.obj_longest_side * scale
        w_s = self.line_args["width"] * scale

        theta_rad = np.deg2rad(angle_deg)
        dx_unscaled = depth_factor * np.cos(theta_rad)
        dy_unscaled = depth_factor * np.sin(theta_rad)

        # 3D vertices of front face (Z = 0) and back face (Z = 1)
        # Front vertices (X, Y in {0, 1})
        # Back vertices scaled by back_scale relative to center (0.5, 0.5)
        X_front = np.array([0, 1, 1, 0])
        Y_front = np.array([0, 0, 1, 1])

        X_back = 0.5 + (X_front - 0.5) * back_scale
        Y_back = 0.5 + (Y_front - 0.5) * back_scale

        # Project to 2D relative coordinates (before bounding box shifting)
        pts_front_unscaled = []
        for i in range(4):
            pts_front_unscaled.append((X_front[i], Y_front[i]))

        pts_back_unscaled = []
        for i in range(4):
            pts_back_unscaled.append((X_back[i] + dx_unscaled, Y_back[i] + dy_unscaled))

        all_pts_unscaled = pts_front_unscaled + pts_back_unscaled
        xs_u = [p[0] for p in all_pts_unscaled]
        ys_u = [p[1] for p in all_pts_unscaled]

        x_min_u, x_max_u = min(xs_u), max(xs_u)
        y_min_u, y_max_u = min(ys_u), max(ys_u)

        W_unscaled = x_max_u - x_min_u
        H_unscaled = y_max_u - y_min_u
        M_unscaled = max(W_unscaled, H_unscaled)

        # Scale factor to make longest side equal to target
        s_s = s_longest / M_unscaled

        # Now compute scaled coordinates
        dx = depth_factor * s_s * np.cos(theta_rad)
        dy = depth_factor * s_s * np.sin(theta_rad)

        pts_front = [(X_front[i] * s_s, Y_front[i] * s_s) for i in range(4)]
        pts_back = [(X_back[i] * s_s + dx, Y_back[i] * s_s + dy) for i in range(4)]

        all_pts = pts_front + pts_back
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        W_box = x_max - x_min
        H_box = y_max - y_min

        # Center bounding box on canvas
        ox = (C_w - W_box) / 2
        oy = (C_h - H_box) / 2

        # Shift all vertices
        V = [(p[0] - x_min + ox, p[1] - y_min + oy) for p in pts_front]
        V_p = [(p[0] - x_min + ox, p[1] - y_min + oy) for p in pts_back]

        V0, V1, V2, V3 = V[0], V[1], V[2], V[3]
        V0_p, V1_p, V2_p, V3_p = V_p[0], V_p[1], V_p[2], V_p[3]

        # Create canvas
        img = self.create_canvas(size=(C_w, C_h))
        draw = ImageDraw.Draw(img)
        fill = self.fill

        # Determine visibility and junction corner based on quadrant of (dx, dy)
        if dx >= 0 and dy >= 0:
            # Down-Right projection
            oblique_edges = [(V1, V1_p), (V2, V2_p), (V3, V3_p)]
            back_edges = [(V1_p, V2_p), (V2_p, V3_p)]
            V_jc = V2
            V_jc_p = V2_p
            dx_sign = -1  # horizontal goes left
            dy_sign = -1  # vertical goes up
        elif dx < 0 and dy < 0:
            # Up-Left projection
            oblique_edges = [(V0, V0_p), (V1, V1_p), (V3, V3_p)]
            back_edges = [(V0_p, V1_p), (V0_p, V3_p)]
            V_jc = V0
            V_jc_p = V0_p
            dx_sign = 1   # horizontal goes right
            dy_sign = 1   # vertical goes down
        elif dx >= 0 and dy < 0:
            # Up-Right projection
            oblique_edges = [(V0, V0_p), (V1, V1_p), (V2, V2_p)]
            back_edges = [(V0_p, V1_p), (V1_p, V2_p)]
            V_jc = V1
            V_jc_p = V1_p
            dx_sign = -1  # horizontal goes left
            dy_sign = 1   # vertical goes down
        else:
            # Down-Left projection
            oblique_edges = [(V0, V0_p), (V2, V2_p), (V3, V3_p)]
            back_edges = [(V0_p, V3_p), (V3_p, V2_p)]
            V_jc = V3
            V_jc_p = V3_p
            dx_sign = 1   # horizontal goes right
            dy_sign = -1  # vertical goes up

        if condition == "basis":
            # Draw the 3 lines of the isolated Y-junction
            # Line 1: Horizontal
            p_horiz = (V_jc[0] + dx_sign * s_s, V_jc[1])
            draw.line([V_jc, p_horiz], fill=fill, width=w_s)
            # Line 2: Vertical
            p_vert = (V_jc[0], V_jc[1] + dy_sign * s_s)
            draw.line([V_jc, p_vert], fill=fill, width=w_s)
            # Line 3: Oblique (connecting front to back vertex)
            draw.line([V_jc, V_jc_p], fill=fill, width=w_s)

        elif condition == "rectangle-hull":
            # Draw the outer bounding box (centered on canvas)
            box_coords = [
                (ox, oy),
                (ox + W_box, oy),
                (ox + W_box, oy + H_box),
                (ox, oy + H_box),
                (ox, oy)
            ]
            draw.line(box_coords, fill=fill, width=w_s)

            # Extended Horizontal Line
            x_horiz_ext = ox if dx_sign < 0 else ox + W_box
            draw.line([V_jc, (x_horiz_ext, V_jc[1])], fill=fill, width=w_s)

            # Extended Vertical Line
            y_vert_ext = oy if dy_sign < 0 else oy + H_box
            draw.line([V_jc, (V_jc[0], y_vert_ext)], fill=fill, width=w_s)

            # Extended Oblique Line
            v_diag_x = V_jc_p[0] - V_jc[0]
            v_diag_y = V_jc_p[1] - V_jc[1]
            diag_angle_rad = np.arctan2(v_diag_y, v_diag_x)

            cx_local = V_jc[0] - ox
            cy_local = V_jc[1] - oy
            ix_local, iy_local = find_ray_box_intersection(cx_local, cy_local, diag_angle_rad, W_box, H_box)
            p_diag_ext = (ox + ix_local, oy + iy_local)
            draw.line([V_jc, p_diag_ext], fill=fill, width=w_s)

        elif condition == "3dshape":
            # Draw front square outline
            draw.line([V0, V1, V2, V3, V0], fill=fill, width=w_s)
            # Draw visible oblique edges
            for p1, p2 in oblique_edges:
                draw.line([p1, p2], fill=fill, width=w_s)
            # Draw visible back edges
            for p1, p2 in back_edges:
                draw.line([p1, p2], fill=fill, width=w_s)

        # Downsample using Lanczos interpolation to antialias
        if self.antialiasing:
            img = img.resize(self.canvas_size, Image.Resampling.LANCZOS)
        return img


@dataclass
class DepthDrawingsConfig(GeneratorConfig):
    """config for depth drawings dataset."""

    num_samples: int = field(
        default=50,
        metadata={"min": 1, "max": 20000, "step": 1, "label": "number of samples"},
    )
    object_longest_side: int = field(
        default=200,
        metadata={"min": 10, "max": 1000, "step": 10, "label": "object longest side"},
    )
    stroke_width: int = field(
        default=3,
        metadata={"min": 1, "max": 20, "step": 1, "label": "stroke width (px)"},
    )
    output_folder: str = field(
        default="data/low_mid_vision/depth_drawings",
        metadata={"label": "output folder"},
    )


@register("depth_drawings", "low_mid_vision")
@generator(DepthDrawingsConfig)
def generate_all(config: DepthDrawingsConfig):
    """generate depth drawings dataset."""
    output_folder = Path(config.output_folder)

    for cond in [
        "basis", "basis_rotated",
        "rectangle-hull", "rectangle-hull_rotated",
        "3dshape", "3dshape_rotated",
    ]:
        (output_folder / cond).mkdir(exist_ok=True, parents=True)

    ds = DrawLinedrawings(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        width=config.stroke_width,
        obj_longest_side=config.object_longest_side,
    )

    # 4 Quadrants: Down-Right (45), Down-Left (135), Up-Left (225), Up-Right (315)
    quadrants = {
        "down_right": [30, 45, 60],
        "down_left": [120, 135, 150],
        "up_left": [210, 225, 240],
        "up_right": [300, 315, 330]
    }
    quadrant_names = list(quadrants.keys())

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow([
            "SampleID", "Shape", "Angle", "DepthFactor", "BackScale",
            "BasisPath", "BasisRotatedPath", "RectangleHullPath", "RectangleHullRotatedPath",
            "3DShapePath", "3DShapeRotatedPath", "BackgroundColor"
        ])

        for n in tqdm(range(config.num_samples)):
            # Randomly select shape type (1.0 = Cube, 0.6 = Truncated Pyramid)
            shape_type = random.choice(["cube", "truncated_pyramid"])
            back_scale = 1.0 if shape_type == "cube" else 0.6

            # Select random quadrant and base angle with jitter
            quad_name = random.choice(quadrant_names)
            base_angle = random.choice(quadrants[quad_name])
            angle = base_angle + random.uniform(-5, 5)

            # Sample random depth factor
            depth_factor = random.uniform(0.4, 1.0)

            stim_id = f"{shape_type}_{n:05d}"

            # Generate basis feature variants
            basis_img = ds.draw_stimulus("basis", angle, depth_factor, back_scale)
            basis_rotated_img = basis_img.rotate(180.0)

            basis_path = Path("basis") / f"{stim_id}.png"
            basis_img.save(output_folder / basis_path)

            basis_rotated_path = Path("basis_rotated") / f"{stim_id}.png"
            basis_rotated_img.save(output_folder / basis_rotated_path)

            # Generate rectangle hull variants
            v1_img = ds.draw_stimulus("rectangle-hull", angle, depth_factor, back_scale)
            v1_rotated_img = v1_img.rotate(180.0)

            v1_path = Path("rectangle-hull") / f"{stim_id}.png"
            v1_img.save(output_folder / v1_path)

            v1_rotated_path = Path("rectangle-hull_rotated") / f"{stim_id}.png"
            v1_rotated_img.save(output_folder / v1_rotated_path)

            # Generate 3D shape variants
            v2_img = ds.draw_stimulus("3dshape", angle, depth_factor, back_scale)
            v2_rotated_img = v2_img.rotate(180.0)

            v2_path = Path("3dshape") / f"{stim_id}.png"
            v2_img.save(output_folder / v2_path)

            v2_rotated_path = Path("3dshape_rotated") / f"{stim_id}.png"
            v2_rotated_img.save(output_folder / v2_rotated_path)

            # Write annotation row
            writer.writerow([
                n,
                shape_type,
                round(angle, 2),
                round(depth_factor, 2),
                back_scale,
                str(basis_path),
                str(basis_rotated_path),
                str(v1_path),
                str(v1_rotated_path),
                str(v2_path),
                str(v2_rotated_path),
                ds.background
            ])

    return str(output_folder)

