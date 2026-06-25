#!/usr/bin/env python3
"""
Procedural generator for Enns & Rensink (1991) depth drawings stimuli.
Generates matched triplets of:
1. basis: 3D-interpretable cube or truncated pyramid in oblique projection.
2. variation1: Isolated 2D corner Y-junction.
3. variation2: 2D corner Y-junction enclosed in a square frame.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

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

def draw_stimulus(condition, s, angle_deg, depth_factor, w, back_scale=1.0, canvas_size=800):
    """
    Generates a single stimulus image for a given condition.
    Uses a high-resolution canvas internally (supersampling) to achieve high-quality antialiasing.
    """
    # Scale factor for supersampling
    scale = 2
    C = canvas_size * scale
    s_s = s * scale
    w_s = w * scale

    theta_rad = np.deg2rad(angle_deg)
    L = depth_factor * s_s
    dx = L * np.cos(theta_rad)
    dy = L * np.sin(theta_rad)

    # 3D vertices of front face (Z = 0) and back face (Z = 1)
    # Front vertices (X, Y in {0, 1})
    # Back vertices scaled by back_scale relative to center (0.5, 0.5)
    X_front = np.array([0, 1, 1, 0])
    Y_front = np.array([0, 0, 1, 1])

    X_back = 0.5 + (X_front - 0.5) * back_scale
    Y_back = 0.5 + (Y_front - 0.5) * back_scale

    # Project to 2D relative coordinates (before bounding box shifting)
    # x = X * s + Z * dx
    # y = Y * s + Z * dy
    pts_front = []
    for i in range(4):
        pts_front.append((X_front[i] * s_s, Y_front[i] * s_s))

    pts_back = []
    for i in range(4):
        pts_back.append((X_back[i] * s_s + dx, Y_back[i] * s_s + dy))

    all_pts = pts_front + pts_back
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    W_box = x_max - x_min
    H_box = y_max - y_min

    # Center bounding box on canvas
    ox = (C - W_box) / 2
    oy = (C - H_box) / 2

    # Shift all vertices
    V = []
    for p in pts_front:
        V.append((p[0] - x_min + ox, p[1] - y_min + oy))
    V_p = []
    for p in pts_back:
        V_p.append((p[0] - x_min + ox, p[1] - y_min + oy))

    V0, V1, V2, V3 = V[0], V[1], V[2], V[3]
    V0_p, V1_p, V2_p, V3_p = V_p[0], V_p[1], V_p[2], V_p[3]

    # Initialize canvas
    img = Image.new("RGB", (C, C), "black")
    draw = ImageDraw.Draw(img)
    fill = (255, 255, 255)

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
        # Draw front square outline
        draw.line([V0, V1, V2, V3, V0], fill=fill, width=w_s)
        # Draw visible oblique edges
        for p1, p2 in oblique_edges:
            draw.line([p1, p2], fill=fill, width=w_s)
        # Draw visible back edges
        for p1, p2 in back_edges:
            draw.line([p1, p2], fill=fill, width=w_s)

    elif condition == "variation1":
        # Draw the 3 lines of the isolated Y-junction
        # Line 1: Horizontal
        p_horiz = (V_jc[0] + dx_sign * s_s, V_jc[1])
        draw.line([V_jc, p_horiz], fill=fill, width=w_s)
        # Line 2: Vertical
        p_vert = (V_jc[0], V_jc[1] + dy_sign * s_s)
        draw.line([V_jc, p_vert], fill=fill, width=w_s)
        # Line 3: Oblique (connecting front to back vertex)
        draw.line([V_jc, V_jc_p], fill=fill, width=w_s)

    elif condition == "variation2":
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
        # Find ray direction from V_jc to V_jc_p
        v_diag_x = V_jc_p[0] - V_jc[0]
        v_diag_y = V_jc_p[1] - V_jc[1]
        diag_angle_rad = np.arctan2(v_diag_y, v_diag_x)

        cx_local = V_jc[0] - ox
        cy_local = V_jc[1] - oy
        ix_local, iy_local = find_ray_box_intersection(cx_local, cy_local, diag_angle_rad, W_box, H_box)
        p_diag_ext = (ox + ix_local, oy + iy_local)
        draw.line([V_jc, p_diag_ext], fill=fill, width=w_s)

    # Downsample using Lanczos interpolation to antialias
    img_resized = img.resize((canvas_size, canvas_size), Image.Resampling.LANCZOS)
    return img_resized

def main():
    assets_dir = Path(__file__).resolve().parent
    pngs_dir = assets_dir / "pngs"

    # 4 Quadrants: Down-Right (45), Down-Left (135), Up-Left (225), Up-Right (315)
    # We vary each base angle by +/- 15 degrees
    quadrants = {
        "down_right": [30, 45, 60],
        "down_left": [120, 135, 150],
        "up_left": [210, 225, 240],
        "up_right": [300, 315, 330]
    }

    depth_factors = [0.4, 0.6, 0.8, 1.0]
    stroke_widths = [6, 10, 14]
    # back_scale controls shape: 1.0 = Cube, 0.6 = Truncated Pyramid
    back_scales = [1.0, 0.6]

    square_side = 320
    canvas_size = 800

    # Ensure output directories exist and are clean
    conditions = ["basis", "variation1", "variation2"]
    for cond in conditions:
        cond_dir = pngs_dir / cond
        cond_dir.mkdir(parents=True, exist_ok=True)
        # Clear existing png files to avoid mixing old and new
        for f in cond_dir.glob("*.png"):
            f.unlink()

    print("Generating stimuli dataset (Cubes & Truncated Pyramids)...")
    count = 0

    # Generate combinatorial variations
    for quad_name, angles in quadrants.items():
        for angle in angles:
            for df in depth_factors:
                for w in stroke_widths:
                    for bs in back_scales:
                        count += 1
                        # Unique matching ID/filename stem
                        shape_type = "cube" if bs == 1.0 else "pyr"
                        stim_id = f"stim_{shape_type}_{count:03d}"

                        # Draw all 3 conditions for these exact parameters
                        for cond in conditions:
                            img = draw_stimulus(
                                condition=cond,
                                s=square_side,
                                angle_deg=angle,
                                depth_factor=df,
                                w=w,
                                back_scale=bs,
                                canvas_size=canvas_size
                            )
                            output_path = pngs_dir / cond / f"{stim_id}.png"
                            img.save(output_path)

    print(f"Successfully generated {count} matched triplets ({count * 3} total images) in {pngs_dir}")

if __name__ == "__main__":
    main()
