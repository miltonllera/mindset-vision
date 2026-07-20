"""nap vs mp 3d geons dataset generator using SDFs and PyRender."""

import os
import csv
from sys import exception
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyrender
import trimesh
import PIL.Image as Image
from skimage.measure import marching_cubes
from tqdm.auto import tqdm

from mindset.generators._base import GeneratorConfig, generator, register


@dataclass
class NapVsMp3dConfig(GeneratorConfig):
    """config for nap vs mp 3d geons dataset."""

    num_samples_per_dimension: int = field(
        default=10,
        metadata={"min": 1, "max": 1000, "step": 1, "label": "number of samples per dimension"},
    )

    # Override default background color from GeneratorConfig to medium gray
    background_color: list = field(
        default_factory=lambda: [0, 0, 0],
        metadata={"label": "background color (RGB)"}
    )

    # Shape base color
    shape_color: list = field(
        default_factory=lambda: [64, 64, 64],
        metadata={"label": "shape color (RGB)"}
    )

    # Mesh extraction resolution
    mesh_resolution: int = field(
        default=200,
        metadata={
            "min": 50, "max": 512, "step": 10, "label": "voxel grid resolution for marching cubes"
        }
    )

    # Shape dimensions to manipulate (represented as a comma-separated string for CLI parsing)
    dimensions: str = field(
        default="axis_curvature,taper,side_curvature,cross_section",
        metadata={"label": "comma-separated list of dimensions to manipulate"}
    )

    # Camera sampling configuration
    random_camera: bool = field(
        default=True,
        metadata={"label": "whether to randomize the camera angle per triplet"}
    )
    min_camera_distance: float = field(
        default=3.5, metadata={"label": "minimum camera distance"}
    )
    max_camera_distance: float = field(
        default=4.5, metadata={"label": "maximum camera distance"}
    )
    min_camera_elevation: float = field(
        default=15.0, metadata={"label": "minimum camera elevation (degrees)"}
    )
    max_camera_elevation: float = field(
        default=35.0, metadata={"label": "maximum camera elevation (degrees)"}
    )
    min_camera_azimuth: float = field(
        default=-15.0, metadata={"label": "minimum camera azimuth (degrees)"}
    )
    max_camera_azimuth: float = field(
        default=15.0, metadata={"label": "maximum camera azimuth (degrees)"}
    )

    # Triplet parameter ranges for sampled metric changes
    min_curvature: float = field(default=0.3, metadata={"label": "minimum curvature value"})
    max_curvature: float = field(default=0.6, metadata={"label": "maximum curvature value"})

    min_taper: float = field(default=0.35, metadata={"label": "minimum taper value"})
    max_taper: float = field(default=0.7, metadata={"label": "maximum taper value"})

    min_side_curvature: float = field(
        default=0.25, metadata={"label": "minimum side curvature value"}
    )
    max_side_curvature: float = field(
        default=0.5, metadata={"label": "maximum side curvature value"}
    )

    min_aspect_ratio: float = field(
        default=1.25, metadata={"label": "minimum aspect ratio value"}
    )
    max_aspect_ratio: float = field(default=1.5, metadata={"label": "maximum aspect ratio value"})

    output_folder: str = field(
        default="data/low_mid_vision/nap_vs_mp_3d_geons",
        metadata={"label": "output folder"},
    )


def unbend(p, curvature):
    """Unbend coordinate points along the Z-axis in the XZ plane."""
    if abs(curvature) < 1e-6:
        return p.copy()

    R = 1.0 / curvature
    x = p[..., 0]
    y = p[..., 1]
    z = p[..., 2]

    sign_R = np.sign(R)
    x_c = x + R
    dist = np.hypot(x_c, z)

    x_unbent = sign_R * (dist - abs(R))
    theta = np.arctan2(z * sign_R, x_c * sign_R)
    z_unbent = R * theta

    return np.stack([x_unbent, y, z_unbent], axis=-1)


def sdf_box(p, half_extents):
    """SDF for a 3D box."""
    q = np.abs(p) - half_extents
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def sdf_cylinder_core(p, r, h):
    """SDF for a 3D cylinder centered at origin."""
    d_h = np.hypot(p[..., 0], p[..., 1]) - r
    d_v = np.abs(p[..., 2]) - h
    outside = np.linalg.norm(np.maximum(np.stack([d_h, d_v], axis=-1), 0.0), axis=-1)
    inside = np.minimum(np.maximum(d_h, d_v), 0.0)
    return outside + inside


def sdf_shape(p, shape_type, half_extents, curvature, taper, side_curvature, aspect_ratio=1.0):
    """Unified SDF shape evaluator."""
    p_unbent = unbend(p, curvature)

    x = p_unbent[..., 0]
    y = p_unbent[..., 1]
    z = p_unbent[..., 2]

    L_half = half_extents[2]
    u = z / L_half
    u_clipped = np.clip(u, -1.0, 1.0)

    S = (1.0 - taper * u_clipped) + side_curvature * (1.0 - u_clipped**2)
    S = np.maximum(S, 0.01)

    x_scaled = x / S
    y_scaled = (y / S) / aspect_ratio

    p_scaled = np.stack([x_scaled, y_scaled, z], axis=-1)

    r0 = half_extents[0]
    if shape_type in ['cube', 'pyramid']:
        d = sdf_box(p_scaled, half_extents)
    elif shape_type in ['cylinder', 'cone']:
        d = sdf_cylinder_core(p_scaled, r0, L_half)
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")

    return d * S


def generate_mesh(shape_type, curvature, taper, side_curvature, aspect_ratio, resolution=200):
    """Extract a 3D mesh from the parameterized shape SDF using marching cubes."""
    limit = 1.6
    x = np.linspace(-limit, limit, resolution)
    y = np.linspace(-limit, limit, resolution)
    z = np.linspace(-limit, limit, resolution)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    pts = np.stack([X, Y, Z], axis=-1)

    half_extents = np.array([0.5, 0.5, 1.0])

    sdf_vals = sdf_shape(
        pts, shape_type, half_extents, curvature, taper, side_curvature, aspect_ratio
    )

    spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    try:
        verts, faces, _, _ = marching_cubes(sdf_vals, level=0.0, spacing=spacing)
        verts += np.array([x[0], y[0], z[0]])

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.unmerge_vertices() # force flat shading
        return mesh
    except ValueError:
        return None


def make_lookat(eye, target, up):
    """Compute look-at camera pose matrix."""
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    pose = np.eye(4)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = eye
    return pose


def render_mesh(mesh, canvas_size, bg_color, shape_color, eye, target, up):
    """Render the mesh using pyrender and return a PIL Image."""
    if mesh is None:
        # Return a solid background image if mesh generation failed
        return Image.new("RGB", tuple(canvas_size), tuple(bg_color))

    bg_color_normalized = [c / 255.0 for c in bg_color]
    shape_color_normalized = [c / 255.0 for c in shape_color] + [1.0]

    scene = pyrender.Scene(bg_color=bg_color_normalized, ambient_light=[0.15, 0.15, 0.15])

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=shape_color_normalized,
        metallicFactor=0.1,
        roughnessFactor=0.55
    )
    pymesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene.add(pymesh)

    camera_pose = make_lookat(eye, target, up)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
    scene.add(camera, pose=camera_pose)

    # Add lights (Three-Point Lighting System for edge definition)
    # 1. Key Light (strong primary light from side/front-right)
    key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    key_light_pose = make_lookat(
        np.array([3.0, -1.5, 1.5]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    scene.add(key_light, pose=key_light_pose)

    # 2. Fill Light (soft light from front-left to illuminate shadows)
    fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
    fill_light_pose = make_lookat(
        np.array([-2.0, -3.0, 0.5]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    scene.add(fill_light, pose=fill_light_pose)

    # 3. Rim Light (back-light from behind/top-back-left to highlight silhouette edges)
    rim_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.2)
    rim_light_pose = make_lookat(
        np.array([-3.0, 3.0, 1.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    scene.add(rim_light, pose=rim_light_pose)

    # Render offscreen
    width, height = canvas_size
    r = pyrender.OffscreenRenderer(width, height)
    color, _ = r.render(scene)
    r.delete()

    return Image.fromarray(color)


@register("nap_vs_mp_3d", "low_mid_vision")
@generator(NapVsMp3dConfig)
def generate_all(config: NapVsMp3dConfig):

    # Check if one of the opengl platforms is supported
    is_render_configured = False
    try:
        pyrender.OffscreenRenderer(20, 20)
        is_render_configured = True
    except:
        for platform in ['osmesa', 'egl']:
            try:
                os.environ['PYOPENGL_PLATFORM'] = platform
                pyrender.OffscreenRenderer(20, 20)
                is_render_configured = True
            except:
                pass
    if not is_render_configured:
        raise AttributeError(
            'No supported OpenGL found. See: '
            'https://pyrender.readthedocs.io/en/latest/examples/offscreen.html'
        )

    """generate nap vs mp 3d geons dataset."""
    output_folder = Path(config.output_folder)

    for cond in ["reference", "mp", "nap"]:
        (output_folder / cond).mkdir(exist_ok=True, parents=True)

    dimensions = [d.strip() for d in config.dimensions.split(",")]

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
                "Dimension",
                "ReferencePath",
                "MPPath",
                "NAPPath",
                "BackgroundColor",
            ]
        )

        sample_id = 0
        for dim in dimensions:
            for i in tqdm(range(config.num_samples_per_dimension), desc=f"Generating {dim}"):
                sample_name = f"{dim}_{i}"

                # 1. Sample Camera pose (shared across the triplet to preserve viewpoint)
                if config.random_camera:
                    distance = np.random.uniform(
                        config.min_camera_distance, config.max_camera_distance
                    )
                    elevation = np.radians(
                        np.random.uniform(config.min_camera_elevation, config.max_camera_elevation)
                    )
                    azimuth = np.radians(np.random.uniform(
                        config.min_camera_azimuth, config.max_camera_azimuth)
                    )
                    # Subtle target randomization
                    target = np.random.uniform(-0.05, 0.05, size=3)
                else:
                    distance = (config.min_camera_distance + config.max_camera_distance) / 2.0
                    elevation = np.radians(
                        (config.min_camera_elevation + config.max_camera_elevation) / 2.0
                    )
                    azimuth = np.radians(
                        (config.min_camera_azimuth + config.max_camera_azimuth) / 2.0
                    )
                    target = np.zeros(3)

                eye = np.array([
                    distance * np.cos(elevation) * np.sin(azimuth),
                    distance * np.cos(elevation) * np.cos(azimuth),
                    distance * np.sin(elevation)
                ])
                up = np.array([0.0, 0.0, 1.0])

                # Pick a random base shape class for this sample
                base_shape = np.random.choice(["cube", "cylinder"])

                # 2. Define geometry parameters for the triplet based on the dimension
                if dim == "axis_curvature":
                    # Sample metric change (MP), then set reference to half of it
                    mp_val = np.random.uniform(config.min_curvature, config.max_curvature)
                    mp_val = np.random.choice([1., -1.]) * mp_val  # flip the curvature direction
                    ref_val = 0.5 * mp_val
                    ref_params = (base_shape, ref_val, 0.0, 0.0, 1.0)
                    mp_params  = (base_shape, mp_val, 0.0, 0.0, 1.0)
                    nap_params = (base_shape, 0.0, 0.0, 0.0, 1.0)
                elif dim == "taper":
                    # Sample metric change (MP), then set reference to half of it
                    mp_val = np.random.uniform(config.min_taper, config.max_taper)
                    ref_val = 0.5 * mp_val
                    ref_params = (base_shape, 0.0, ref_val, 0.0, 1.0)
                    mp_params  = (base_shape, 0.0, mp_val, 0.0, 1.0)
                    nap_params = (base_shape, 0.0, 0.0, 0.0, 1.0)
                elif dim == "side_curvature":
                    # Sample metric change (MP), then set reference to half of it
                    mp_val = np.random.uniform(config.min_side_curvature, config.max_side_curvature)
                    mp_val = np.random.choice([1., -1.]) * mp_val  # flip between convex and concave
                    ref_val = 0.5 * mp_val
                    ref_params = (base_shape, 0.0, 0.0, ref_val, 1.0)
                    mp_params  = (base_shape, 0.0, 0.0, mp_val, 1.0)
                    nap_params = (base_shape, 0.0, 0.0, 0.0, 1.0)
                elif dim == "cross_section":
                    # Sample metric aspect ratio (MP), then set reference to halfway between 1.0 and MP
                    mp_val = np.random.uniform(config.min_aspect_ratio, config.max_aspect_ratio)
                    ref_val = 1.0 + 0.5 * (mp_val - 1.0)
                    ref_params = ("cube", 0.0, 0.0, 0.0, ref_val)
                    mp_params  = ("cube", 0.0, 0.0, 0.0, mp_val)
                    nap_params = ("cylinder", 0.0, 0.0, 0.0, 1.0)
                else:
                    # Default fallback
                    ref_params = (base_shape, 0.0, 0.0, 0.0, 1.0)
                    mp_params  = (base_shape, 0.0, 0.0, 0.0, 1.0)
                    nap_params = (base_shape, 0.0, 0.0, 0.0, 1.0)

                def render_condition(params):
                    shape_type, curvature, taper, side_curvature, aspect_ratio = params
                    mesh = generate_mesh(
                        shape_type=shape_type,
                        curvature=curvature,
                        taper=taper,
                        side_curvature=side_curvature,
                        aspect_ratio=aspect_ratio,
                        resolution=config.mesh_resolution
                    )
                    return render_mesh(
                        mesh=mesh,
                        canvas_size=config.canvas_size,
                        bg_color=config.background_color,
                        shape_color=config.shape_color,
                        eye=eye,
                        target=target,
                        up=up
                    )

                ref_img = render_condition(ref_params)
                ref_path = save_image(ref_img, "reference", sample_id, sample_name)

                mp_img = render_condition(mp_params)
                mp_path = save_image(mp_img, "mp", sample_id, sample_name)

                nap_img = render_condition(nap_params)
                nap_path = save_image(nap_img, "nap", sample_id, sample_name)

                writer.writerow(
                    [
                        sample_id,
                        sample_name,
                        dim,
                        ref_path,
                        mp_path,
                        nap_path,
                        config.background_color,
                    ]
                )
                sample_id += 1

    return str(output_folder)
