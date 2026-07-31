"""shared non-drawing utilities."""

import os
import pathlib
import random
import shutil
import subprocess
from typing import Any

import cv2
import numpy as np
import PIL
import PIL.Image
import sty
import tqdm
from PIL import ImageFilter

DEFAULTS = {
    "canvas_size": [224, 224],
    "background_color": [0, 0, 0],
    "antialiasing": True,
    "behaviour_if_present": "overwrite",
}


def to_list(val: Any) -> list[Any]:
    """ensure val is a list of options."""
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]


def conditional_tqdm(iterable, enable_tqdm, **kwargs):
    """wrap iterable with tqdm if enabled."""
    if enable_tqdm:
        return tqdm.tqdm(iterable, **kwargs)
    return iterable


def conver_tensor_to_plot(tensor, mean, std):
    """convert normalized tensor to plottable numpy array."""
    tensor = tensor.numpy().transpose((1, 2, 0))
    image = std * tensor + mean
    image = np.clip(image, 0, 1)
    if np.shape(image)[2] == 1:
        image = np.squeeze(image)
    return image


def convert_normalized_tensor_to_plottable_array(tensor, mean, std, text):
    """convert tensor to uint8 image array with text overlay."""
    image = conver_tensor_to_plot(tensor, mean, std)
    canvas_size = np.shape(image)
    font_scale = np.ceil(canvas_size[1]) / 150
    font = cv2.QT_FONT_NORMAL
    umat = cv2.UMat(image * 255)
    umat = cv2.putText(
        img=cv2.UMat(umat),
        text=text,
        org=(0, int(canvas_size[1] - 3)),
        fontFace=font,
        fontScale=font_scale,
        color=[0, 0, 0],
        lineType=cv2.LINE_AA,
        thickness=6,
    )
    umat = cv2.putText(
        img=cv2.UMat(umat),
        text=text,
        org=(0, int(canvas_size[1] - 3)),
        fontFace=font,
        fontScale=font_scale,
        color=[255, 255, 255],
        lineType=cv2.LINE_AA,
        thickness=1,
    )
    image = cv2.UMat.get(umat)
    return np.array(image, np.uint8)


def convert_lists_to_strings(obj):
    """recursively convert lists to string representations in a dict."""
    if isinstance(obj, list):
        return str(obj)
    if isinstance(obj, dict):
        return {k: convert_lists_to_strings(v) for k, v in obj.items()}
    return obj


def pretty_print_dict(dictionary, indent=0, name=None):
    """print a dict with colored keys and values."""
    if name is not None:
        print(sty.fg.red + f"~~~ {name} ~~~" + sty.rs.fg)
    for key, value in sorted(dictionary.items()):
        print(" " * indent + sty.fg.blue + key + sty.rs.fg, end=": ")
        if isinstance(value, dict):
            print()
            pretty_print_dict(value, indent + 4)
        else:
            print(sty.fg.green + str(value) + sty.rs.fg)


def update_dict(dictA, dictB, replace=True):
    """recursively update dictA with values from dictB."""
    for key in dictB:
        if dictB[key] is None:
            continue
        if (
            key in dictA
            and isinstance(dictA[key], dict)
            and isinstance(dictB[key], dict)
        ):
            update_dict(dictA[key], dictB[key], replace)
        elif replace or key not in dictA:
            old_value = dictA.get(key, "none")
            dictA[key] = dictB[key]
            if old_value != dictB[key]:
                print(
                    sty.fg.blue
                    + f"updated {key}: "
                    + sty.rs.fg
                    + sty.fg.red
                    + f"{old_value} => "
                    + sty.rs.fg
                    + sty.fg.green
                    + f"{dictB[key]}"
                    + sty.rs.fg
                )
    return dictA


def apply_antialiasing(img: PIL.Image, amount=None):
    """apply gaussian blur for antialiasing."""
    if amount is None:
        amount = min(img.size) * 0.00334
    return img.filter(ImageFilter.GaussianBlur(radius=amount))


def delete_and_recreate_path(path: pathlib.Path):
    """remove directory if exists, then recreate it."""
    shutil.rmtree(path) if path.exists() else None
    path.mkdir(parents=True, exist_ok=True)


def generate_random_color():
    """return a random RGB tuple."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def check_download_ETH_80_dataset(destination_dir):
    """download ETH-80 dataset if not present."""
    repo_url = "https://github.com/chenchkx/ETH-80/"
    if not os.path.exists(destination_dir):
        print(f"ETH-80 dataset not found in {destination_dir}. downloading (~308MB).")
        subprocess.run(["git", "clone", repo_url, destination_dir])
        _reorganize_eth_80(destination_dir)
    else:
        print(f"ETH-80 dataset found in {destination_dir}")


def _reorganize_eth_80(base_dir):
    """reorganize ETH-80 directory structure after cloning."""
    images_dir = os.path.join(base_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path) and folder != "images":
            shutil.move(folder_path, images_dir)
    target_dir = os.path.join(base_dir, "maps")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for class_folder in os.listdir(images_dir):
        class_path = os.path.join(images_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        for object_folder in os.listdir(class_path):
            object_path = os.path.join(class_path, object_folder)
            if not os.path.isdir(object_path):
                continue
            map_folder_path = os.path.join(object_path, "maps")
            if os.path.exists(map_folder_path):
                target_map = os.path.join(target_dir, class_folder, object_folder)
                os.makedirs(target_map, exist_ok=True)
                for map_file in os.listdir(map_folder_path):
                    shutil.move(os.path.join(map_folder_path, map_file), target_map)
                shutil.rmtree(map_folder_path)
