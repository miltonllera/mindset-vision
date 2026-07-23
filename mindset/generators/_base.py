"""base infrastructure for dataset generators: config dataclass, decorator, and registration."""

import dataclasses
import functools
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import sty

from mindset.generators import REGISTRY
from mindset.utils import delete_and_recreate_path


@dataclass
class GeneratorConfig:
    """base config shared by all generators."""

    canvas_size: tuple[int, int] = field(
        default_factory=lambda: (224, 224),
        metadata={"min": 32, "max": 1024, "step": 16, "label": "canvas size"},
    )
    background_color: list = field(
        default_factory=lambda: [0, 0, 0], metadata={"label": "background color (RGB)"}
    )
    antialiasing: bool = field(default=True, metadata={"label": "antialiasing"})
    behaviour_if_present: str = field(
        default="overwrite",
        metadata={"choices": ["overwrite", "skip"], "label": "if folder exists"},
    )
    output_folder: str = field(default="", metadata={"label": "output folder"})


def generator(config_cls):
    """decorator that handles all generator boilerplate."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(**kwargs):
            config = config_cls(**{k: v for k, v in kwargs.items() if v is not None})
            output_folder = Path(config.output_folder)

            if output_folder.exists() and config.behaviour_if_present == "skip":
                print(sty.fg.yellow + "dataset already exists. skipping" + sty.rs.fg)
                return str(output_folder)

            delete_and_recreate_path(output_folder)

            with open(output_folder / "config.json", "w") as f:
                json.dump(asdict(config), f, indent=2, default=str)

            return func(config)

        wrapper.config_cls = config_cls
        return wrapper

    return decorator


def register(name, category):
    """register a generator in the global registry."""

    def decorator(func):
        REGISTRY[name] = {
            "func": func,
            "category": category,
            "config_cls": func.config_cls,
        }
        return func

    return decorator


def config_to_argparser(config_cls, description=""):
    """build argparse.ArgumentParser from a config dataclass."""
    import argparse

    parser = argparse.ArgumentParser(description=description)
    for fld in fields(config_cls):
        name = f"--{fld.name}"
        default = (
            fld.default
            if fld.default is not dataclasses.MISSING
            else fld.default_factory()
        )
        meta = fld.metadata

        match fld.type:
            case t if t == bool:
                parser.add_argument(name, action="store_true", default=default)
                continue
            case t if t == list:
                kwargs = {"nargs": "+", "type": int}
            case t if t == int:
                kwargs = {"type": int}
            case t if t == float:
                kwargs = {"type": float}
            case _:
                kwargs = {"type": str}

        kwargs["default"] = default
        if "choices" in meta:
            kwargs["choices"] = meta["choices"]
        if "label" in meta:
            kwargs["help"] = meta["label"]
        parser.add_argument(name, **kwargs)
    return parser
