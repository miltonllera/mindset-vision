"""fast smoke tests for mindset-vision package."""
import shutil
from pathlib import Path


def test_package_imports():
    """verify core modules import correctly."""
    from mindset.utils import DEFAULTS
    assert "canvas_size" in DEFAULTS

    from mindset.drawing.base import DrawStimuli
    assert DrawStimuli is not None


def test_generator_registry():
    """verify all 33 generators register via auto-discovery."""
    from mindset.cli import _load_registry
    from mindset.generators import list_generators

    registry = _load_registry()
    assert len(registry) == 33

    cats = list_generators()
    assert len(cats["visual_illusions"]) == 10
    assert len(cats["low_mid_vision"]) == 9
    assert len(cats["shape_recognition"]) == 14


def test_generate_ebbinghaus():
    """smoke test: generate a small ebbinghaus dataset."""
    from mindset.generators.visual_illusions.ebbinghaus import generate_all

    out = Path("/tmp/mindset_ci_ebbinghaus")
    if out.exists():
        shutil.rmtree(out)

    result = generate_all(
        num_samples_scrambled=2,
        num_samples_illusory=1,
        output_folder=str(out),
    )

    assert Path(result).exists()
    assert (out / "annotation.csv").exists()
    assert len(list(out.rglob("*.png"))) >= 3
    shutil.rmtree(out)


def test_generate_ponzo_random_target_lines(tmp_path):
    """smoke test: generate ponzo scrambled samples with random targets."""
    from mindset.generators.visual_illusions.ponzo import generate_all

    out = tmp_path / "ponzo-random-target-lines"

    result = generate_all(
        num_samples_scrambled=1,
        num_samples_illusory=1,
        rnd_target_lines=True,
        output_folder=str(out),
    )

    assert Path(result).exists()
    assert (out / "annotation.csv").exists()
    assert len(list(out.rglob("*.png"))) == 3


def test_cli_entry_point():
    """verify the CLI entry point is callable."""
    from mindset.cli import main
    assert callable(main)


def test_generate_relational_vs_coordinate(tmp_path):
    """smoke test: generate a small relational vs coordinate dataset."""
    from mindset.generators.low_mid_vision.relational_vs_coordinate import generate_all

    out = tmp_path / "relational-vs-coordinate-smoke"

    result = generate_all(
        num_samples=2,
        output_folder=str(out),
    )

    assert Path(result).exists()
    assert (out / "annotation.csv").exists()
    # 2 samples * 3 conditions (basis, coord change, relation change) = 6 images
    assert len(list(out.rglob("*.png"))) == 6


def test_generate_uncrowding(tmp_path):
    """smoke test: generate a small uncrowding dataset."""
    from mindset.generators.low_mid_vision.uncrowding import generate_all

    out = tmp_path / "uncrowding-smoke"

    result = generate_all(
        num_samples_vernier_inside=4,
        num_samples_vernier_outside=4,
        output_folder=str(out),
    )

    assert Path(result).exists()
    assert (out / "annotation.csv").exists()
    assert len(list(out.rglob("*.png"))) == 8


