"""Shared repository paths for standalone experiment drivers.

The experiment scripts remain directly executable from the repository root.
Stored data, generated output, and paper previews live outside the source
directory so that the code remains easy to browse.
"""

from __future__ import annotations

from pathlib import Path


EXPERIMENT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIRECTORY.parent

DATA_DIRECTORY = REPOSITORY_ROOT / "data"
TANH_DATA_DIRECTORY = DATA_DIRECTORY / "tanh"
GAUSSIAN_DATA_DIRECTORY = DATA_DIRECTORY / "gaussian"
COMPLETION_DATA_DIRECTORY = DATA_DIRECTORY / "completion"
WEAK_PDE_DATA_DIRECTORY = DATA_DIRECTORY / "weak_pde"
INGHAM_DATA_DIRECTORY = DATA_DIRECTORY / "ingham"

OUTPUT_DIRECTORY = REPOSITORY_ROOT / "outputs"
NATIVE_DIRECTORY = REPOSITORY_ROOT / "native"
PREVIEW_FIGURE_DIRECTORY = REPOSITORY_ROOT / "figures" / "previews"


def experiment_output_directory(topic: str) -> Path:
    """Return the generated-output directory for one topic."""

    return OUTPUT_DIRECTORY / topic
