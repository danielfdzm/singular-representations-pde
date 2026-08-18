"""Shared repository paths for standalone experiment drivers.

The experiment scripts remain directly executable from the repository root.
Reference data and generated output live outside the source directory so a
reproduction run cannot leave plots or caches mixed in with the code.
"""

from __future__ import annotations

from pathlib import Path


EXPERIMENT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIRECTORY.parent

REFERENCE_DATA_DIRECTORY = REPOSITORY_ROOT / "data" / "reference"
TANH_DATA_DIRECTORY = REFERENCE_DATA_DIRECTORY / "tanh"
GAUSSIAN_DATA_DIRECTORY = REFERENCE_DATA_DIRECTORY / "gaussian"
COMPLETION_DATA_DIRECTORY = REFERENCE_DATA_DIRECTORY / "completion"
WEAK_PDE_DATA_DIRECTORY = REFERENCE_DATA_DIRECTORY / "weak_pde"
INGHAM_DATA_DIRECTORY = REFERENCE_DATA_DIRECTORY / "ingham"

OUTPUT_DIRECTORY = REPOSITORY_ROOT / "outputs"
NATIVE_DIRECTORY = REPOSITORY_ROOT / "native"
GALLERY_DIRECTORY = REPOSITORY_ROOT / "figures" / "gallery"
SUPPLEMENTARY_FIGURE_DIRECTORY = REPOSITORY_ROOT / "figures" / "supplementary"


def experiment_output_directory(topic: str) -> Path:
    """Return the ignored output directory for one topic."""

    return OUTPUT_DIRECTORY / topic
