"""Reorder the archived tanh trajectories without redrawing scientific paths.

The original publication PDF is the vector source. Its two plot-and-title
regions are exchanged; axes, tick labels, page dimensions, embedded fonts,
trajectory paths, arrowheads, and start/end markers are retained. The legend
moves with the single-precision panel to the left. Only panel letters change.

Python's standard library performs the PDF composition. Ghostscript is used
only to make a 600-dpi PNG preview; the PDF itself remains vector artwork.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zlib


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
DATA = REPOSITORY / "data" / "tanh_bias"
SOURCE_SHA256 = "5e680f039d8b32ba7fc736a4235031291bf5f0faf926fefe0be55f96721f5c4e"
PAGE_WIDTH = 519.951625
PAGE_HEIGHT = 177.9225
LEFT_X = 38.415625
RIGHT_X = 304.7095197368
AXIS_WIDTH = 208.0421052632
AXIS_BOTTOM = 34.36875
AXIS_HEIGHT = 122.76


def pdf_literal(text: str) -> bytes:
    """Use the original embedded font's UTF-16BE string encoding."""
    encoded = text.encode("utf-16be")
    for character in (b"\\", b"(", b")"):
        encoded = encoded.replace(character, b"\\" + character)
    return encoded


def read_vector_stream(source: bytes) -> bytes:
    match = re.search(
        rb"9 0 obj\s*<<.*?>>\s*stream\r?\n(.*?)\r?\nendstream",
        source, re.S,
    )
    if not match:
        raise ValueError("Expected archived Matplotlib page-content stream 9")
    return zlib.decompress(match.group(1))


def translated_region(stream: bytes, destination_x: float, shift: float) -> bytes:
    # Half a point includes the complete 0.75-point frame without removing
    # the adjacent y-axis labels or the x-axis ticks below the plotting box.
    pad = 0.5
    x = destination_x - pad
    y = AXIS_BOTTOM - pad
    width = AXIS_WIDTH + 2 * pad
    height = PAGE_HEIGHT - y
    prefix = (
        f"\nq\n{x:.10f} {y:.10f} {width:.10f} {height:.10f} re W n\n"
        f"1 0 0 1 {shift:.10f} 0 cm\n"
    ).encode("ascii")
    return prefix + stream + b"\nQ\n"


def retain_text(stream: bytes, *, title: str | None = None,
                axes: bool = False, legend: bool = False) -> bytes:
    """Avoid duplicate, clipped-out labels in the composed PDF text layer."""
    def select(match: re.Match[bytes]) -> bytes:
        block = match.group()
        if b"/F1 9.5 Tf" in block:
            keep = title is not None and pdf_literal(title) in block
        elif re.search(rb"/F[123] 7\.7 Tf", block):
            keep = legend
        else:
            keep = axes
        return block if keep else b""
    return re.sub(rb"\bBT\b.*?\bET\b", select, stream, flags=re.S)


def append_replacement_stream(source: bytes, stream: bytes) -> bytes:
    """Write a standards-compliant incremental PDF page-content update."""
    compressed = zlib.compress(stream)
    previous_xref = int(re.findall(rb"startxref\s+(\d+)", source)[-1])
    trailers = re.findall(rb"trailer\s*<<(.*?)>>\s*startxref", source, re.S)
    if len(trailers) != 1:
        raise ValueError("Expected the single-revision archived source PDF")
    trailer = trailers[0].strip()
    stream_offset = len(source) + 1
    update = (
        b"\n9 0 obj\n<< /Length " + str(len(compressed)).encode("ascii")
        + b" /Filter /FlateDecode >>\nstream\n" + compressed
        + b"\nendstream\nendobj\n"
    )
    xref_offset = len(source) + len(update)
    update += (
        f"xref\n9 1\n{stream_offset:010d} 00000 n \ntrailer\n<< ".encode("ascii")
        + trailer + f" /Prev {previous_xref} >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return source + update


def verify_endpoint_mapping(stream: bytes, original_directory: Path) -> dict:
    """Check which source panel contains the two saved precision endpoints."""
    report = {}
    x_scale = (178.8796526344 - 142.4366694579) / 5
    y_scale = 118.3296629924 - 93.1246265217
    for precision, filename, source_center in (
        ("single", "tanh_parameter_escape_loss_float32.json", 408.7305641947),
        ("double", "tanh_parameter_escape_loss.json", 142.4366694579),
    ):
        path = original_directory / filename
        trace = json.loads(path.read_text())
        row = trace["history"][-1]
        # Original endpoint markers are triangles centered on the final data
        # values; their first vertex is shifted by 2.5 PDF points horizontally.
        center_x = source_center + x_scale * row["w1"]
        center_y = 93.1246265217 + y_scale * row["b1"]
        tip = (center_x + 2.5, center_y)
        vertices = [(float(x), float(y)) for x, y in re.findall(
            rb"([\d.-]+) ([\d.-]+) m", stream)]
        error = min(max(abs(x - tip[0]), abs(y - tip[1])) for x, y in vertices)
        if error > 1e-4:
            raise ValueError(f"Archived {precision} endpoint does not match the saved trace")
        report[precision] = {
            "source_trace": filename,
            "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "history_rows": len(trace["history"]),
            "final_w1": row["w1"],
            "final_b1": row["b1"],
            "final_w2": row["w2"],
            "final_b2": row["b2"],
            "source_endpoint_match_error_pdf_points": error,
            "destination_panel": "a, left" if precision == "single" else "b, right",
        }
    return report


def main() -> None:
    original_directory = DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=original_directory / "source_trajectories.pdf")
    parser.add_argument("--output", type=Path,
                        default=REPOSITORY / "outputs" / "figures" / "figure_03")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--pdf-only", action="store_true",
                        help="Write the vector PDF without requiring Ghostscript")
    args = parser.parse_args()
    source = args.source.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Source PDF does not match the verified publication original")
    stream = read_vector_stream(source)
    endpoints = verify_endpoint_mapping(stream, original_directory)
    relabelled = stream
    for old, new in (("(a) double precision", "(b) double precision"),
                     ("(b) single precision", "(a) single precision")):
        encoded = pdf_literal(old)
        if relabelled.count(encoded) != 1:
            raise ValueError(f"Expected one original title: {old}")
        relabelled = relabelled.replace(encoded, pdf_literal(new))
    separation = RIGHT_X - LEFT_X
    updated_stream = (
        retain_text(stream, axes=True)
        + translated_region(retain_text(relabelled, title="(a) single precision",
                                        legend=True), LEFT_X, -separation)
        + translated_region(retain_text(relabelled, title="(b) double precision"),
                            RIGHT_X, separation)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output.with_suffix(".pdf")
    pdf_path.write_bytes(append_replacement_stream(source, updated_stream))
    png_path = None
    if not args.pdf_only:
        ghostscript = shutil.which("gs")
        if not ghostscript:
            raise RuntimeError("Vector PDF was written; install Ghostscript or use --pdf-only")
        png_path = args.output.with_suffix(".png")
        subprocess.run([
            ghostscript, "-q", "-dSAFER", "-dNOPAUSE", "-dBATCH",
            "-sDEVICE=png16m", "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
            f"-r{args.dpi}", f"-sOutputFile={png_path}", str(pdf_path),
        ], check=True)
    metadata = {
        "method": "vector PDF region exchange; no redrawing, interpolation, or raster PDF content",
        "source": args.source.name,
        "source_sha256": digest,
        "source_precision_order": ["double", "single"],
        "output_precision_order": ["single", "double"],
        "output_titles": ["(a) single precision", "(b) double precision"],
        "legend_position": "upper right of left, single-precision panel",
        "page_size_pdf_points": [PAGE_WIDTH, PAGE_HEIGHT],
        "page_size_inches": [PAGE_WIDTH / 72, PAGE_HEIGHT / 72],
        "horizontal_panel_translation_pdf_points": separation,
        "preserved": ["original vector scientific paths", "embedded fonts", "colors",
                      "stroke widths", "trajectory arrows", "start and end markers",
                      "legend", "axes and tick labels", "page dimensions"],
        "endpoint_precision_checks": endpoints,
        "png_dpi": args.dpi if png_path else None,
        "outputs": {"pdf": pdf_path.name, "png": png_path.name if png_path else None},
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
