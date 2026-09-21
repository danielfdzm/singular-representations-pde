"""Read uncompressed reruns or the compressed published records."""
import gzip
import json
from pathlib import Path


def load_results(directory):
    path = Path(directory)/"results.json"
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(path.with_suffix(".json.gz"), "rt", encoding="utf-8") as stream:
        return json.load(stream)
