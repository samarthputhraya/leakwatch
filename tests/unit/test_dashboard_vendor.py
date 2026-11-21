"""Vendored front-end libraries (D-052): the files on disk are the ones named in the manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import leakwatch.dashboard as dashboard

VENDOR = Path(dashboard.__file__).parent / "static" / "vendor"
LIBRARIES = {"deck.gl.min.js", "echarts.min.js", "htm-preact.module.js"}
LICENCES = {
    "LICENSE-deck.gl.txt",
    "LICENSE-echarts.txt",
    "NOTICE-echarts.txt",
    "LICENSE-htm.txt",
    "LICENSE-preact.txt",
}


def manifest() -> list[dict]:
    return json.loads((VENDOR / "manifest.json").read_text(encoding="utf-8"))


def test_every_vendored_file_matches_its_recorded_hash_and_size():
    entries = manifest()
    assert {e["file"] for e in entries} == LIBRARIES
    for entry in entries:
        body = (VENDOR / entry["file"]).read_bytes()
        assert len(body) == entry["bytes"], entry["file"]
        assert hashlib.sha256(body).hexdigest() == entry["sha256"], entry["file"]


def test_the_manifest_names_version_source_and_licence():
    for entry in manifest():
        assert entry["version"] in entry["source"]
        assert entry["source"].startswith("https://cdn.jsdelivr.net/npm/")
        assert entry["license"] and entry["repository"].startswith("https://github.com/")


def test_every_vendored_library_ships_its_licence_text():
    assert {p.name for p in VENDOR.iterdir()} >= LICENCES
