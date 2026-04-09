"""Tests verifying the symusic/HF migration is complete."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import miditok
from miditok import REMI, Octuple
from miditok.midi_adapter import AdapterScore

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src" / "miditok"
SAMPLE_MIDI = REPO_ROOT / "violin_partita_bwv-1004_1_(c)grossman.mid"


def _iter_python_files(root: Path):
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _file_imports(p: Path) -> list[str]:
    try:
        tree = ast.parse(p.read_text())
    except SyntaxError:
        return []
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def test_no_symusic_imports() -> None:
    offending = []
    for p in _iter_python_files(SRC_DIR):
        for mod in _file_imports(p):
            if mod == "symusic" or mod.startswith("symusic."):
                offending.append((p, mod))
    assert offending == [], f"symusic imports remain: {offending}"


def test_no_huggingface_imports() -> None:
    offending = []
    for p in _iter_python_files(SRC_DIR):
        for mod in _file_imports(p):
            if mod == "huggingface_hub" or mod.startswith("huggingface_hub."):
                offending.append((p, mod))
    assert offending == [], f"huggingface_hub imports remain: {offending}"


def test_no_tokenizers_imports() -> None:
    offending = []
    for p in _iter_python_files(SRC_DIR):
        for mod in _file_imports(p):
            if mod == "tokenizers" or mod.startswith("tokenizers."):
                offending.append((p, mod))
    assert offending == [], f"tokenizers imports remain: {offending}"


def test_pyproject_dependencies() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text()
    assert "symusic" not in text.split("[project.optional-dependencies]")[0]
    assert "huggingface_hub" not in text
    assert '"tokenizers' not in text


def test_remi_encode_token_types() -> None:
    tok = REMI()
    score = AdapterScore.from_file(SAMPLE_MIDI)
    encoded = tok.encode(score)
    seq = encoded[0] if isinstance(encoded, list) else encoded
    assert any(t.startswith("Bar") for t in seq.tokens)
    assert any(t.startswith("Position") for t in seq.tokens)
    assert any(t.startswith("Pitch") for t in seq.tokens)
    assert any(t.startswith("Velocity") for t in seq.tokens)
    assert any(t.startswith("Duration") for t in seq.tokens)


def test_remi_decode_produces_score() -> None:
    tok = REMI()
    score = AdapterScore.from_file(SAMPLE_MIDI)
    encoded = tok.encode(score)
    decoded = tok.decode(encoded)
    assert isinstance(decoded, AdapterScore)
    assert decoded.note_num() > 0


def test_remi_round_trip_token_stability() -> None:
    tok = REMI()
    score = AdapterScore.from_file(SAMPLE_MIDI)
    enc1 = tok.encode(score)
    decoded = tok.decode(enc1)
    enc2 = tok.encode(decoded)
    seqs1 = enc1 if isinstance(enc1, list) else [enc1]
    seqs2 = enc2 if isinstance(enc2, list) else [enc2]
    assert sum(len(s) for s in seqs1) == sum(len(s) for s in seqs2)


def test_octuple_encode_decode() -> None:
    tok = Octuple()
    score = AdapterScore.from_file(SAMPLE_MIDI)
    encoded = tok.encode(score)
    decoded = tok.decode(encoded)
    assert isinstance(decoded, AdapterScore)
    assert decoded.note_num() > 0


def test_octuple_round_trip() -> None:
    tok = Octuple()
    score = AdapterScore.from_file(SAMPLE_MIDI)
    enc1 = tok.encode(score)
    decoded = tok.decode(enc1)
    enc2 = tok.encode(decoded)
    seqs1 = enc1 if isinstance(enc1, list) else [enc1]
    seqs2 = enc2 if isinstance(enc2, list) else [enc2]
    assert len(seqs1) == len(seqs2)


def test_tokenizer_save_load_json(tmp_path: Path) -> None:
    tok = REMI()
    out = tmp_path / "tok.json"
    tok.save(out)
    loaded_data = json.loads(out.read_text())
    assert "config" in loaded_data
    assert loaded_data["tokenization"] == "REMI"
    assert "symusic_version" not in loaded_data
    assert "hf_tokenizers_version" not in loaded_data


def test_train_raises_not_implemented() -> None:
    tok = REMI()
    with pytest.raises(NotImplementedError):
        tok.train(vocab_size=1000)


def test_top_level_imports() -> None:
    assert hasattr(miditok, "REMI")
    assert hasattr(miditok, "Octuple")
    assert hasattr(miditok, "MusicTokenizer")
