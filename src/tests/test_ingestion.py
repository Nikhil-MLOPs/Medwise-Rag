from pathlib import Path

from src.ingestion.loaders.pdf_loader import load_pdf


def test_pdf_loader_exists(tmp_path: Path):
    assert load_pdf is not None