"""Ingest PDFs / text files into Oracle: text -> chunks -> embeddings -> VECTOR rows.

Usage:
    python -m aero.ingest                 # everything under data/pdfs/
    python -m aero.ingest path/to/file.pdf --doc-type paper

Sub-folders become the doc_type filter, e.g. data/pdfs/standards/foo.pdf
-> doc_type="standards".
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from . import config, db
from .embeddings import embed

SUPPORTED = {".pdf", ".txt", ".md"}


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", max(start, end - 200), end)  # prefer a word boundary
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if len(piece) > 40:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def read_pages(path: Path) -> list[tuple[int, str]]:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(i + 1, (p.extract_text() or "")) for i, p in enumerate(reader.pages)]
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]


def ingest_file(conn, path: Path, doc_type: str) -> int:
    rows: list[tuple[int, str, list[float]]] = []
    pending: list[tuple[int, str]] = []
    for page_no, text in read_pages(path):
        for chunk in chunk_text(text):
            pending.append((page_no, chunk))
    if not pending:
        print(f"  ! no extractable text in {path.name} (scanned PDF?)")
        return 0
    vectors = embed([c for _, c in pending])
    rows = [(pg, ch, v) for (pg, ch), v in zip(pending, vectors)]
    db.replace_source(conn, path.name, doc_type, rows)
    return len(rows)


def ingest_path(conn, target: Path, doc_type: str | None = None) -> int:
    files = (
        [target]
        if target.is_file()
        else sorted(p for p in target.rglob("*") if p.suffix.lower() in SUPPORTED)
    )
    total = 0
    for f in files:
        dtype = doc_type or (f.parent.name if f.parent != target else "general")
        n = ingest_file(conn, f, dtype)
        print(f"  {f.name}: {n} chunks (doc_type={dtype})")
        total += n
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("path", nargs="?", default=str(config.DATA_DIR))
    ap.add_argument("--doc-type", default=None)
    args = ap.parse_args()
    conn = db.connect()
    db.init_schema(conn)
    total = ingest_path(conn, Path(args.path), args.doc_type)
    print(f"Done. {total} chunks ingested; table now holds {db.count_chunks(conn)} chunks.")


if __name__ == "__main__":
    main()
