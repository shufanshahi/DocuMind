"""Loaders that turn raw files (PDF, Markdown, plain text) into a common
`Document` representation: one big string of text plus enough metadata to
trace any later chunk back to its source (and, for PDFs, back to a page).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


@dataclass
class PageSpan:
    """Marks which character range of `Document.text` came from which PDF page."""

    page_number: int  # 1-indexed
    start_char: int
    end_char: int


@dataclass
class Document:
    doc_id: str
    source_path: str
    filetype: str  # "pdf" | "markdown" | "text"
    text: str
    page_spans: list[PageSpan] = field(default_factory=list)

    def page_for_offset(self, char_offset: int) -> int | None:
        """Return the 1-indexed PDF page that char_offset falls in, if known."""
        for span in self.page_spans:
            if span.start_char <= char_offset < span.end_char:
                return span.page_number
        return None


def _make_doc_id(path: Path) -> str:
    # Stable-ish, readable id: filename stem + short hash of the full path,
    # so two files named "notes.md" in different folders don't collide.
    digest = uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve())).hex[:8]
    return f"{path.stem}-{digest}"


def load_txt(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="replace")
    return Document(doc_id=_make_doc_id(path), source_path=str(path), filetype="text", text=text)


_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


def load_markdown(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8", errors="replace")
    # Strip a leading YAML frontmatter block (--- ... ---) if present, since it's
    # metadata, not prose, and would otherwise pollute the first chunk.
    text = _FRONTMATTER_RE.sub("", raw, count=1)
    return Document(doc_id=_make_doc_id(path), source_path=str(path), filetype="markdown", text=text)


def load_pdf(path: Path) -> Document:
    from pypdf import PdfReader  # imported lazily so txt/md-only users don't need it

    reader = PdfReader(str(path))
    parts: list[str] = []
    spans: list[PageSpan] = []
    cursor = 0
    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if not page_text.endswith("\n"):
            page_text += "\n"
        start = cursor
        parts.append(page_text)
        cursor += len(page_text)
        spans.append(PageSpan(page_number=i, start_char=start, end_char=cursor))
    return Document(
        doc_id=_make_doc_id(path),
        source_path=str(path),
        filetype="pdf",
        text="".join(parts),
        page_spans=spans,
    )


_LOADERS = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_txt,
}


def load_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix not in _LOADERS:
        raise ValueError(f"Unsupported file type {suffix!r} for {path}")
    return _LOADERS[suffix](path)


def load_documents_from_folder(folder: Path) -> list[Document]:
    """Recursively load every supported file under `folder`, skipping anything
    with an extension we don't have a loader for (and hidden files/dirs).
    """
    folder = Path(folder)
    paths = sorted(
        p
        for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(part.startswith(".") for part in p.relative_to(folder).parts)
    )
    return [load_document(p) for p in paths]
