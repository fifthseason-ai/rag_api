# app/utils/document_loader.py

import os
import codecs
import tempfile
import zipfile

from typing import Iterator, List, Optional
import chardet

from langchain_core.documents import Document

from app.config import known_source_ext, PDF_EXTRACT_IMAGES, CHUNK_OVERLAP, logger
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredEPubLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredRSTLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)




# ---------------------------------------------------------------------------
# Terminal verdicts (KI-02 SP-01.5)
#
# A file we will not ingest must say WHY in a machine-readable way. Before this,
# every parser failure collapsed into one opaque string ("Error during file
# processing: ..."), so an encrypted workbook, a truncated one and a genuinely
# unsupported format were indistinguishable to the caller and to the user.
# ---------------------------------------------------------------------------


class DocumentVerdictError(Exception):
    """A terminal, honest verdict about a file that cannot be ingested.

    `verdict` is the stable machine-readable token; `str(self)` is the
    human-readable explanation surfaced to the uploader. Nothing is stored for
    a file that raises this (it is raised during loading, before any
    `add_documents` call).
    """

    verdict = "failed"

    def __init__(self, message: str, *, filename: Optional[str] = None):
        super().__init__(message)
        self.filename = filename


class EncryptedDocumentError(DocumentVerdictError):
    """The file is password-protected/encrypted: readable only with its password."""

    verdict = "encrypted"


class CorruptDocumentError(DocumentVerdictError):
    """The container is damaged or is not the format its extension claims."""

    verdict = "corrupt"

def detect_file_encoding(filepath: str) -> str:
    """
    Detect the encoding of a file using BOM markers and chardet for broader support.
    Returns the detected encoding or 'utf-8' as default.
    """
    with open(filepath, "rb") as f:
        raw = f.read(4096)  # Read a larger sample for better detection

    # Check for BOM markers first
    if raw.startswith(codecs.BOM_UTF16_LE):
        return "utf-16-le"
    elif raw.startswith(codecs.BOM_UTF16_BE):
        return "utf-16-be"
    elif raw.startswith(codecs.BOM_UTF16):
        return "utf-16"
    elif raw.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    elif raw.startswith(codecs.BOM_UTF32_LE):
        return "utf-32-le"
    elif raw.startswith(codecs.BOM_UTF32_BE):
        return "utf-32-be"

    # Use chardet to detect encoding if no BOM is found
    result = chardet.detect(raw)
    encoding = result.get("encoding")
    if encoding:
        return encoding.lower()
    # Default to utf-8 if detection fails
    return "utf-8"


def cleanup_temp_encoding_file(loader) -> None:
    """
    Clean up temporary UTF-8 file if it was created for encoding conversion.

    :param loader: The document loader that may have created a temporary file
    """
    if hasattr(loader, "_temp_filepath") and loader._temp_filepath is not None:
        try:
            os.remove(loader._temp_filepath)
        except Exception as e:
            logger.warning(f"Failed to remove temporary UTF-8 file: {e}")


def get_loader(filename: str, file_content_type: str, filepath: str):
    """Get the appropriate document loader based on file type and\or content type."""
    file_ext = filename.split(".")[-1].lower()
    known_type = True

    # File Content Type reference:
    # ref.: https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/MIME_types/Common_types
    if file_ext == "pdf" or file_content_type == "application/pdf":
        loader = SafePyPDFLoader(filepath, extract_images=PDF_EXTRACT_IMAGES)
    elif file_ext == "csv" or file_content_type == "text/csv":
        # Detect encoding for CSV files
        encoding = detect_file_encoding(filepath)

        if encoding != "utf-8":
            # For non-UTF-8 encodings, convert to UTF-8 using streaming
            # to avoid holding the entire file in memory as a single string
            temp_file = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", suffix=".csv", delete=False
                ) as temp_file:
                    with open(
                        filepath, "r", encoding=encoding, errors="replace"
                    ) as original_file:
                        while True:
                            chunk = original_file.read(64 * 1024)
                            if not chunk:
                                break
                            temp_file.write(chunk)

                    temp_filepath = temp_file.name

                loader = CSVLoader(temp_filepath)
                loader._temp_filepath = temp_filepath
            except Exception as e:
                if temp_file and os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                raise e
        else:
            loader = CSVLoader(filepath)
    elif file_ext == "rst":
        loader = UnstructuredRSTLoader(filepath, mode="elements")
    elif file_ext == "xml" or file_content_type in [
        "application/xml",
        "text/xml",
        "application/xhtml+xml",
    ]:
        loader = UnstructuredXMLLoader(filepath)
    elif (
        file_ext == "pptx"
        or file_content_type
        == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    ):
        # Slide-level chunking: one Document per slide so the text splitter
        # keeps slide context intact (see slide_number/slide_title metadata).
        loader = SlidePowerPointLoader(filepath)
    elif file_ext == "ppt" or file_content_type == "application/vnd.ms-powerpoint":
        # Legacy binary .ppt is not supported by python-pptx; fall back.
        loader = UnstructuredPowerPointLoader(filepath)
    elif file_ext == "md" or file_content_type in [
        "text/markdown",
        "text/x-markdown",
        "application/markdown",
        "application/x-markdown",
    ]:
        loader = UnstructuredMarkdownLoader(filepath)
    elif file_ext == "epub" or file_content_type == "application/epub+zip":
        loader = UnstructuredEPubLoader(filepath)
    elif file_ext in ["doc", "docx"] or file_content_type in [
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]:
        loader = Docx2txtLoader(filepath)
    elif file_ext in ["xls", "xlsx"] or file_content_type in [
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ]:
        # SheetExcelLoader still parses through UnstructuredExcelLoader with
        # mode="elements", so each element keeps its sheet-level citation
        # (`page_name` = sheet name, `page_number` = sheet index); the default
        # ("single") mode would collapse the workbook into one Document with no
        # sheet metadata (KI-02 WP-C). The wrapper adds the encrypted-vs-corrupt
        # verdict and the uncached-formula status (KI-02 SP-01.5). The
        # `msoffcrypto` import UnstructuredExcelLoader performs at load time is
        # now satisfied: msoffcrypto-tool is a pinned requirement. Before that it
        # was absent from the image, so EVERY .xlsx failed with
        # ModuleNotFoundError regardless of mode.
        loader = SheetExcelLoader(filepath)
    elif file_ext == "json" or file_content_type == "application/json":
        loader = TextLoader(filepath, autodetect_encoding=True)
    elif file_ext in known_source_ext or (
        file_content_type and file_content_type.find("text/") >= 0
    ):
        loader = TextLoader(filepath, autodetect_encoding=True)
    else:
        loader = TextLoader(filepath, autodetect_encoding=True)
        known_type = False

    return loader, known_type, file_ext


def clean_text(text: str) -> str:
    """
    Clean up text from PDF lopader

    :param text: The original text
    :return: Cleaned text
    """
    text = remove_null(text)
    text = remove_non_utf8(text)
    return text


def remove_null(text: str) -> str:
    """
    Remove NUL (0x00) characters from a string.

    :param text: The original text with potential NUL characters.
    :return: Cleaned text without NUL characters.
    """
    return text.replace("\x00", "")


def remove_non_utf8(text: str) -> str:
    """
    Remove invalid UTF-8 characters from a string, such as surrogate characters

    :param text: The original text with potential invalid utf-8 characters
    :return: Cleaned text without invalid utf-8 characters.
    """
    try:
        return text.encode("utf-8", "ignore").decode("utf-8")
    except UnicodeError:
        return text


def process_documents(documents: List[Document]) -> str:
    processed_text = ""
    last_page: Optional[int] = None
    doc_basename = ""

    for doc in documents:
        if "source" in doc.metadata:
            doc_basename = doc.metadata["source"].split("/")[-1]
            break

    processed_text += f"{doc_basename}\n"

    for doc in documents:
        current_page = doc.metadata.get("page")
        # `is not None` (not a falsy check) so the 0-indexed FIRST page (page == 0)
        # gets its "# PAGE 0" marker; formats without page metadata (page == None,
        # e.g. pptx/docx/xlsx) still get no marker (KI-02 WP-C-F, MINOR-3).
        if current_page is not None and current_page != last_page:
            processed_text += f"\n# PAGE {doc.metadata['page']}\n\n"
            last_page = current_page

        new_content = doc.page_content
        if processed_text.endswith(new_content[:CHUNK_OVERLAP]):
            processed_text += new_content[CHUNK_OVERLAP:]
        else:
            processed_text += new_content

    return processed_text.strip()


class SafePyPDFLoader:
    """
    A wrapper around PyPDFLoader that handles image extraction failures gracefully.
    Falls back to text-only extraction when image extraction fails.

    This is a workaround for issues with PyPDFLoader that can occur when extracting images
    from PDFs, which can lead to KeyError exceptions if the PDF is malformed or has unsupported
    image formats. This class attempts to load the PDF with image extraction enabled, and if it
    fails due to a KeyError related to image filters, it falls back to loading the PDF
    without image extraction.
    ref.: https://github.com/langchain-ai/langchain/issues/26652
    """

    def __init__(self, filepath: str, extract_images: bool = False):
        self.filepath = filepath
        self.extract_images = extract_images
        self._temp_filepath = None  # For compatibility with cleanup function

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load PDF documents with automatic fallback on image extraction errors."""
        loader = PyPDFLoader(self.filepath, extract_images=self.extract_images)

        if not self.extract_images:
            # No image extraction: no fallback needed, stream directly
            yield from loader.lazy_load()
            return

        # extract_images=True: must collect eagerly so that a mid-stream
        # KeyError doesn't leave already-yielded pages duplicated by the
        # fallback (yield from + try/except would deliver partial + full).
        try:
            pages = list(loader.lazy_load())
        except KeyError as e:
            if "/Filter" in str(e):
                logger.warning(
                    f"PDF image extraction failed for {self.filepath}, falling back to text-only: {e}"
                )
                fallback_loader = PyPDFLoader(self.filepath, extract_images=False)
                pages = list(fallback_loader.lazy_load())
            else:
                # Re-raise if it's a different error
                raise
        yield from pages

    def load(self) -> List[Document]:
        """Load PDF documents with automatic fallback on image extraction errors."""
        return list(self.lazy_load())


class SheetExcelLoader:
    """Load a workbook as sheet-cited Documents, with honest terminal verdicts.

    Wraps `UnstructuredExcelLoader(..., mode="elements")` — kept because that is
    what carries the per-sheet citation metadata (`page_name` = sheet name,
    `page_number` = sheet index) the RAG pipeline must surface (KI-02 WP-C) —
    and adds the two things the raw loader cannot express:

    1. **Encrypted vs corrupt.** An encrypted OOXML workbook is an OLE2 compound
       file, byte-for-byte unlike the ZIP an .xlsx normally is, so `unstructured`
       reports both as the same exception type. A password-protected workbook is
       not damaged — the user only needs to supply an unprotected copy — so it
       gets its own `encrypted` verdict, decided by the declared `msoffcrypto`
       dependency rather than by matching a third-party error string.

    2. **Uncached formulas.** A workbook stores a formula AND the value Excel
       last computed for it. Files written by a library (or saved with
       calculation off) carry the formula with NO cached value, and the loader
       then extracts the row's label with the number silently missing — a
       "Total" line with no total. We never invent the value (computing it here
       would be a number the source does not contain); instead every Document of
       an affected sheet carries `formula_uncached` (count),
       `formula_uncached_cells` (bounded sample) and `formula_scan`, and the
       route lifts that into the extraction receipt. `formula_scan` is
       "unavailable" — never a silent zero — when the workbook cannot be re-read
       for the scan (e.g. legacy .xls, which openpyxl cannot open).
    """

    #: Compound File Binary header. An .xlsx is a ZIP; an ENCRYPTED .xlsx is an
    #: OLE2 container holding the encrypted package. Legacy .xls is also OLE2.
    _OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    _ZIP_MAGIC = b"PK\x03\x04"

    #: Cap on the per-sheet sample of uncached-formula cell references carried in
    #: metadata, so a pathological workbook cannot inflate every chunk's metadata.
    _MAX_REPORTED_CELLS = 25

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._temp_filepath = None  # For compatibility with cleanup function

    # -- verdicts -----------------------------------------------------------

    def _head(self, n: int = 8) -> bytes:
        try:
            with open(self.filepath, "rb") as f:
                return f.read(n)
        except OSError:
            return b""

    def _encrypted_error(self) -> "EncryptedDocumentError":
        return EncryptedDocumentError(
            "The workbook is password-protected, so its contents cannot be read. "
            "Upload a copy saved without a password.",
            filename=os.path.basename(self.filepath),
        )

    def _corrupt_error(self) -> "CorruptDocumentError":
        return CorruptDocumentError(
            "The file is damaged or is not a real Excel workbook, so no content "
            "could be read from it.",
            filename=os.path.basename(self.filepath),
        )

    def _precheck_container(self) -> None:
        """Decide encrypted vs corrupt from the container, before parsing.

        A workbook is one of exactly two containers: a ZIP (OOXML .xlsx) or an
        OLE2 compound file (legacy .xls — and also what an ENCRYPTED .xlsx is).
        So the header alone separates the cases the parser cannot:

        * ZIP        -> nothing to decide here; parse normally.
        * OLE2       -> ask msoffcrypto. Encrypted gives the `encrypted` verdict;
                        a container it cannot even open is a damaged file, which
                        is exactly what the parser would fail on next anyway.
        * neither    -> not a workbook at all.

        A missing msoffcrypto is the one case that must NOT become a verdict: the
        parser's own import error is the truthful report, so we fall through.
        """
        head = self._head()
        if head.startswith(self._ZIP_MAGIC):
            return
        if not head.startswith(self._OLE2_MAGIC):
            raise self._corrupt_error()
        try:
            import msoffcrypto
        except ImportError as e:  # pragma: no cover - msoffcrypto is a requirement
            logger.warning(
                "msoffcrypto unavailable (%s); deferring the encryption check to "
                "the parser",
                e,
            )
            return
        try:
            with open(self.filepath, "rb") as f:
                encrypted = msoffcrypto.OfficeFile(f).is_encrypted()
        except Exception as e:  # noqa: BLE001 - an unreadable OLE container is damaged
            logger.info("Unreadable OLE container for %s: %s", self.filepath, e)
            raise self._corrupt_error() from e
        if encrypted:
            raise self._encrypted_error()

    def _translate(self, error: Exception) -> Exception:
        """Map a parser failure to a terminal verdict.

        The container pre-check already decided the cases it can see from the
        header; this is the second line, for damage that only shows up once the
        package is opened (a ZIP that is not an OOXML workbook, a truncated
        sheet part). Anything we cannot classify is re-raised unchanged rather
        than labelled — a mislabelled failure would be worse than a generic one.
        """
        text = str(error).lower()
        if "password" in text or "encrypt" in text:
            return self._encrypted_error()
        if isinstance(error, zipfile.BadZipFile) or "not a valid" in text:
            return self._corrupt_error()
        return error

    # -- uncached formulas --------------------------------------------------

    def _uncached_formulas(self):
        """Return ``(per_sheet_cells, scan_status)`` for formulas with no cached value.

        Two streaming (`read_only`) passes over the same workbook: one keeping
        formulas, one keeping the cached results. A cell whose formula pass holds
        a formula string while its value pass holds ``None`` has no stored result.
        """
        if not self._head(4).startswith(self._ZIP_MAGIC):
            # Legacy .xls (or anything not an OOXML package): openpyxl cannot
            # read it, so we must not claim a clean scan.
            return {}, "unavailable"
        try:
            from openpyxl import load_workbook

            per_sheet = {}
            formulas = load_workbook(self.filepath, data_only=False, read_only=True)
            try:
                values = load_workbook(self.filepath, data_only=True, read_only=True)
                try:
                    for name in formulas.sheetnames:
                        missing = []
                        f_rows = formulas[name].iter_rows()
                        v_rows = values[name].iter_rows()
                        for f_row, v_row in zip(f_rows, v_rows):
                            for f_cell, v_cell in zip(f_row, v_row):
                                if (
                                    isinstance(f_cell.value, str)
                                    and f_cell.value.startswith("=")
                                    and v_cell.value is None
                                ):
                                    missing.append(f_cell.coordinate)
                        if missing:
                            per_sheet[name] = missing
                finally:
                    values.close()
            finally:
                formulas.close()
            return per_sheet, "complete"
        except Exception as e:  # noqa: BLE001 - the scan is diagnostic, never fatal
            logger.warning("Uncached-formula scan failed for %s: %s", self.filepath, e)
            return {}, "unavailable"

    def _annotate(self, documents: List[Document]) -> List[Document]:
        per_sheet, scan_status = self._uncached_formulas()
        for doc in documents:
            doc.metadata["formula_scan"] = scan_status
            cells = per_sheet.get(doc.metadata.get("page_name"), [])
            if cells:
                doc.metadata["formula_uncached"] = len(cells)
                doc.metadata["formula_uncached_cells"] = cells[
                    : self._MAX_REPORTED_CELLS
                ]
        return documents

    # -- loader interface ---------------------------------------------------

    def load(self) -> List[Document]:
        self._precheck_container()
        inner = UnstructuredExcelLoader(self.filepath, mode="elements")
        try:
            documents = inner.load()
        except Exception as e:
            raise self._translate(e) from e
        return self._annotate(documents)

    def lazy_load(self) -> Iterator[Document]:
        # The annotation pass needs the whole sheet set, and `unstructured`
        # materializes the workbook anyway, so there is nothing to stream.
        yield from self.load()


class SlidePowerPointLoader:
    """
    Load a .pptx deck as one Document per slide, preserving slide context.

    Unlike UnstructuredPowerPointLoader (which collapses the whole deck into a
    single text stream), this yields a Document per slide so the downstream
    text splitter never merges content across slide boundaries. Each Document
    carries `slide_number` and `slide_title` metadata, which propagates to every
    resulting chunk.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._temp_filepath = None  # For compatibility with cleanup function

    @staticmethod
    def _slide_title(slide) -> str:
        title_shape = slide.shapes.title
        if title_shape is not None and title_shape.has_text_frame:
            return title_shape.text.strip()
        return ""

    @staticmethod
    def _table_text(table) -> str:
        """Row-major text of a PPTX table: cells joined by ' | ', rows by newline.

        Evidence often sits in a table cell (KI-02 WP-C goal). Empty rows are
        dropped so a spacer row does not add blank lines.
        """
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        return "\n".join(rows)

    @staticmethod
    def _chart_text(chart) -> str:
        """Best-effort chart labels: title, category labels and series names.

        Evidence can sit in a chart label (KI-02 WP-C goal). Chart XML is
        variable, so every access is guarded — a chart we cannot read must be
        skipped, never crash the whole deck.
        """
        parts: List[str] = []
        try:
            if chart.has_title and chart.chart_title.text_frame.text.strip():
                parts.append(chart.chart_title.text_frame.text.strip())
        except Exception:  # noqa: BLE001 - chart metadata is untrusted/variable
            pass
        try:
            for plot in chart.plots:
                try:
                    cats = [str(c).strip() for c in plot.categories if str(c).strip()]
                    if cats:
                        parts.append(" ".join(cats))
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        try:
            for series in chart.series:
                try:
                    name = (series.name or "").strip()
                    if name:
                        parts.append(name)
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        return "\n".join(parts)

    @classmethod
    def _collect_shape_texts(cls, shapes) -> List[str]:
        """Walk a slide's shape tree and collect text from every text-bearing
        shape: plain text frames, tables, charts, and (recursively) grouped
        shapes. Without this, evidence inside a group, table or chart label is
        silently dropped (KI-02 WP-C)."""
        from pptx.enum.shapes import MSO_SHAPE_TYPE

        out: List[str] = []
        for shape in shapes:
            try:
                if getattr(shape, "shape_type", None) == MSO_SHAPE_TYPE.GROUP:
                    out.extend(cls._collect_shape_texts(shape.shapes))
                    continue
                if getattr(shape, "has_table", False):
                    text = cls._table_text(shape.table)
                    if text:
                        out.append(text)
                    continue
                if getattr(shape, "has_chart", False):
                    text = cls._chart_text(shape.chart)
                    if text:
                        out.append(text)
                    continue
                if getattr(shape, "has_text_frame", False) and shape.text.strip():
                    out.append(shape.text.strip())
            except Exception as e:  # noqa: BLE001 - never let one shape abort a deck
                logger.warning(
                    "Skipped a shape while extracting PPTX text: %s", e
                )
        return out

    @classmethod
    def _has_picture(cls, shapes) -> bool:
        """True if the shape tree contains at least one picture (recursing into
        groups). Distinguishes an image-only slide (a picture but no extractable
        text — kept identifiable) from a truly blank slide (no shapes — dropped).
        KI-02 WP-C-F, MAJOR-1."""
        from pptx.enum.shapes import MSO_SHAPE_TYPE

        picture_types = {MSO_SHAPE_TYPE.PICTURE, MSO_SHAPE_TYPE.LINKED_PICTURE}
        for shape in shapes:
            try:
                stype = getattr(shape, "shape_type", None)
                if stype in picture_types:
                    return True
                if stype == MSO_SHAPE_TYPE.GROUP and cls._has_picture(shape.shapes):
                    return True
            except Exception as e:  # noqa: BLE001 - never let one shape abort a deck
                logger.warning(
                    "Skipped a shape while detecting PPTX pictures: %s", e
                )
        return False

    def lazy_load(self) -> Iterator[Document]:
        from pptx import Presentation

        prs = Presentation(self.filepath)
        for idx, slide in enumerate(prs.slides, start=1):
            title = self._slide_title(slide)

            texts = self._collect_shape_texts(slide.shapes)

            if (
                slide.has_notes_slide
                and slide.notes_slide.notes_text_frame is not None
            ):
                note = slide.notes_slide.notes_text_frame.text.strip()
                if note:
                    texts.append(f"[Notes] {note}")

            content = "\n".join(texts).strip()
            if not content:
                # An image-only slide (a picture but no extractable text) must
                # stay identifiable rather than vanish — parity with PDF scan
                # pages, which emit an empty-but-locatable Document. The text is
                # EMPTY (not a placeholder string) on purpose: an all-image deck
                # then yields only empty content and is still rejected by the
                # per-file empty-extraction guard (422), while a mixed deck still
                # cites this slide's slide_number. A TRULY blank slide (no shapes
                # at all) is still dropped, so blank spacer slides never renumber
                # the deck. KI-02 WP-C-F, MAJOR-1.
                if self._has_picture(slide.shapes):
                    yield Document(
                        page_content="",
                        metadata={
                            "source": self.filepath,
                            "slide_number": idx,
                            "slide_title": title,
                            "image_only": True,
                        },
                    )
                continue

            yield Document(
                page_content=content,
                metadata={
                    "source": self.filepath,
                    "slide_number": idx,
                    "slide_title": title,
                },
            )

    def load(self) -> List[Document]:
        return list(self.lazy_load())
