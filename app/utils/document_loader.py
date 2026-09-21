# app/utils/document_loader.py

import os
import codecs
import csv
import tempfile
import zipfile

from typing import Iterator, List, Optional
import chardet
from pypdf import PdfReader

from langchain_core.documents import Document

from app.utils.extraction_budget import (
    ATTEMPTED_KEY,
    NOT_INCLUDED_KEY,
    STOPPED_KEY,
)
from app.config import (
    known_source_ext,
    PDF_EXTRACT_IMAGES,
    PDF_OCR_ENABLED,
    CHUNK_OVERLAP,
    logger,
)
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


class UnsupportedDocumentError(DocumentVerdictError):
    """The file is intact and readable — we simply have no extractor for this format."""

    verdict = "unsupported"


#: Leading byte signatures for formats this service has no text extractor for. Named rather than
#: lumped into "binary" because "we do not read images" is actionable and "unsupported file" is not.
_BINARY_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "a PNG image"),
    (b"\xff\xd8\xff", "a JPEG image"),
    (b"GIF87a", "a GIF image"),
    (b"GIF89a", "a GIF image"),
    (b"BM", "a BMP image"),
    (b"PK\x03\x04", "a ZIP archive"),
    (b"Rar!\x1a\x07", "a RAR archive"),
    (b"7z\xbc\xaf\x27\x1c", "a 7-Zip archive"),
    (b"\x1f\x8b", "a gzip archive"),
    (b"ID3", "an MP3 audio file"),
    (b"OggS", "an Ogg media file"),
    (b"fLaC", "a FLAC audio file"),
    (b"\x00\x00\x00\x18ftyp", "an MP4 video file"),
    (b"\x00\x00\x00\x20ftyp", "an MP4 video file"),
    (b"MZ", "a Windows executable"),
    (b"\x7fELF", "a Linux executable"),
    (b"SQLite format 3\x00", "a SQLite database"),
)


def describe_unsupported_binary(head: bytes) -> Optional[str]:
    """Name the format behind `head`, or None when nothing recognisable matches.

    RIFF containers carry their real type four bytes in (WAVE / AVI / WEBP), so they are checked
    separately rather than given a misleading generic name.
    """
    if head.startswith(b"RIFF") and len(head) >= 12:
        return {
            b"WAVE": "a WAV audio file",
            b"AVI ": "an AVI video file",
            b"WEBP": "a WebP image",
        }.get(head[8:12], "a RIFF media file")
    if head.startswith(b"PK\x03\x04"):
        # Office and OpenDocument files ARE zips. Saying "a ZIP archive" to someone whose .docx
        # reached us with the wrong extension or content type is true and useless — it sends them
        # looking for an archive they never made. The package declares itself in the first entry.
        if b"[Content_Types].xml" in head:
            return "an Office file (.docx/.xlsx/.pptx) that arrived with the wrong file name or type"
        if b"mimetypeapplication/vnd.oasis.opendocument" in head:
            return "an OpenDocument file that arrived with the wrong file name or type"
        return "a ZIP archive"
    for signature, description in _BINARY_SIGNATURES:
        if head.startswith(signature):
            return description
    return None


#: Byte-order marks this service already honours elsewhere (`detect_file_encoding`). A BOM is an
#: explicit declaration that the file is text, and it outranks every heuristic below.
_TEXT_BOMS = (
    codecs.BOM_UTF8,
    codecs.BOM_UTF32_LE,
    codecs.BOM_UTF32_BE,
    codecs.BOM_UTF16_LE,
    codecs.BOM_UTF16_BE,
)


def decodes_as_utf8(sample: bytes) -> bool:
    """Whether `sample` is valid UTF-8 TEXT, tolerating a code point cut by the fixed-size read.

    A NUL disqualifies it. NUL is a perfectly valid UTF-8 code point, so a bare `decode()` call
    happily accepts a run of zero bytes and reports a WAV header — or any NUL-padded binary — as
    text. Real text does not contain NUL; the encodings that do (UTF-16/32) are recognised by their
    BOM before this, or by `decodes_as_utf16` after the signature check.
    """
    if b"\x00" in sample:
        return False
    for trim in range(4):
        candidate = sample[: len(sample) - trim] if trim else sample
        try:
            candidate.decode("utf-8")
            return True
        except UnicodeDecodeError:
            continue
    return False


def decodes_as_utf16(sample: bytes) -> bool:
    """Whether `sample` is UTF-16 text without a BOM.

    UTF-16 encodes ASCII as alternating character/NUL bytes, so a NUL check alone condemns it — and
    UTF-16 is text this service is expected to read (`detect_file_encoding` handles its BOMs). An
    arbitrary binary will usually also "decode" as UTF-16 into nonsense, so decoding is not enough on
    its own: the result must also be overwhelmingly printable.
    """
    nul_positions = [index for index, byte in enumerate(sample) if byte == 0]
    # UTF-16 text in a Latin script is about half NUL bytes, and they sit consistently on the odd
    # (little-endian) or even (big-endian) offsets. Requiring that alignment is what separates real
    # UTF-16 from arbitrary binary: any byte run "decodes" as UTF-16 into CJK-looking characters that
    # `str.isprintable()` happily accepts, so printability alone would wave binaries through.
    if len(nul_positions) < len(sample) * 0.25:
        return False
    odd = sum(1 for index in nul_positions if index % 2)
    if odd not in (0, len(nul_positions)):
        return False

    even = sample[: len(sample) - (len(sample) % 2)]
    for encoding in ("utf-16-le", "utf-16-be"):
        try:
            text = even.decode(encoding)
        except UnicodeDecodeError:
            continue
        if not text:
            continue
        printable = sum(1 for ch in text if ch.isprintable() or ch in "\r\n\t")
        if printable / len(text) > 0.9:
            return True
    return False


def looks_like_text(sample: bytes, before_signatures: bool) -> bool:
    """Whether `sample` should be read as text.

    Called twice by `raise_if_unsupported_binary`, around the signature check, because the two halves
    have different strength and the order matters:

    * `before_signatures=True` — only EVIDENCE that outranks a magic number: a BOM, or valid UTF-8.
      This is what keeps a note beginning "MZ is the DOS header magic" from being refused as a
      Windows executable. `MZ` and `BM` are ordinary English bigrams and `ID3`/`GIF87a` occur in
      technical prose, so a two-byte prefix must never beat the file actually decoding.
    * `before_signatures=False` — the weaker evidence, consulted only once no signature matched:
      BOM-less UTF-16, then chardet. chardet is last on purpose. Single-byte encodings decode ANY
      byte sequence, so a confident-looking guess would wave real binaries through — a PNG does
      exactly that.

    Deciding by DECODING rather than by counting printable bytes is the other half. A ratio test
    looks reasonable until you feed it Japanese or Arabic: in UTF-8 every byte of those scripts is
    >= 0x80, so a byte-counting heuristic calls a perfectly good document binary and refuses it.
    """
    if not sample:
        return True
    if before_signatures:
        return sample.startswith(_TEXT_BOMS) or decodes_as_utf8(sample)

    if decodes_as_utf16(sample):
        return True
    detected = chardet.detect(sample)
    encoding, confidence = detected.get("encoding"), detected.get("confidence") or 0
    if encoding and confidence >= 0.7:
        try:
            sample.decode(encoding, errors="strict")
        except (UnicodeDecodeError, LookupError):
            return False
        # A single-byte codec cannot fail, so "it decoded" is only meaningful for a codec that can.
        return encoding.lower().startswith(("utf", "iso-8859", "windows-125", "cp"))
    return False


def looks_like_binary(sample: bytes) -> bool:
    """Whether `sample` cannot be read as text.

    Mirrors `raise_if_unsupported_binary`'s decision exactly — including consulting the signature
    table in the middle — so the predicate and the refusal can never disagree about the same bytes.
    """
    if looks_like_text(sample, before_signatures=True):
        return False
    if describe_unsupported_binary(sample) is not None:
        return True
    return not looks_like_text(sample, before_signatures=False)


def raise_if_unsupported_binary(filepath: str, filename: str) -> None:
    """Refuse a file we have no extractor for, instead of reading its bytes as prose.

    The fallback branch of `get_loader` hands anything unrecognised to `TextLoader`, and nothing
    downstream refuses on `known_type=False` — it is only reported. So a ZIP used to arrive as one
    Document of container framing plus fragments of its members, pass the empty-extraction guard and
    be stored with an extraction receipt reading `complete`. That is a garbage extraction counted as
    a success, the sibling of the empty-extraction defect WP-C closed, and the uploader was told
    nothing. An image fared differently but no better: `Could not detect encoding`, which names our
    internals rather than their problem.

    The order below is the whole design, and it is what stops the refusal being over-eager:
    decode-as-text evidence that outranks a magic number, THEN signatures, THEN the weaker text
    evidence. Getting it wrong in either direction is a real loss — refuse a document that used to
    ingest, or wave a binary through.
    """
    try:
        with open(filepath, "rb") as handle:
            sample = handle.read(8192)
    except OSError:
        # Unreadable here is not evidence of anything; let the loader open it and report honestly.
        return

    if looks_like_text(sample, before_signatures=True):
        return

    described = describe_unsupported_binary(sample)
    if described is None:
        if looks_like_text(sample, before_signatures=False):
            return
        described = "not a text-based format"

    raise UnsupportedDocumentError(
        f"'{filename}' is {described}, which this service cannot read as text. Upload a document "
        f"format instead — PDF, Word, PowerPoint, Excel, CSV or plain text.",
        filename=filename,
    )


#: OLE2 compound-file magic. This is the container the pre-2007 Office binaries use (.doc, .xls,
#: .ppt) and also what an ENCRYPTED OOXML file is. `SheetExcelLoader` reads it to separate encrypted
#: from corrupt; the Word branch reads it to refuse a legacy .doc with a verdict instead of letting
#: `docx2txt` die on the zip check.
OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def refuse_legacy_word_binary(filepath: str, filename: str) -> None:
    """Refuse a pre-2007 binary `.doc` with a verdict, before `Docx2txtLoader` sees it.

    `.doc` and `.docx` share one branch, and that branch hands both to `Docx2txtLoader`. But
    `docx2txt` reads an OOXML package -- a ZIP -- while a Word 97 `.doc` is an OLE2 compound file,
    so the legacy half of the branch could only ever fail. MEASURED on 36d4fb6 with a genuine Word
    97 document (LibreOffice Writer 25.2.3.2, filter "MS Word 97"): `get_loader` returned
    `Docx2txtLoader` with `known_type=True`, `load()` raised `zipfile.BadZipFile("File is not a zip
    file")`, and `/embed` answered 400 "The cause is not established - it may be the file or this
    service." with zero rows.

    Nothing was stored and nothing claimed to succeed, so this is not the garbage-extraction defect
    SP-01.6a closed. It is the other half of that charter: the service can tell exactly what this
    file is, and said it could not tell. `attribution=undetermined` on a format we positively
    recognise is a non-answer to someone who can fix their file in ten seconds.

    This is ALSO the gap #42 left explicitly open. That guard turns a missing-LibreOffice failure
    into an actionable 400, but it fires on `OSError("soffice command was not found")` -- and a
    `.doc` never reaches soffice at all, because the Word branch claims it first and dies in
    `zipfile`. So `.ppt` gets the actionable answer and `.doc` cannot, which is why #42's comment
    records that `.doc` "was never tested, so it is not claimed either".

    The message deliberately does NOT offer the LibreOffice operator fix that #42's does. Installing
    LibreOffice would not make THIS path work: `Docx2txtLoader` would still be handed the same OLE2
    bytes. Claiming it would is the same false family claim #42 removed once already, and this lane
    does not re-add it on the strength of a neighbouring format's behaviour.

    A `.docx` is untouched: it is a ZIP, so the header never matches.
    """
    try:
        with open(filepath, "rb") as f:
            head = f.read(len(OLE2_MAGIC))
    except OSError:
        # Unreadable here means unreadable for the loader a line later, and its error is the
        # truthful report. Never convert an I/O fault into a verdict about the FORMAT.
        return
    if not head.startswith(OLE2_MAGIC):
        return
    raise UnsupportedDocumentError(
        f"'{filename}' is a legacy Word document (the pre-2007 .doc format), which this service "
        f"cannot read. Open it in Word and re-save it as .docx, then upload that.",
        filename=filename,
    )


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


def get_loader(
    filename: str,
    file_content_type: str,
    filepath: str,
    ocr_budget=None,
    extraction_budget=None,
):
    """Get the appropriate document loader based on file type and\or content type.

    `ocr_budget` is the caller's bounded allowance for reading SCANNED PDF pages
    locally (FILES-01). It is optional and PDF-only: omitting it leaves every loader
    behaving exactly as it did before OCR existed.

    `extraction_budget` bounds how much of a NATIVE PDF is read at all (FILES-01). Also
    optional, also PDF-only, and OFF unless an operator configures a limit -- so omitting
    it, or passing an unconfigured budget, changes nothing about cost or behaviour.
    """
    file_ext = filename.split(".")[-1].lower()
    known_type = True

    # File Content Type reference:
    # ref.: https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/MIME_types/Common_types
    if file_ext == "pdf" or file_content_type == "application/pdf":
        loader = SafePyPDFLoader(
            filepath,
            extract_images=PDF_EXTRACT_IMAGES,
            ocr_budget=ocr_budget,
            extraction_budget=extraction_budget,
        )
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

                loader = RowCSVLoader(temp_filepath)
                loader._temp_filepath = temp_filepath
            except Exception as e:
                if temp_file and os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                raise e
        else:
            loader = RowCSVLoader(filepath)
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
        # A legacy binary .doc can never be read by docx2txt (see refuse_legacy_word_binary).
        refuse_legacy_word_binary(filepath, filename)
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
        # Nothing above claimed this file. Before treating it as prose, check that it actually IS
        # text: this branch is the only one reached by a format we have no extractor for, so it is
        # the only place the check belongs. A recognised binary — or anything that does not read as
        # text — gets an honest `unsupported` verdict naming the format, instead of having its bytes
        # embedded. Everything else still falls through to TextLoader exactly as before, so a genuine
        # text file with an unusual extension is unaffected.
        raise_if_unsupported_binary(filepath, filename)
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

    def __init__(
        self,
        filepath: str,
        extract_images: bool = False,
        ocr_budget=None,
        extraction_budget=None,
    ):
        self.filepath = filepath
        self.extract_images = extract_images
        self._temp_filepath = None  # For compatibility with cleanup function
        #: How much of this document may be READ at all (FILES-01). Separate from the OCR
        #: allowance on purpose: that one caps work we choose to do on a page, this one caps
        #: how far into the file we go. `None`, or a budget with neither bound configured,
        #: leaves the previous behaviour and cost exactly unchanged.
        self.extraction_budget = extraction_budget
        #: One document's OCR allowance, supplied by the route. `None` means "do not
        #: OCR at all" -- so every existing direct construction of this loader keeps
        #: exactly its previous behaviour and cost.
        self.ocr_budget = ocr_budget
        self._ocr_reader_obj = None
        self._ocr_handle = None

    #: A PDF can be "encrypted" in two very different ways and only ONE of them is a refusal.
    #: Measured with pypdf in the shipped image, not assumed:
    #:
    #:   user password       is_encrypted=True, decrypt("") -> 0 (NOT_DECRYPTED), pages UNREADABLE
    #:   owner password ONLY is_encrypted=True, decrypt("") -> 1, pages READ PERFECTLY WELL
    #:
    #: Owner-password PDFs carry only usage restrictions (no printing, no copying) and extract
    #: today. Refusing on `is_encrypted` alone would reject a whole class of files that currently
    #: work -- which is why the test is "the empty password does not unlock it", not "it is
    #: encrypted".
    def _refuse_if_locked(self) -> None:
        """Give a password-protected PDF the same actionable verdict a workbook already gets.

        FILES-01, approved as a small maintenance change: a locked PDF used to answer the generic
        400 "the cause is not established" while an encrypted .xlsx answered 422 `encrypted` with
        an instruction the uploader can act on. Same condition, same contract now.

        Deliberately NOT done, and not wanted: storing passwords, prompting for them, or trying to
        break them. This only names what is wrong.
        """
        try:
            reader = PdfReader(self.filepath)
            if not reader.is_encrypted:
                return
            unlocked = reader.decrypt("")
        except Exception as probe_error:
            # The check itself failed. Say nothing rather than invent a verdict from a probe that
            # did not work -- the parser below produces its own honest answer, i.e. we degrade to the
            # behaviour that shipped before this check existed.
            #
            # No `except DocumentVerdictError: raise` guard here: review showed it was unreachable.
            # Nothing inside this `try` raises one, and the EncryptedDocumentError below is raised
            # OUTSIDE it, so it can never be swallowed. An uncovered guard that implies a tested path
            # is worse than no guard.
            #
            # `info`, not `debug`, to match SheetExcelLoader._precheck_container: an inconclusive
            # pre-check means the caller may get a vaguer message than we could have given, which an
            # operator should be able to see.
            logger.info(
                "PDF encryption pre-check inconclusive for %s: %s", self.filepath, probe_error
            )
            return

        if not unlocked:
            raise EncryptedDocumentError(
                "The PDF is password-protected, so its contents cannot be read. "
                "Upload a copy saved without a password.",
                filename=os.path.basename(self.filepath),
            )

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load PDF documents with automatic fallback on image extraction errors."""
        # Decide the encrypted verdict BEFORE parsing, the same way SheetExcelLoader decides
        # encrypted-vs-corrupt from the container first.
        self._refuse_if_locked()
        loader = PyPDFLoader(self.filepath, extract_images=self.extract_images)

        if not self.extract_images:
            # No image extraction: no fallback needed, stream directly.
            # The read bound is applied BEFORE OCR so a page that will be dropped is never
            # OCR'd -- otherwise the expensive work would happen and then be thrown away.
            yield from self._with_ocr(self._within_budget(loader.lazy_load()))
            return

        # extract_images=True: must collect eagerly so that a mid-stream
        # KeyError doesn't leave already-yielded pages duplicated by the
        # fallback (yield from + try/except would deliver partial + full).
        #
        # THE BOUND WRAPS THE PRODUCER, NOT THE RESULT. Applying it to an already-materialised
        # list -- which is what this did before review -- costs the full parse of every page and
        # then throws the surplus away: measured at 10 pages parsed for a 3-page bound with
        # `extract_images=True`, i.e. the exact exhaustion this exists to prevent, plus seven
        # pages of data loss for nothing. Wrapping the generator means the pages past the bound
        # are never produced.
        try:
            pages = list(self._within_budget(loader.lazy_load()))
        except KeyError as e:
            if "/Filter" in str(e):
                logger.warning(
                    f"PDF image extraction failed for {self.filepath}, falling back to text-only: {e}"
                )
                fallback_loader = PyPDFLoader(self.filepath, extract_images=False)
                # A FRESH budget. The first attempt spent the original one, and a spent budget
                # yields nothing at all -- which would turn a recoverable image-extraction failure
                # into an empty document. The fallback is a retry of the same work, so it gets the
                # same allowance, not the remains of the allowance the failed attempt consumed.
                pages = list(
                    self._within_budget(
                        fallback_loader.lazy_load(), budget=self._fresh_budget()
                    )
                )
            else:
                # Re-raise if it's a different error
                raise
        yield from self._with_ocr(iter(pages))

    # -- bounded reading of a native PDF (FILES-01) ---------------------------------
    #
    # Nothing else in the chain protects this service from a large native PDF: the edge
    # permits more than the service can parse before every timeout above it has expired,
    # and an unbounded parse keeps allocating for a response no caller is waiting for. A
    # worker killed for memory cannot deliver the honest failure contract at all -- the
    # uploader just sees a dropped connection.
    #
    # Stopping is not truncating silently: the pages already read are KEPT, the LAST of
    # them carries which bound stopped the work and how many pages were never opened, and
    # the receipt turns that into `partial`. A stopped read is never `complete`, and -- the
    # case that matters most -- never `empty`, which is a refusal meaning something else.

    def _remaining_page_count(self, pages_read: int):
        """How many pages were never opened. `None` when the file's own page count cannot be
        established -- reported as unknown rather than guessed as zero.

        The file is re-opened only on the truncation path, so an unbounded read (and every
        read that finishes inside its bounds) pays nothing for this.

        WHAT THE TRUNCATION PATH PAYS, measured 2026-09-20 rather than reasoned about. Raised in
        review as a SUSPECTED finding and left unmeasured by both the reviewer and me until now;
        300-page fixture, median of 5, inside the test image:

            healthy xref   open + 2 pages = 21.5 ms   this call adds 16.9 ms   (+79%)
            damaged xref   open + 2 pages = 41.9 ms   this call adds 43.2 ms  (+103%)

        So this call is ~100% of the overhead a bounded read adds over the floor, and on a file
        whose cross-reference table is damaged pypdf rebuilds it by scanning, which roughly
        doubles that again.

        THE BOUND STILL BOUNDS -- that was worth checking before calling this a defect. Against
        reading all 300 pages the bounded read saves 56% (healthy) and 29% (damaged). The first
        version of this measurement compared bounded against unbounded with no FLOOR and read
        "68% of unbounded" as "the bound is not bounding"; most of that 68% is the unavoidable
        cost of opening a 300-page document at all, which no bound can avoid.

        WHY IT IS STILL DONE THIS WAY. The alternative -- counting pages up front -- moves the
        cost onto EVERY read, including the unbounded ones that are the common case, to benefit
        the truncated ones that are rare. Charging the rare path is the better trade. It is
        written down here so the next person does not have to re-measure it, and so the trade is
        visible as a choice rather than looking like an oversight.
        """
        try:
            with open(self.filepath, "rb") as handle:
                total = len(PdfReader(handle).pages)
        except Exception as error:  # a malformed tail is exactly when this can fail
            logger.info("Could not count pages of %s: %s", self.filepath, error)
            return None
        return max(0, total - pages_read)

    def _fresh_budget(self):
        """A new budget with the same limits and none of the spending.

        Only the retry path needs this: a budget counts pages for ONE pass over ONE document, and
        handing a second pass the remains of the first makes the retry read less than the operator
        configured -- or, once the first pass has used the whole allowance, nothing at all.
        """
        budget = self.extraction_budget
        if budget is None:
            return None
        return ExtractionBudget(
            max_pages=budget.max_pages,
            time_budget_seconds=budget.time_budget_seconds,
        )

    def _within_budget(self, pages: Iterator[Document], budget=None) -> Iterator[Document]:
        """Yield pages while the read budget allows, stamping the last one when it does not."""
        budget = budget if budget is not None else self.extraction_budget
        if budget is None or not budget.enabled:
            yield from pages
            return

        held = None
        stopped = False
        for page in pages:
            if not budget.may_read_page():
                stopped = True
                break
            if held is not None:
                yield held
            held = page

        if held is None and stopped:
            # A budget that was already spent before the first page: nothing to yield, and -- far
            # worse -- nothing to STAMP, so the receipt would say `empty` with no `extraction_bound`
            # at all and the caller would be told their readable file has no text. Unreachable on
            # the live path (each load builds its own budget, and the retry above takes a fresh
            # one), which is exactly why it must be loud rather than silent: the day it becomes
            # reachable, a crash naming the cause is honest and an empty receipt is not.
            raise RuntimeError(
                "extraction budget was already spent before this pass began "
                "(stopped_reason=%s, pages_read=%d): a budget counts one pass over one "
                "document -- build a fresh ExtractionBudget per load."
                % (budget.stopped_reason, budget.pages_read)
            )

        if held is not None:
            if stopped:
                # Stamped on the last page actually read, so the receipt can find it without
                # the loader having to invent a synthetic page for the ones it never opened.
                not_included = self._remaining_page_count(budget.pages_read)
                held.metadata[STOPPED_KEY] = budget.stopped_reason
                held.metadata[ATTEMPTED_KEY] = budget.pages_read
                held.metadata[NOT_INCLUDED_KEY] = not_included
                # "not included", not "not opened": the generator holds a page back so it can stamp
                # the last one it keeps, so one page beyond the bound was pulled from the producer
                # and discarded. The count is right for "absent from this receipt" and was wrong by
                # one for "never opened", which is what the sentence used to say.
                logger.warning(
                    "Stopped reading %s after %d page(s): %s (%s page(s) not included)",
                    self.filepath,
                    budget.pages_read,
                    budget.stopped_reason,
                    "unknown" if not_included is None else not_included,
                )
            yield held

    # -- local-first OCR (FILES-01, FS-CONTINUE-R3) ---------------------------------
    #
    # Native text FIRST, always: a page pypdf could read is never sent to OCR, so a
    # native PDF costs exactly what it cost before and a mixed document never gets the
    # same page twice (the one duplication risk this design has to avoid). OCR runs
    # ONLY on pages that produced no usable text -- which is precisely a scan.
    #
    # `PDF_EXTRACT_IMAGES` is NOT this switch. Measured in the shipped image,
    # `PyPDFLoader(extract_images=True)` returns 0 characters on a genuine scan even
    # with langchain's own OCR image parser attached, so the page image is fetched
    # here explicitly instead.

    def _ocr_reader(self):
        """The pypdf reader used for OCR, opened at most once and only if needed.

        Opened lazily so a native PDF never pays for it, and kept on an explicit file
        handle that `close_ocr_reader` releases rather than relying on GC.
        """
        if self._ocr_reader_obj is None:
            self._ocr_handle = open(self.filepath, "rb")
            reader = PdfReader(self._ocr_handle)
            if reader.is_encrypted:
                # Re-applies to THIS reader what `_refuse_if_locked` already established:
                # the empty password opens the file. An owner-password scan is a readable
                # scan and must be OCR'd like any other.
                #
                # Not "a user-password PDF never reaches here" -- review was right that
                # the claim was too absolute. If the pre-check's own probe fails, it stays
                # silent by design and a locked file can arrive here. There is no bypass:
                # `decrypt("")` then returns NOT_DECRYPTED, reading the page fails, the
                # failure is caught and the page is reported empty. Nothing is stored and
                # no password is guessed.
                reader.decrypt("")
            self._ocr_reader_obj = reader
        return self._ocr_reader_obj

    def close_ocr_reader(self) -> None:
        self._ocr_reader_obj = None
        if self._ocr_handle is not None:
            try:
                self._ocr_handle.close()
            finally:
                self._ocr_handle = None

    def _with_ocr(self, pages: Iterator[Document]) -> Iterator[Document]:
        """Pass native pages through untouched; OCR the ones that came back empty."""
        from app.utils.ocr import OcrCancelled, ocr_page  # local: keeps OCR off the import path

        budget = self.ocr_budget
        try:
            for document in pages:
                metadata = document.metadata if document.metadata is not None else {}
                document.metadata = metadata
                if (document.page_content or "").strip():
                    # Real text on the page. Never OCR it -- that is what would produce
                    # duplicate text for a mixed document.
                    #
                    # The provenance stamp is gated on the feature being ON. Review found
                    # it was being written unconditionally, so a NATIVE PDF gained a
                    # `text_sources` block in its receipt even with OCR disabled -- which
                    # made the "byte-identical to the pre-OCR build" claim false for the
                    # one format the feature touches, exactly where the kill switch is
                    # supposed to be total.
                    if PDF_OCR_ENABLED:
                        metadata.setdefault("text_source", "native")
                    yield document
                    continue
                if budget is None:
                    yield document
                    continue

                page_index = metadata.get("page")
                page = None
                try:
                    reader = self._ocr_reader()
                    if isinstance(page_index, int) and 0 <= page_index < len(reader.pages):
                        page = reader.pages[page_index]
                except Exception as error:
                    # Reopening for OCR failed. The page keeps its honest empty result.
                    logger.info("OCR could not open %s: %s", self.filepath, error)

                if page is None:
                    yield document
                    continue

                result = ocr_page(page, budget)
                if result.reason == "disabled":
                    # The kill switch is off: leave no trace at all, so the receipt is
                    # byte-identical to the build before OCR existed.
                    yield document
                    continue
                metadata["ocr_reason"] = result.reason
                metadata["ocr_attempted"] = result.attempted
                if result.images_seen:
                    metadata["ocr_images"] = result.images_seen
                if result.notes:
                    metadata["ocr_notes"] = ",".join(sorted(set(result.notes)))
                if result.text.strip():
                    # Extracted characters are NEVER discarded here, including weak
                    # ones. Whether coverage is good enough to call the document
                    # ingested is Core's judgement, not this service's -- so a weak
                    # page is REPORTED as weak (and escalated) rather than silently
                    # dropped, which would hide from Core the very thing it decides on.
                    document.page_content = result.text
                    metadata["text_source"] = "ocr"
                    metadata["ocr_confidence"] = round(result.confidence, 4)
                    metadata["ocr_chars"] = len(result.text.strip())
                else:
                    metadata["text_source"] = "none"
                yield document
        except OcrCancelled:
            # Stop producing pages rather than returning while a worker thread keeps
            # OCR-ing a document nobody is waiting for.
            logger.info("OCR cancelled for %s; stopping extraction", self.filepath)
            raise
        finally:
            self.close_ocr_reader()

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
    #: One definition, module level, shared with the Word branch.
    _OLE2_MAGIC = OLE2_MAGIC
    _ZIP_MAGIC = b"PK\x03\x04"

    #: Cap on the per-sheet sample of uncached-formula cell references carried in
    #: metadata, so a pathological workbook cannot inflate every chunk's metadata.
    _MAX_REPORTED_CELLS = 25

    #: Bounds on the formula scan. It is a DIAGNOSTIC pass over a file an untrusted
    #: uploader controls, and it re-reads the workbook twice on top of the parse the
    #: extractor already did — so it must not be the thing that makes a big upload
    #: expensive. Past either bound the scan stops and reports `unavailable`, which
    #: is the truth (we did not finish checking), never a clean zero.
    _MAX_SCAN_BYTES = 25 * 1024 * 1024
    _MAX_SCAN_CELLS = 2_000_000

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
            if os.path.getsize(self.filepath) > self._MAX_SCAN_BYTES:
                logger.info(
                    "Skipping the uncached-formula scan for %s: over the size bound",
                    self.filepath,
                )
                return {}, "unavailable"
        except OSError:
            return {}, "unavailable"
        try:
            from openpyxl import load_workbook

            per_sheet = {}
            cells_seen = 0
            formulas = load_workbook(self.filepath, data_only=False, read_only=True)
            try:
                values = load_workbook(self.filepath, data_only=True, read_only=True)
                try:
                    for name in formulas.sheetnames:
                        missing = []
                        f_rows = formulas[name].iter_rows()
                        v_rows = values[name].iter_rows()
                        for f_row, v_row in zip(f_rows, v_rows):
                            cells_seen += len(f_row)
                            if cells_seen > self._MAX_SCAN_CELLS:
                                logger.info(
                                    "Abandoning the uncached-formula scan for %s: "
                                    "over the cell bound",
                                    self.filepath,
                                )
                                return {}, "unavailable"
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


class RowCSVLoader(CSVLoader):
    """CSVLoader plus the one fact CSVLoader cannot express: a row with no values.

    CSVLoader already numbers every data row (`row`, 0-indexed) and rag_api already
    stored that number on every chunk -- so a CSV citation could always have named a
    row. What it could NOT do is say that a row is blank, because CSVLoader renders a
    value-less row as its COLUMN LABELS ALONE:

        region: \nrevenue:

    which is not empty text. Left alone, every row of every CSV counted as extracted,
    a gappy file and a complete file produced byte-identical receipts, and the label
    scaffolding was indexed as if it were content.

    The decision is made HERE, from the parsed field VALUES, and deliberately not by
    pattern-matching the rendered text: a cell whose content is "revenue: 4200000"
    renders indistinguishably from the scaffolding, so a text-shaped rule would call a
    real value empty. The values are only knowable at the parse, which is why this
    lives in the loader and not in the receipt.

    A blank row is yielded with EMPTY page_content and its `row` metadata intact --
    the same shape, for the same reason, as an image-only PPTX slide. Keeping the
    Document rather than dropping it is load-bearing: dropping it would shift every
    later row's citation by one, so a citation would point at the wrong record while
    looking correct.
    """

    def _blank_rows(self) -> set:
        """Indices of rows whose every field value is blank, read from the file.

        Uses the same `csv_args` and `encoding` the base loader parses with, so the
        indices refer to the same rows.

        A failure here is never fatal: it reports no blanks rather than inventing a
        coverage claim, which is the safe direction -- a real row is never wrongly
        blanked. It is NOT a general guarantee that blanks are always found, and the
        earlier wording here claimed that it was. Independent review found the case:
        this pass has no `autodetect_encoding` fallback, so a caller constructing this
        loader with `autodetect_encoding=True`, on a file the BASE loader recovers by
        re-detecting, would load successfully with its blank rows missed. `get_loader`
        never does that -- it converts non-UTF-8 to a UTF-8 temp file first and leaves
        autodetect off -- so through the route both passes read the same bytes or both
        fail together, which was measured. The claim was true as CONSTRUCTED and false
        as stated, which is the more dangerous of the two.
        """
        blank = set()
        try:
            with open(self.file_path, newline="", encoding=self.encoding) as fh:
                for i, row in enumerate(csv.DictReader(fh, **self.csv_args)):
                    values = []
                    for v in row.values():
                        # DictReader puts overflow fields in a LIST under the restkey.
                        if isinstance(v, list):
                            values.extend(x for x in v if x is not None)
                        elif v is not None:
                            values.append(v)
                    if not any(str(v).strip() for v in values):
                        blank.add(i)
        except Exception as e:  # noqa: BLE001 - diagnostic pass, never fatal
            logger.warning("CSV blank-row scan failed for %s: %s", self.file_path, e)
            return set()
        return blank

    def lazy_load(self) -> Iterator[Document]:
        blank = self._blank_rows()
        for doc in super().lazy_load():
            if doc.metadata.get("row") in blank:
                doc.page_content = ""
            yield doc
