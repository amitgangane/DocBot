import re

from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger("chunking")


LOW_SIGNAL_PATTERNS = [
    re.compile(r"!\["),
    re.compile(r"\bsummary:\b", re.IGNORECASE),
    re.compile(r"\bfigure\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bfig\.\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bdiagram\b", re.IGNORECASE),
    re.compile(r"\breferences?\b", re.IGNORECASE),
    re.compile(r"\bbibliography\b", re.IGNORECASE),
    re.compile(r"\bworks cited\b", re.IGNORECASE),
]

EXPLANATORY_PATTERNS = [
    re.compile(r"\bis\b", re.IGNORECASE),
    re.compile(r"\bare\b", re.IGNORECASE),
    re.compile(r"\buses?\b", re.IGNORECASE),
    re.compile(r"\busing\b", re.IGNORECASE),
    re.compile(r"\bimproves?\b", re.IGNORECASE),
    re.compile(r"\bretriev(?:e|es|ed|ing)\b", re.IGNORECASE),
    re.compile(r"\bgenerat(?:e|es|ed|ing)\b", re.IGNORECASE),
    re.compile(r"\bconsists?\s+of\b", re.IGNORECASE),
    re.compile(r"\brefers?\s+to\b", re.IGNORECASE),
    re.compile(r"\benables?\b", re.IGNORECASE),
]


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def _looks_like_table(text: str) -> bool:
    normalized = _normalize_text(text)
    if "|" in text:
        return True
    if re.search(r"\btable\s+[ivx0-9]+\b", normalized, re.IGNORECASE):
        return True
    return False


def _looks_like_reference_chunk(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return True

    if re.search(r"^(references|bibliography|works cited)\b", normalized, re.IGNORECASE):
        return True

    citation_like_lines = 0
    total_lines = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        total_lines += 1
        if re.match(r"^\[\d+\]", line):
            citation_like_lines += 1
        elif re.match(r"^[A-Z][A-Za-z' -]+,\s*[A-Z]\.", line):
            citation_like_lines += 1
        elif re.search(r"\(\d{4}\)", line) and len(line) < 220:
            citation_like_lines += 1

    return total_lines > 0 and citation_like_lines / total_lines >= 0.6


def _is_reference_heavy(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False

    bracket_citations = len(re.findall(r"\[\d+\]", text))
    years = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
    arxiv_terms = len(re.findall(r"\b(?:arxiv|preprint|doi|proceedings)\b", text, re.IGNORECASE))
    author_patterns = len(re.findall(r"\b[A-Z]\.\s*[A-Z][a-z]+", text))

    citation_score = (
        bracket_citations * 2
        + years
        + arxiv_terms * 2
        + author_patterns
    )
    return citation_score >= 6


def _citation_density(text: str) -> float:
    normalized = _normalize_text(text)
    if not normalized:
        return 0.0

    word_count = max(1, len(normalized.split()))
    bracket_citations = len(re.findall(r"\[\d+\]", text))
    years = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
    arxiv_terms = len(re.findall(r"\b(?:arxiv|preprint|doi|proceedings)\b", text, re.IGNORECASE))
    author_patterns = len(re.findall(r"\b[A-Z]\.\s*[A-Z][a-z]+", text))

    weighted_hits = (bracket_citations * 2) + years + (arxiv_terms * 2) + author_patterns
    return weighted_hits / word_count


def _starts_like_citation_blob(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False

    head = normalized[:240]
    if re.match(r"^\[\d+\]", head):
        return True

    head_bracket_citations = len(re.findall(r"\[\d+\]", head))
    head_author_patterns = len(re.findall(r"\b[A-Z]\.\s*[A-Z][a-z]+", head))
    head_years = len(re.findall(r"\b(?:19|20)\d{2}\b", head))

    return (
        head_bracket_citations >= 2
        or (head_author_patterns >= 2 and head_years >= 1)
    )


def _has_explanatory_prose(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False

    sentence_count = len(re.findall(r"[.!?]", text))
    explanatory_terms = sum(1 for pattern in EXPLANATORY_PATTERNS if pattern.search(text))

    prose_score = sentence_count + (explanatory_terms * 2)
    return prose_score >= 4


def _low_signal_hits(text: str) -> int:
    return sum(1 for pattern in LOW_SIGNAL_PATTERNS if pattern.search(text))


def _infer_content_type(text: str) -> str:
    normalized = _normalize_text(text)
    lower = normalized.lower()

    if _looks_like_table(text):
        return "table"
    if _looks_like_reference_chunk(text) or _is_reference_heavy(text):
        return "references"
    if lower.startswith(("fig.", "figure ")):
        return "figure_caption"
    if lower.startswith(("summary:", "### summary", "![", "image summary")):
        return "image_summary"
    return "body"


def _annotate_chunk_metadata(chunks: list) -> list:
    for chunk in chunks:
        text = chunk.page_content or ""
        metadata = dict(getattr(chunk, "metadata", {}) or {})
        metadata["content_type"] = _infer_content_type(text)
        metadata["is_reference_heavy"] = _is_reference_heavy(text)
        chunk.metadata = metadata
    return chunks


def _should_keep_chunk(text: str) -> bool:
    normalized = _normalize_text(text)
    if len(normalized) < 80:
        return False

    if _looks_like_table(text):
        return True

    if _looks_like_reference_chunk(text):
        return False

    if _is_reference_heavy(text) and not _has_explanatory_prose(text):
        return False

    # Drop citation-dense blobs even when they appear in the middle of a PDF.
    if _starts_like_citation_blob(text):
        return False

    if _citation_density(text) >= 0.08 and not _looks_like_table(text):
        return False

    hits = _low_signal_hits(text)
    word_count = len(normalized.split())

    # Drop image/figure summary chunks that are mostly caption-style metadata.
    if hits >= 2 and word_count < 140:
        return False

    if normalized.lower().startswith(("fig.", "figure ", "summary:", "![", "### summary")) and word_count < 180:
        return False

    return True


def _filter_chunks(chunks: list) -> list:
    kept = []
    dropped = 0
    for chunk in chunks:
        if _should_keep_chunk(chunk.page_content):
            kept.append(chunk)
        else:
            dropped += 1

    if dropped:
        logger.info(f"Filtered out {dropped} low-signal chunks before embedding")
    return kept


def chunk_documents(docs: list) -> list:
    """
    Split documents into chunks for embedding.
    """
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    chunks = _annotate_chunk_metadata(char_splitter.split_documents(docs))
    filtered_chunks = _filter_chunks(chunks)
    print(f"Total chunks: {len(chunks)} -> {len(filtered_chunks)} after filtering")
    return filtered_chunks
