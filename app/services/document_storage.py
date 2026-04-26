"""Document storage helpers for uploaded PDFs."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

import httpx

from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger("document_storage")


def storage_enabled() -> bool:
    return bool(settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY and settings.SUPABASE_STORAGE_BUCKET)


def _build_storage_path(document_id: str, filename: str) -> str:
    safe_filename = Path(filename).name
    return f"{document_id}/{safe_filename}"


def _storage_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": settings.SUPABASE_SERVICE_ROLE_KEY,
    }


async def upload_document(*, document_id: str, filename: str, file_bytes: bytes) -> str:
    """Upload a PDF to cloud storage when configured, else local storage."""
    if storage_enabled():
        object_path = _build_storage_path(document_id, filename)
        encoded_path = quote(object_path, safe="/")
        url = f"{settings.SUPABASE_URL}/storage/v1/object/{settings.SUPABASE_STORAGE_BUCKET}/{encoded_path}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                headers={
                    **_storage_headers(),
                    "x-upsert": "true",
                    "Content-Type": "application/pdf",
                },
                content=file_bytes,
            )
            response.raise_for_status()

        logger.info(f"Uploaded PDF to Supabase Storage: {object_path}")
        return object_path

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(settings.UPLOAD_DIR, Path(filename).name)
    with open(file_path, "wb") as buffer:
        buffer.write(file_bytes)
    logger.info(f"Stored PDF locally: {file_path}")
    return file_path


async def delete_document_file(source_path: str) -> None:
    """Delete a stored PDF from cloud storage or local disk."""
    if not source_path:
        return

    if storage_enabled():
        url = f"{settings.SUPABASE_URL}/storage/v1/object/remove"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers={
                    **_storage_headers(),
                    "Content-Type": "application/json",
                },
                json={"prefixes": [source_path]},
            )
            response.raise_for_status()
        logger.info(f"Deleted PDF from Supabase Storage: {source_path}")
        return

    try:
        os.remove(source_path)
        logger.info(f"Deleted local PDF: {source_path}")
    except FileNotFoundError:
        logger.warning(f"Local PDF not found during delete: {source_path}")
