import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


def get_image_parser():
    """Get LLM-based image parser for PDFs."""
    from langchain_community.document_loaders.parsers import LLMImageBlobParser
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)
    return LLMImageBlobParser(model=llm)


def load_pdf(file_path: str) -> list:
    """
    Load a PDF and return a list of LangChain Document objects.
    One Document per page, images described by GPT, tables as markdown.
    """
    from langchain_pymupdf4llm import PyMuPDF4LLMLoader

    image_parser = get_image_parser()

    loader = PyMuPDF4LLMLoader(
        file_path=file_path,
        mode="page",
        extract_images=True,
        table_strategy="lines",
        images_parser=image_parser,
    )

    docs = loader.load()
    print(f"Loaded {len(docs)} pages from {os.path.basename(file_path)}")
    return docs
