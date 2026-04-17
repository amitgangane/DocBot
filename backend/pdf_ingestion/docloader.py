from dotenv import load_dotenv
from getpass import getpass
import os
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_openai import ChatOpenAI



load_dotenv()


if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass("OpenAI API key =")



# CONFIG
FILE_PATH = "/Users/amit/Desktop/DocBot/backend/pdf_ingestion/documents/attention.pdf"


# INIT LLM (once, reused)
def get_image_parser() -> LLMImageBlobParser:
    llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)
    return LLMImageBlobParser(model=llm)


def load_pdf(file_path: str = FILE_PATH) -> list:
    """
    Load a PDF and return a list of LangChain Document objects.
    One Document per page, images described by GPT, tables as markdown.
    """
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

# docs = (load_pdf(file_path=FILE_PATH))
# print(docs[2].page_content)