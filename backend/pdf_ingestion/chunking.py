from pdf_ingestion.docloader import load_pdf
from langchain_text_splitters.character import RecursiveCharacterTextSplitter


def chunk_documents() -> list:
    """Automatically loads PDF and returns chunks."""

    docs = load_pdf()

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size    = 500,
        chunk_overlap = 100,
    )

    final_chunks = char_splitter.split_documents(docs)
    print(f"Total chunks: {len(final_chunks)}")
    return final_chunks


#if __name__ == "__main__":
    # chunks = chunk_documents()
    # print("+" * 50)
    # print(chunks[20].page_content)
