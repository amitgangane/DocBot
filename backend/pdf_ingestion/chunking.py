from langchain_text_splitters import RecursiveCharacterTextSplitter
from docloader import load_pdf


text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_text(load_pdf)

print(texts)