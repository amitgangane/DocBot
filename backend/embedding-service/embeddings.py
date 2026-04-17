import os
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()



embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024
)

