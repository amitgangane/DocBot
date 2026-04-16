
import os
from ragas.metrics import LLMContextPrecisionWithoutReference
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv()

def test_context_precision():
    # create object of class for tha specific metric
    load_dotenv()
    llm = ChatOpenAI(model = 'gpt-4', temperature = 0)
    langchain_llm = LangchainLLMWrapper(llm)
    context_precision = LLMContextPrecisionWithoutReference(llm = langchain_llm)


    # score