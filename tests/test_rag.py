import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def test_llm_connection():
    """Test that LLM connection works."""
    llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=50)
    response = llm.invoke("Say hello")
    assert response.content is not None


def test_openai_key_exists():
    """Test that OpenAI API key is set."""
    assert os.getenv("OPENAI_API_KEY") is not None
