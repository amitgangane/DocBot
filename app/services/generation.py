import time
from typing import AsyncGenerator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger("generation")


def get_llm(streaming: bool = False) -> ChatOpenAI:
    """Get the LLM instance."""
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        max_retries=2,
        streaming=streaming,
    )


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions about research papers.
Use the context provided to answer the question. Synthesize information from the context
to give a comprehensive answer. If the context contains partial information, use it to
provide the best possible answer. Only say 'I don't know' if the context is completely
unrelated to the question.

IMPORTANT FORMATTING RULES:
- Do NOT repeat or restate the question in your answer. Start directly with the answer.
- When writing mathematical equations, formulas, or expressions:
  - Use LaTeX format for all math
  - Wrap inline math with single dollar signs: $E = mc^2$
  - Wrap block/display math with double dollar signs
  - Use proper LaTeX commands for symbols (\\alpha, \\beta, \\sum, \\int, \\frac, etc.)"""),
    ("human", """Context:
{context}

Question: {question}

Answer (do not repeat the question):""")
])




