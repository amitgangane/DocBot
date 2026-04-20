import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger("generation")


def get_llm() -> ChatOpenAI:
    """Get the LLM instance."""
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        max_retries=2,
    )


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions about research papers.
Use the context provided to answer the question. Synthesize information from the context
to give a comprehensive answer. If the context contains partial information, use it to
provide the best possible answer. Only say 'I don't know' if the context is completely
unrelated to the question."""),
    ("human", """Context:
{context}

Question: {question}

Answer:""")
])


def generate_answer(context: str, question: str) -> str:
    """Generate an answer using the LLM."""
    start_time = time.time()
    logger.debug(f"Generating answer for: \"{question[:50]}...\"")

    llm = get_llm()
    chain = RAG_PROMPT | llm
    response = chain.invoke({"context": context, "question": question})

    elapsed = time.time() - start_time
    logger.info(f"LLM response: {len(response.content)} chars in {elapsed:.2f}s")
    return response.content
