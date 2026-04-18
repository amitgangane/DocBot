from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings


def get_llm() -> ChatOpenAI:
    """Get the LLM instance."""
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        max_retries=2,
    )


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers
     questions about research papers. Answer using only
     the context provided. If the answer is not in the
     context, say 'I don't know'."""),
    ("human", """
     Context:
     {context}

     Question: {question}
     """)
])


def generate_answer(context: str, question: str) -> str:
    """Generate an answer using the LLM."""
    llm = get_llm()
    chain = RAG_PROMPT | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content
