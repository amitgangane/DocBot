import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from embeddings.vector_st import get_retriever


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.6,
    max_tokens=500,
    max_retries=2,
)

prompt = ChatPromptTemplate.from_messages([
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


def query(question: str) -> str:
    """Query the research paper with a question."""
    retriever = get_retriever()

    # Get relevant context
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Create chain and invoke
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    return response.content


if __name__ == "__main__":
    answer = query("can you explain the componants of transformer archeticture?")
    print(answer)
