import asyncio
from uuid import uuid4

from dotenv import load_dotenv
from openai import AsyncOpenAI
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException

from app.core.config import settings
from app.services.graph import close_checkpointer
from app.services.reranker import rerank_documents
from app.services.retrieval import search_similar
from app.services.rag_service import query as rag_query
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

load_dotenv()


PRECISION_THRESHOLD = 0.5
RECALL_THRESHOLD = 0.5
FAITHFULNESS_THRESHOLD = 0.5
ANSWER_RELEVANCY_THRESHOLD = 0.5
TESTSET = [
    {
        "q": "What is BERT?",
        "ref": (
            "BERT is a language representation model that pre-trains deep "
            "bidirectional representations by conditioning on both left and "
            "right context."
        ),
    },
    {
        "q": "What makes BERT different from traditional language models?",
        "ref": (
            "BERT differs from traditional language models by using "
            "bidirectional context instead of unidirectional left-to-right or "
            "right-to-left modeling."
        ),
    },
    {
        "q": "What is masked language modeling in BERT?",
        "ref": (
            "Masked language modeling randomly masks tokens in the input and "
            "trains BERT to predict the original tokens from surrounding "
            "context."
        ),
    },
    {
        "q": "What is RAG?",
        "ref": (
            "Retrieval-Augmented Generation improves language model answers by "
            "retrieving relevant information from an external knowledge source "
            "before generation."
        ),
    },
    {
        "q": "What is Modular RAG?",
        "ref": (
            "Modular RAG framework is designed to be flexible and adaptable, "
            "allowing users to customize the RAG pipeline to their specific "
            "needs and requirements."
        ),
    },
]

def _run_retrieval_stack(question: str):
    """Return the final reranked docs for a question."""
    initial_docs = search_similar(question, k=settings.RERANKER_INITIAL_K)
    reranked_docs = rerank_documents(question, initial_docs, top_k=settings.RETRIEVER_K)
    return reranked_docs


def _get_retrieved_contexts(question: str) -> list[str]:
    reranked_docs = _run_retrieval_stack(question)
    return [doc.page_content for doc in reranked_docs if doc.page_content]


def _get_eval_llm():
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return llm_factory("gpt-4o-mini", provider="openai", client=client)


def _get_eval_embeddings():
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return RagasOpenAIEmbeddings(client=client, model=settings.EMBEDDING_MODEL)


def _get_generated_answer(question: str) -> str:
    async def _run_query() -> str:
        try:
            result = await rag_query(
                question,
                thread_id=f"faithfulness-{uuid4()}",
                use_cache=False,
            )
            return result["answer"]
        finally:
            await close_checkpointer()

    return asyncio.run(_run_query())


@pytest.mark.parametrize(
    ("question", "reference"),
    [(item["q"], item["ref"]) for item in TESTSET],
    ids=[item["q"] for item in TESTSET],
)
def test_context_precision(question: str, reference: str):
    """
    Evaluate retrieval quality for a known question.

    Context precision measures whether the retrieved chunks are relevant to the
    supplied reference answer. We score the app's actual retrieved contexts,
    rather than calling the HTTP API, because `/query` returns source metadata
    instead of the full retrieved document texts needed by the metric.
    """
    if not settings.OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY is not configured")

    try:
        retrieved_contexts = _get_retrieved_contexts(question)
    except ResponseHandlingException as exc:
        pytest.skip(f"Qdrant is not reachable in this environment: {exc}")
    except Exception as exc:
        if "nodename nor servname provided" in str(exc).lower():
            pytest.skip(f"External retrieval services are not reachable: {exc}")
        raise

    assert retrieved_contexts, "Retrieval returned no contexts"

    llm = _get_eval_llm()
    scorer = ContextPrecision(llm=llm)

    result = asyncio.run(
        scorer.ascore(
            user_input=question,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        )
    )

    assert result.value >= PRECISION_THRESHOLD, (
        f"Context precision {result.value:.4f} is below threshold "
        f"{PRECISION_THRESHOLD:.2f}"
    )


@pytest.mark.parametrize(
    ("question", "reference"),
    [(item["q"], item["ref"]) for item in TESTSET],
    ids=[item["q"] for item in TESTSET],
)
def test_context_recall(question: str, reference: str):
    """
    Evaluate whether the retrieved chunks collectively cover the information
    needed by the reference answer.
    """
    if not settings.OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY is not configured")

    try:
        retrieved_contexts = _get_retrieved_contexts(question)
    except ResponseHandlingException as exc:
        pytest.skip(f"Qdrant is not reachable in this environment: {exc}")
    except Exception as exc:
        if "nodename nor servname provided" in str(exc).lower():
            pytest.skip(f"External retrieval services are not reachable: {exc}")
        raise

    assert retrieved_contexts, "Retrieval returned no contexts"

    llm = _get_eval_llm()
    scorer = ContextRecall(llm=llm)

    result = asyncio.run(
        scorer.ascore(
            user_input=question,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        )
    )

    assert result.value >= RECALL_THRESHOLD, (
        f"Context recall {result.value:.4f} is below threshold "
        f"{RECALL_THRESHOLD:.2f}"
    )


@pytest.mark.parametrize(
    ("question", "reference"),
    [(item["q"], item["ref"]) for item in TESTSET],
    ids=[item["q"] for item in TESTSET],
)
def test_faithfulness(question: str, reference: str):
    """
    Evaluate whether the app's generated answer stays grounded in the retrieved
    contexts for a given question.
    """
    if not settings.OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY is not configured")

    try:
        retrieved_contexts = _get_retrieved_contexts(question)
        answer = _get_generated_answer(question)
    except ResponseHandlingException as exc:
        pytest.skip(f"Qdrant is not reachable in this environment: {exc}")
    except Exception as exc:
        if "nodename nor servname provided" in str(exc).lower():
            pytest.skip(f"External retrieval services are not reachable: {exc}")
        raise

    assert retrieved_contexts, "Retrieval returned no contexts"
    assert answer, "Generation returned no answer"

    llm = _get_eval_llm()
    scorer = Faithfulness(llm=llm)

    result = asyncio.run(
        scorer.ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=retrieved_contexts,
        )
    )

    assert result.value >= FAITHFULNESS_THRESHOLD, (
        f"Faithfulness {result.value:.4f} is below threshold "
        f"{FAITHFULNESS_THRESHOLD:.2f}"
    )


@pytest.mark.parametrize(
    ("question", "reference"),
    [(item["q"], item["ref"]) for item in TESTSET],
    ids=[item["q"] for item in TESTSET],
)
def test_answer_relevancy(question: str, reference: str):
    """
    Evaluate whether the generated answer directly addresses the user question.
    """
    if not settings.OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY is not configured")

    try:
        answer = _get_generated_answer(question)
    except ResponseHandlingException as exc:
        pytest.skip(f"Qdrant is not reachable in this environment: {exc}")
    except Exception as exc:
        if "nodename nor servname provided" in str(exc).lower():
            pytest.skip(f"External retrieval services are not reachable: {exc}")
        raise

    assert answer, "Generation returned no answer"

    llm = _get_eval_llm()
    embeddings = _get_eval_embeddings()
    scorer = AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1)

    result = asyncio.run(
        scorer.ascore(
            user_input=question,
            response=answer,
        )
    )

    assert result.value >= ANSWER_RELEVANCY_THRESHOLD, (
        f"Answer relevancy {result.value:.4f} is below threshold "
        f"{ANSWER_RELEVANCY_THRESHOLD:.2f}"
    )
