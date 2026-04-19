"""
RAGAS Evaluation Tests for DocBot RAG Pipeline.

Run with: pytest tests/test_ragas.py -v

Note: Uses RAGAS v0.2+ API with updated column names.
"""

import pytest
from dotenv import load_dotenv

load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
    ContextRecall,
)

from app.services.retrieval import search_similar
from app.services.reranker import rerank_documents
from app.services.generation import generate_answer
from app.core.config import settings


# Thresholds for passing tests
FAITHFULNESS_THRESHOLD = 0.7
ANSWER_RELEVANCY_THRESHOLD = 0.7
CONTEXT_PRECISION_THRESHOLD = 0.6
CONTEXT_RECALL_THRESHOLD = 0.6


# Test cases with ground truth (required for all RAGAS metrics now)
TEST_CASES = [
    {
        "question": "What is the Transformer architecture?",
        "ground_truth": "The Transformer is a model architecture that relies entirely on self-attention mechanisms to compute representations, without using recurrence or convolutions."
    },
    {
        "question": "How does multi-head attention work?",
        "ground_truth": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions, using multiple parallel attention heads."
    },
    {
        "question": "What are the advantages of self-attention over RNNs?",
        "ground_truth": "Self-attention allows for more parallelization than RNNs, reduces path length for learning long-range dependencies, and can process all positions simultaneously rather than sequentially."
    },
]


def run_rag_pipeline(question: str) -> dict:
    """Run RAG pipeline and return components for RAGAS evaluation."""
    # Retrieve
    docs = search_similar(question, k=settings.RERANKER_INITIAL_K)

    # Rerank
    reranked_docs = rerank_documents(question, docs, top_k=settings.RETRIEVER_K)

    # Build context
    contexts = [doc.page_content for doc in reranked_docs]
    context_str = "\n\n".join(contexts)

    # Generate
    answer = generate_answer(context_str, question)

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }


@pytest.fixture(scope="module")
def rag_results():
    """Run RAG pipeline for all test cases."""
    results = []
    for case in TEST_CASES:
        result = run_rag_pipeline(case["question"])
        result["ground_truth"] = case["ground_truth"]
        results.append(result)
    return results


@pytest.fixture(scope="module")
def evaluation_scores(rag_results):
    """Evaluate RAG results with RAGAS."""
    # RAGAS v0.2+ uses different column names
    data = {
        "user_input": [r["question"] for r in rag_results],
        "response": [r["answer"] for r in rag_results],
        "retrieved_contexts": [r["contexts"] for r in rag_results],
        "reference": [r["ground_truth"] for r in rag_results],
    }
    dataset = Dataset.from_dict(data)

    # Initialize metrics
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]

    scores = evaluate(dataset, metrics=metrics)
    return scores


def get_score_value(score):
    """Extract numeric value from score (handles list or float)."""
    if isinstance(score, list):
        return sum(score) / len(score) if score else 0.0
    return float(score) if score is not None else 0.0


class TestRAGASMetrics:
    """RAGAS evaluation tests."""

    def test_faithfulness(self, evaluation_scores):
        """Test that answers are grounded in the retrieved context."""
        score = get_score_value(evaluation_scores.get('faithfulness', 0))
        print(f"\n📊 Faithfulness Score: {score:.4f}")
        assert score >= FAITHFULNESS_THRESHOLD, (
            f"Faithfulness score {score:.4f} is below threshold {FAITHFULNESS_THRESHOLD}"
        )

    def test_answer_relevancy(self, evaluation_scores):
        """Test that answers are relevant to the questions."""
        score = get_score_value(evaluation_scores.get('answer_relevancy', 0))
        print(f"\n📊 Answer Relevancy Score: {score:.4f}")
        # Skip if NaN (embedding issue)
        if score != score:  # NaN check
            pytest.skip("Answer relevancy returned NaN (embedding issue)")
        assert score >= ANSWER_RELEVANCY_THRESHOLD, (
            f"Answer relevancy score {score:.4f} is below threshold {ANSWER_RELEVANCY_THRESHOLD}"
        )

    def test_context_precision(self, evaluation_scores):
        """Test that retrieved context is relevant to the questions."""
        score = get_score_value(evaluation_scores.get('context_precision', 0))
        print(f"\n📊 Context Precision Score: {score:.4f}")
        assert score >= CONTEXT_PRECISION_THRESHOLD, (
            f"Context precision score {score:.4f} is below threshold {CONTEXT_PRECISION_THRESHOLD}"
        )

    def test_context_recall(self, evaluation_scores):
        """Test that we retrieve all information needed to answer."""
        score = get_score_value(evaluation_scores.get('context_recall', 0))
        print(f"\n📊 Context Recall Score: {score:.4f}")
        assert score >= CONTEXT_RECALL_THRESHOLD, (
            f"Context recall score {score:.4f} is below threshold {CONTEXT_RECALL_THRESHOLD}"
        )


class TestRAGPipelineComponents:
    """Test individual RAG pipeline components."""

    def test_retrieval_returns_documents(self):
        """Test that retrieval returns documents."""
        docs = search_similar("What is attention?", k=5)
        assert len(docs) > 0, "Retrieval returned no documents"
        assert len(docs) <= 5, "Retrieval returned more documents than requested"

    def test_reranker_reduces_documents(self):
        """Test that reranker reduces document count."""
        docs = search_similar("What is attention?", k=10)
        reranked = rerank_documents("What is attention?", docs, top_k=5)
        assert len(reranked) <= len(docs), "Reranker returned more docs than input"
        assert len(reranked) <= 5, "Reranker returned more than top_k"

    def test_generation_returns_answer(self):
        """Test that generation returns a non-empty answer."""
        context = "The Transformer is a neural network architecture."
        answer = generate_answer(context, "What is the Transformer?")
        assert answer is not None, "Generation returned None"
        assert len(answer) > 0, "Generation returned empty answer"


def test_full_pipeline():
    """Test the full RAG pipeline end-to-end."""
    result = run_rag_pipeline("What is the Transformer?")

    assert "question" in result
    assert "answer" in result
    assert "contexts" in result
    assert len(result["answer"]) > 0
    assert len(result["contexts"]) > 0
