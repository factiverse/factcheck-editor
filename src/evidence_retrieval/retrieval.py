import json
from typing import List, Union
from unittest.mock import MagicMock, mock_open
import time
import numba
import numpy as np
from numpy import ndarray

import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sklearn.metrics import f1_score, precision_score, recall_score

import numpy as np
import urllib


@numba.njit
def pairwise_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes cosine similarity between aligned pairs of vectors in a and b.

    Args:
        a: Array, shape (N, D)
        b: Array, shape (N, D)

    Returns:
        np.ndarray: Cosine similarity for each pair, shape (N,)
    """
    N, D = a.shape
    result = np.empty(N, dtype=np.float64)

    for i in range(N):
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for j in range(D):
            dot += a[i, j] * b[i, j]
            norm_a += a[i, j] * a[i, j]
            norm_b += b[i, j] * b[i, j]
        if norm_a > 0 and norm_b > 0:
            result[i] = dot / (np.sqrt(norm_a) * np.sqrt(norm_b))
        else:
            result[i] = 0.0
    return result


def load_source_relevance_feedback():
    column_mapping = {
        "relevanceLabel": "relevance_label",
        "evidence.publishDate": "evidence.publish_date",
    }
    return pd.read_json(  # type: ignore
        "data/relevance/sourceRelevance.jsonl_aggregated.jsonl",
        lines=True,
        convert_dates=["evidence.publishDate"],
    ).rename(columns=column_mapping)


def compute_batch_pairwise_similarity(
    query_embeddings: Union[ndarray, List[float]],
    doc_sentences_embeddings: Union[ndarray, List[float]],
) -> List[float]:
    """Compute pairwise similarity between query and sentence embeddings.

    Args:
        query_embeddings: List/array of query embeddings
        doc_sentences_embeddings: List/array of sentence embeddings

    Returns:
        List of similarity scores for each query-sentence pair
    """
    # Convert to array if needed
    if isinstance(query_embeddings, list):
        query_embeddings = np.stack(query_embeddings)
    if isinstance(doc_sentences_embeddings, list):
        doc_sentences_embeddings = np.stack(doc_sentences_embeddings)

    # Ensure embeddings have same first dimension
    if query_embeddings.shape[0] != doc_sentences_embeddings.shape[0]:
        raise ValueError("Number of query and sentence embeddings must match")

    # Compute cosine similarity between corresponding pairs
    similarities = pairwise_cosine_similarity(
        query_embeddings, doc_sentences_embeddings
    )

    return similarities.tolist()

def compute_cross_encoder_similarity_local(queries, docs, cross_encoder, batch_size=32):
    """Computes similarity scores using a cross-encoder model.

    Args:
        cross_encoder: The CrossEncoder model instance.
        queries: List of query strings.
        docs: List of document strings.
        batch_size: Number of pairs to process in each batch.

    Returns:
        List of similarity scores for each query-document pair.
    """
    scores = []
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i : i + batch_size]  # noqa: E203
        batch_docs = docs[i : i + batch_size]  # noqa: E203
        pairs = list(zip(batch_queries, batch_docs))
        batch_scores = cross_encoder.predict(pairs)
        scores.extend(batch_scores)
    return scores

def relevance_accuracy():
    """Tests relevance computation using bi-encoder and cross-encoder.

    Args:
        source_relevance_feedback: Dataframe containing relevance feedback.
    """
    label = []
    source_relevance_feedback_df = load_source_relevance_feedback()

    # Bi-encoder predictions
    pred_biencoder = []
    threshold_biencoder = 0.6

    # Cross-encoder predictions
    pred_crossencoder = []
    threshold_crossencoder = 0.6

    source_relevance_feedback = source_relevance_feedback_df.loc[
        (source_relevance_feedback_df["num_annotations"] >= 3)
        & (source_relevance_feedback_df["agree_ratio"] == 1)
    ]
    # Print data statistics for relevance
    print("\nData statistics for relevance:")
    print("  Total samples:", len(source_relevance_feedback))
    print("  Relevance label value counts:")
    print(source_relevance_feedback["relevance_label"].value_counts())
    print("  Relevance label distribution (%):")
    print((source_relevance_feedback["relevance_label"].value_counts(normalize=True) * 100).round(2))
    batch_size = 64
    queries = []
    docs = []

    for _, row in source_relevance_feedback.iterrows():
        queries.append(row["claim"])
        docs.append(row["evidence.snippet"])
        label.append(row["relevance_label"])
    # connector = LangChainLLMClient()
    # Initialize sentence encoder (bi-encoder) model
    # Using a multilingual model that handles query-document similarity
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # model_name = "perplexity-ai/pplx-embed-v1-0.6b"
    sentence_encoder = SentenceTransformer(
        model_name,
        trust_remote_code=True
    )

    # Initialize cross-encoder model
    # Using a multilingual model suitable for the task
    cross_encoder = CrossEncoder("cross_encoder_fine_tuned")

    print("\n" + "=" * 80)
    print("COMPARING BI-ENCODER (SentenceTransformer) VS CROSS-ENCODER")
    print("=" * 80)

    # Timing variables
    time_biencoder = 0
    time_crossencoder = 0
    start_cross = time.time()
    # scores_crossencoder = comput_cross_encoder_similarity(queries, docs, batch_size=batch_size)
    scores_crossencoder = compute_cross_encoder_similarity_local(queries, docs, batch_size=batch_size, cross_encoder=cross_encoder)
    time_crossencoder += time.time() - start_cross

    # Process in batches
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i : i + batch_size]  # noqa: E203
        batch_docs = docs[i : i + batch_size]  # noqa: E203

        # Bi-encoder approach using SentenceTransformer directly
        start_bi = time.time()
        query_emb = sentence_encoder.encode(
            batch_queries, convert_to_tensor=False, show_progress_bar=False
        )
        doc_emb = sentence_encoder.encode(
            batch_docs, convert_to_tensor=False, show_progress_bar=False
        )
        # query_emb = connector.get_query_embeddings(batch_queries)
        # doc_emb = connector.get_embeddings(batch_docs) 
        sims_biencoder = compute_batch_pairwise_similarity(query_emb, doc_emb)
        time_biencoder += time.time() - start_bi
        pred_biencoder.extend(
            [1 if sim >= threshold_biencoder else 0 for sim in sims_biencoder]
        )

        # Cross-encoder approach (direct pair scoring)

        
    pred_crossencoder.extend(
        [
            1 if score >= threshold_crossencoder else 0
            for score in scores_crossencoder
        ]
    )

    # Calculate metrics for bi-encoder
    f1_bi = f1_score(label, pred_biencoder, average="binary")
    recall_bi = recall_score(label, pred_biencoder, average="binary")
    precision_bi = precision_score(label, pred_biencoder, average="binary")

    # Calculate metrics for cross-encoder
    f1_cross = f1_score(label, pred_crossencoder, average="binary")
    recall_cross = recall_score(label, pred_crossencoder, average="binary")
    precision_cross = precision_score(label, pred_crossencoder, average="binary")

    # Calculate throughput metrics
    num_pairs = len(queries)
    throughput_bi = num_pairs / time_biencoder if time_biencoder > 0 else 0
    throughput_cross = (
        num_pairs / time_crossencoder if time_crossencoder > 0 else 0
    )

    print("\nBI-ENCODER (Embedding + Cosine Similarity) Results:")
    print(f"  F1 Score:  {f1_bi:.4f}")
    print(f"  Recall:    {recall_bi:.4f}")
    print(f"  Precision: {precision_bi:.4f}")
    print(f"  Threshold: {threshold_biencoder}")
    print(f"  Time:      {time_biencoder:.2f}s ({throughput_bi:.1f} pairs/sec)")

    print("\nCROSS-ENCODER (Direct Pair Scoring) Results:")
    print(f"  F1 Score:  {f1_cross:.4f}")
    print(f"  Recall:    {recall_cross:.4f}")
    print(f"  Precision: {precision_cross:.4f}")
    print(f"  Threshold: {threshold_crossencoder}")
    print(
        f"  Time:      {time_crossencoder:.2f}s ({throughput_cross:.1f} pairs/sec)"
    )

    print("\nAccuracy Comparison:")
    print(
        f"  F1 Improvement:        {(f1_cross - f1_bi):+.4f} ({((f1_cross/f1_bi - 1) * 100):+.2f}%)"
    )
    print(
        f"  Recall Improvement:    {(recall_cross - recall_bi):+.4f} ({((recall_cross/recall_bi - 1) * 100):+.2f}%)"
    )
    print(
        f"  Precision Improvement: {(precision_cross - precision_bi):+.4f} ({((precision_cross/precision_bi - 1) * 100):+.2f}%)"
    )

    print("\nSpeed Comparison:")
    speedup = time_crossencoder / time_biencoder if time_biencoder > 0 else 0
    if speedup > 1:
        print(f"  Bi-encoder is {speedup:.2f}x FASTER than cross-encoder")
    else:
        print(f"  Cross-encoder is {1/speedup:.2f}x FASTER than bi-encoder")
    print(f"  Time difference: {abs(time_crossencoder - time_biencoder):.2f}s")
    print("=" * 80 + "\n")


    # Cross-encoder should generally perform better (informational, not enforced)
    if f1_cross < f1_bi:
        print(
            f"⚠️  Warning: Cross-encoder F1 ({f1_cross:.4f}) is lower than bi-encoder ({f1_bi:.4f})"
        )
    else:
        print(
            f"✓ Cross-encoder F1 ({f1_cross:.4f}) >= bi-encoder ({f1_bi:.4f})"
        )

if __name__ == "__main__":
    relevance_accuracy()