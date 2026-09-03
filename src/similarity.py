"""
similarity.py
--------------
Document-level semantic similarity, used to catch near-duplicate documents
within a batch (e.g. the same invoice resubmitted with one field changed).

We represent each document as a TF-IDF vector ("embedding") over its text
and compare documents with cosine similarity. TF-IDF is used here because it
has zero external dependencies and runs anywhere; the interface
(`embed(texts) -> matrix`, `pairwise_similarity(matrix) -> matrix`) is kept
separate from the rest of the pipeline so it can be swapped for dense
transformer embeddings (e.g. sentence-transformers / an embeddings API) with
no changes to calling code — see README "Future Improvements".
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NEAR_DUPLICATE_THRESHOLD = 0.92


def embed(texts: list) -> np.ndarray:
    """Turn a list of raw document texts into TF-IDF embedding vectors."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    return matrix


def pairwise_similarity(matrix) -> np.ndarray:
    return cosine_similarity(matrix)


def find_near_duplicates(doc_ids: list, texts: list, threshold: float = NEAR_DUPLICATE_THRESHOLD) -> list:
    """Returns a list of (doc_id_a, doc_id_b, similarity) tuples for any pair
    of documents in the batch whose content is nearly identical.
    """
    if len(texts) < 2:
        return []
    matrix = embed(texts)
    sims = pairwise_similarity(matrix)
    results = []
    n = len(doc_ids)
    for i in range(n):
        for j in range(i + 1, n):
            score = sims[i, j]
            if score >= threshold:
                results.append((doc_ids[i], doc_ids[j], float(score)))
    return results
