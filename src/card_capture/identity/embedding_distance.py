"""Embedding-based same-card criterion using OSNet cosine embeddings.

Replaces pHash Hamming distance with more robust OSNet embeddings.
Uses cosine distance metric for comparing 256-dim OSNet ReID embeddings.
"""

from typing import Tuple

import numpy as np
from scipy.spatial.distance import cosine


def compute_embedding_distance(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Compute cosine distance between two embeddings.

    Cosine distance = 1 - cosine_similarity.
    Range: [0, 2] where 0 = identical, 1 = orthogonal, 2 = opposite.

    Args:
        emb_a: First embedding, shape (D,) where D is embedding dimension.
        emb_b: Second embedding, shape (D,).

    Returns:
        Cosine distance (float). Returns NaN if either embedding is zero vector.
    """
    # scipy.spatial.distance.cosine handles normalization internally
    # It computes: 1 - (dot(u, v) / (norm(u) * norm(v)))
    return float(cosine(emb_a, emb_b))


def embedding_same_card_score(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    threshold: float = 0.5,
) -> bool:
    """Determine if two embeddings represent the same physical card.

    Uses cosine distance with a configurable threshold. Lower threshold
    means stricter matching (higher confidence same-card).

    Args:
        emb_a: First embedding (e.g., front side).
        emb_b: Second embedding (e.g., back side).
        threshold: Cosine distance threshold (default 0.5). Return True if
                  distance < threshold.

    Returns:
        True if distance < threshold (same card), False otherwise.
    """
    distance = compute_embedding_distance(emb_a, emb_b)

    # Handle NaN (zero vectors)
    if np.isnan(distance):
        return False

    return distance < threshold
