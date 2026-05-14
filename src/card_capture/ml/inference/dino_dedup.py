"""DINOv2 + FAISS deduplication logic.

Provides clustering and similarity search for card instances across
sessions and videos.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from PIL import Image

from ..models.dino_embedder import DinoEmbedder, DINO_VARIANT


class DinoDeduplicator:
    """Handles embedding generation and indexing for deduplication."""

    def __init__(
        self,
        variant: DINO_VARIANT = "vits14",
        threshold: float = 0.92,
        index_path: Path | None = None,
    ):
        self.embedder = DinoEmbedder(variant=variant)
        self.threshold = threshold
        self.index_path = index_path
        
        # Use IndexFlatIP (Inner Product) because embeddings are L2-normalized,
        # making Inner Product equivalent to Cosine Similarity.
        self.index = faiss.IndexFlatIP(self.embedder.dim)
        
        # Mapping from index position -> card_instance_id
        self.id_map: list[str] = []

        if index_path and index_path.exists():
            self.load_index(index_path)

    def add_instances(self, instance_ids: list[str], image_paths: list[Path]):
        """Embed and index a list of card instances."""
        images = [Image.open(p) for p in image_paths]
        embeddings = self.embedder.embed_batch(images)
        
        # Convert to float32 numpy for FAISS
        embeddings_np = embeddings.cpu().numpy().astype("float32")
        
        self.index.add(embeddings_np)
        self.id_map.extend(instance_ids)
        
        if self.index_path:
            self.save_index(self.index_path)

    def find_duplicates(self, image_path: Path, top_k: int = 5) -> list[tuple[str, float]]:
        """Find existing instances that match the given image."""
        emb = self.embedder.embed_image(image_path)
        emb_np = emb.cpu().numpy().astype("float32")
        
        scores, indices = self.index.search(emb_np, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= self.threshold:
                results.append((self.id_map[idx], float(score)))
                
        return results

    def cluster_all(self) -> list[list[str]]:
        """Perform greedy clustering of all indexed instances based on threshold."""
        if self.index.ntotal == 0:
            return []

        # Get all embeddings
        embeddings = faiss.vector_to_array(self.index.get_xb()).reshape(self.index.ntotal, self.index.d)
        
        clusters = []
        visited = set()
        
        for i in range(self.index.ntotal):
            if i in visited:
                continue
                
            # Current instance is the head of a new cluster
            cluster = [self.id_map[i]]
            visited.add(i)
            
            # Find all neighbors within threshold
            # search(query, top_k)
            scores, indices = self.index.search(embeddings[i:i+1], self.index.ntotal)
            
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx not in visited and score >= self.threshold:
                    cluster.append(self.id_map[idx])
                    visited.add(idx)
            
            clusters.append(cluster)
            
        return clusters

    def save_index(self, path: Path):
        """Persist FAISS index and ID mapping."""
        faiss.write_index(self.index, str(path))
        # Save id_map as companion json
        import json
        path.with_suffix(".json").write_text(json.dumps(self.id_map))

    def load_index(self, path: Path):
        """Load FAISS index and ID mapping."""
        self.index = faiss.read_index(str(path))
        import json
        self.id_map = json.loads(path.with_suffix(".json").read_text())
