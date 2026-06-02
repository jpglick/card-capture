from typing import Literal, Optional
import cv2
import numpy as np

class VisualDeduplicator:
    def __init__(
        self, 
        threshold: int = 6, 
        backend: Literal["phash", "dino"] = "phash",
        dino_threshold: float = 0.92,
        dino_variant: str = "vits14"
    ):
        self.threshold = threshold
        self.backend = backend
        self.dino_threshold = dino_threshold
        self.dino_variant = dino_variant
        self._dino_deduper = None

    def _get_dino(self):
        if self._dino_deduper is None:
            from card_capture.ml.inference.dino_dedup import DinoDeduplicator
            self._dino_deduper = DinoDeduplicator(
                variant=self.dino_variant, 
                threshold=self.dino_threshold
            )
        return self._dino_deduper

    def compute_phash(self, image: np.ndarray) -> str:
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")

        h, w = image.shape[:2]
        margin_h, margin_w = int(h * 0.2), int(w * 0.2)
        inner = image[margin_h:h-margin_h, margin_w:w-margin_w]
        
        gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        dct_low = dct[0:8, 0:8]
        median = np.median(dct_low)
        
        bits = []
        for i in range(8):
            for j in range(8):
                bits.append("1" if dct_low[i, j] > median else "0")
        
        return f"{int(''.join(bits), 2):016x}"

    def hamming_distance(self, hash1: str, hash2: str) -> int:
        h1 = int(hash1, 16)
        h2 = int(hash2, 16)
        return bin(h1 ^ h2).count("1")

    def is_duplicate(self, hash1: str, hash2: str) -> bool:
        return self.hamming_distance(hash1, hash2) <= self.threshold

    def is_reid_duplicate(
        self,
        emb1: "np.ndarray",
        emb2: "np.ndarray",
        threshold: float = 0.15,
    ) -> bool:
        """Returns True if cosine distance between embeddings is <= threshold.

        Cosine distance = 1 - cosine_similarity.
        Threshold of 0.15 means >85% cosine similarity → same card.
        """
        n1 = np.linalg.norm(emb1)
        n2 = np.linalg.norm(emb2)
        if n1 == 0 or n2 == 0:
            return False
        cosine_sim = float(np.dot(emb1, emb2) / (n1 * n2))
        cosine_distance = 1.0 - cosine_sim
        return cosine_distance <= threshold
