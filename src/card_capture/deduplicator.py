import cv2
import numpy as np

class VisualDeduplicator:
    def __init__(self, threshold: int = 4):
        self.threshold = threshold

    def compute_phash(self, image: np.ndarray) -> str:
        """
        Computes a perceptual hash (pHash) for an image, focusing on the inner 60%.
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")

        # 1. Focus on inner 60% (artwork) to ignore border variations
        h, w = image.shape[:2]
        margin_h, margin_w = int(h * 0.2), int(w * 0.2)
        inner = image[margin_h:h-margin_h, margin_w:w-margin_w]
        
        # 2. pHash implementation:
        # - Grayscale
        if len(inner.shape) == 3:
            gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
        else:
            gray = inner
            
        # - Resize to 32x32
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        
        # - Discrete Cosine Transform (DCT)
        resized_float = np.float32(resized)
        dct = cv2.dct(resized_float)
        
        # - Extract top-left 8x8 coefficients
        dct_low = dct[:8, :8]
        
        # - Compute median of these 64 coefficients
        median = np.median(dct_low)
        
        # - Construct 64-bit integer based on whether coeff > median
        diff = dct_low > median
        
        # Return as hex string
        # Flatten and convert bits to integer
        flat = diff.flatten()
        hash_int = 0
        for bit in flat:
            hash_int = (hash_int << 1) | int(bit)
        
        return f"{hash_int:016x}"

    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        Calculates the Hamming distance between two hex-encoded hashes.
        """
        h1 = int(hash1, 16)
        h2 = int(hash2, 16)
        return bin(h1 ^ h2).count('1')

    def is_duplicate(self, hash1: str, hash2: str) -> bool:
        """
        Checks if two hashes represent duplicate images based on the threshold.
        """
        return self.hamming_distance(hash1, hash2) < self.threshold
