import cv2
import numpy as np

class VisualDeduplicator:
    def __init__(self, threshold: int = 4):
        self.threshold = threshold

    def compute_phash(self, image: np.ndarray) -> str:
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
        
        return hex(int("".join(bits), 2))

    def is_duplicate(self, hash1: str, hash2: str) -> bool:
        h1 = int(hash1, 16)
        h2 = int(hash2, 16)
        return bin(h1 ^ h2).count('1') < self.threshold
