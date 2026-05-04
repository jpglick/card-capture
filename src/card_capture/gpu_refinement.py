import numpy as np
import cv2
from typing import List, Tuple, Union
from .models import Point
from .cropper import _orient_for_target_canvas, order_points_clockwise
from .detectors import probe_torch_device_status

try:
    import torch
    import kornia
except Exception:  # pragma: no cover - optional acceleration path
    torch = None
    kornia = None

class KorniaNormalizer:
    def __init__(self, width: int = 750, height: int = 1050, device: str = "auto"):
        if torch is None or kornia is None:
            raise RuntimeError("kornia/torch not available")
        self.width = width
        self.height = height
        if device == "auto":
            device = probe_torch_device_status("auto").resolved
        self.device = torch.device(device)

    def warp_canonical_batch(
        self, batch_data: List[Tuple[Union[str, np.ndarray], List[Point]]]
    ) -> List[np.ndarray]:
        """
        batch_data: List of (image_or_path, corners)
        """
        tensors = []
        matrices = []

        for image_or_path, corners in batch_data:
            if isinstance(image_or_path, np.ndarray):
                img = image_or_path
            else:
                img = cv2.imread(image_or_path)
            if img is None:
                continue
            
            # Convert to RGB and then Tensor (C, H, W)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t = kornia.image_to_tensor(img_rgb, keepdim=True).float() / 255.0
            
            # Destination corners (portrait)
            pts_dst = np.array([[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]], dtype=np.float32)
            ordered = order_points_clockwise(corners)
            oriented = _orient_for_target_canvas(ordered, self.width, self.height)
            pts_src = np.array(oriented, dtype=np.float32)
            
            # Get perspective transform
            M = cv2.getPerspectiveTransform(pts_src, pts_dst)
            tensors.append(img_t)
            matrices.append(torch.from_numpy(M).float())
            
        if not tensors:
            return []
            
        batch_t = torch.cat(tensors).to(self.device)
        batch_m = torch.stack(matrices).to(self.device)
        
        # Warp on GPU
        warped = kornia.geometry.transform.warp_perspective(batch_t, batch_m, (self.height, self.width))
        
        # Move back to CPU
        images: List[np.ndarray] = []
        for w in warped.cpu():
            rgb = kornia.tensor_to_image(w * 255.0).astype(np.uint8)
            images.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return images
