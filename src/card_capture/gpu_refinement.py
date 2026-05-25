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

    def _perspective_matrix(self, corners: List[Point]) -> np.ndarray:
        """CPU-side 3x3 perspective matrix mapping the card quad → portrait canvas."""
        pts_dst = np.array(
            [[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]],
            dtype=np.float32,
        )
        ordered = order_points_clockwise(corners)
        oriented = _orient_for_target_canvas(ordered, self.width, self.height)
        pts_src = np.array(oriented, dtype=np.float32)
        return cv2.getPerspectiveTransform(pts_src, pts_dst)

    def _warp_from_stacked(
        self, batch_u8: "torch.Tensor", matrices_np: List[np.ndarray], rotate_180: bool
    ) -> List[np.ndarray]:
        """Warp a stacked uint8 (B,H,W,3) BGR tensor (already on device) → list of BGR crops.

        Single GPU warp core shared by the numpy and GPU-tensor entry points.
        Channel handling is unchanged from the original warp_canonical_batch:
        BGR in (index-swapped to RGB for kornia, swapped back to BGR on output).
        """
        batch_t = batch_u8.permute(0, 3, 1, 2)[:, [2, 1, 0], :, :].float() / 255.0
        del batch_u8

        batch_m = torch.from_numpy(np.stack(matrices_np, axis=0).astype(np.float32)).to(
            self.device, non_blocking=True
        )
        warped = kornia.geometry.transform.warp_perspective(
            batch_t, batch_m, (self.height, self.width)
        )
        del batch_t

        warped_u8 = (warped[:, [2, 1, 0], :, :] * 255.0).clamp_(0, 255).to(torch.uint8)
        warped_u8 = warped_u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        del warped

        images: List[np.ndarray] = []
        for bgr in warped_u8:
            if rotate_180:
                bgr = cv2.rotate(bgr, cv2.ROTATE_180)
            images.append(bgr)
        return images

    def warp_canonical_batch(
        self, batch_data: List[Tuple[Union[str, np.ndarray], List[Point]]], rotate_180: bool = True
    ) -> List[np.ndarray]:
        """Warp from numpy images (or image paths). Uploads to GPU, then warps.

        Optimized 2026-05-24 — measured 1138 ms/batch in production, dominated
        by per-candidate CPU float-conversion of 4K frames (~24MB uint8 each
        → 95MB float32). Uploads uint8 to GPU and does float/scale/channel
        reorder on the device. PCIe bandwidth per candidate drops ~4x; per-batch
        CPU work drops from ~520 ms (8 candidates) to ~40 ms.
        Shared warp core (_warp_from_stacked) also used by warp_canonical_batch_gpu.
        """
        imgs: List[np.ndarray] = []
        mats: List[np.ndarray] = []
        for image_or_path, corners in batch_data:
            img = image_or_path if isinstance(image_or_path, np.ndarray) else cv2.imread(image_or_path)
            if img is None:
                continue
            imgs.append(img)
            mats.append(self._perspective_matrix(corners))
        if not imgs:
            return []
        batch_u8 = torch.from_numpy(np.stack(imgs, axis=0)).to(self.device, non_blocking=True)
        return self._warp_from_stacked(batch_u8, mats, rotate_180)

    def warp_canonical_batch_gpu(
        self, batch_data: List[Tuple["torch.Tensor", List[Point]]], rotate_180: bool = True
    ) -> List[np.ndarray]:
        """Warp from GPU-resident uint8 (H,W,3) BGR tensors — no host→device upload.

        Each item's image is a torch tensor already on the GPU (a slice of the
        decoded decord batch). Skips the np.stack/from_numpy/.to() upload.

        Contract: all input tensors must be uint8 (H,W,3) and reside on the same device.
        Mixed-device inputs will raise at torch.stack(); no additional runtime check is added.
        """
        tensors: List["torch.Tensor"] = []
        mats: List[np.ndarray] = []
        for img_t, corners in batch_data:
            if img_t is None:
                continue
            tensors.append(img_t)
            mats.append(self._perspective_matrix(corners))
        if not tensors:
            return []
        batch_u8 = torch.stack(tensors, dim=0).to(self.device, non_blocking=True)
        return self._warp_from_stacked(batch_u8, mats, rotate_180)
