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
        self,
        batch_u8: "torch.Tensor",
        matrices_np: List[np.ndarray],
        rotate_180: bool,
        return_gpu: bool = False,
    ) -> Union[List[np.ndarray], "torch.Tensor"]:
        """Warp a stacked uint8 (B,H,W,3) BGR tensor (already on device) → list of BGR crops or batched tensor.

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

        out = (warped[:, [2, 1, 0], :, :] * 255.0).clamp_(0, 255).to(torch.uint8)
        out = out.permute(0, 2, 3, 1).contiguous()  # (N,H,W,3) BGR
        del warped

        if rotate_180:
            out = torch.rot90(out, 2, dims=[1, 2])

        if return_gpu:
            return out  # device tensor, stays resident

        np_out = out.cpu().numpy()
        return [np_out[i] for i in range(np_out.shape[0])]

    def warp_canonical_batch(
        self,
        batch_data: List[Tuple[Union[str, np.ndarray], List[Point]]],
        rotate_180: bool = True,
        return_gpu: bool = False,
        chunk_size: int = 8,
    ) -> Union[List[np.ndarray], "torch.Tensor"]:
        """Warp from numpy images (or image paths). Uploads to GPU, then warps.

        Optimized 2026-05-24 — measured 1138 ms/batch in production.
        Chunked processing added to prevent OOM on large 4K batches.
        """
        if not batch_data:
            return [] if not return_gpu else torch.empty(0).to(self.device)

        results: List[np.ndarray] = []
        gpu_results: List["torch.Tensor"] = []

        # Process in chunks to avoid massive memory allocation
        for i in range(0, len(batch_data), chunk_size):
            chunk = batch_data[i : i + chunk_size]
            imgs: List[np.ndarray] = []
            mats: List[np.ndarray] = []
            
            for image_or_path, corners in chunk:
                img = image_or_path if isinstance(image_or_path, np.ndarray) else cv2.imread(image_or_path)
                if img is None:
                    continue
                imgs.append(img)
                mats.append(self._perspective_matrix(corners))
            
            if not imgs:
                continue
                
            batch_u8 = torch.from_numpy(np.stack(imgs, axis=0)).to(self.device, non_blocking=True)
            res = self._warp_from_stacked(batch_u8, mats, rotate_180, return_gpu=return_gpu)
            
            if return_gpu:
                gpu_results.append(res)
            else:
                results.extend(res)

        if return_gpu:
            return torch.cat(gpu_results, dim=0) if gpu_results else torch.empty(0).to(self.device)
        return results

    def warp_canonical_batch_gpu(
        self,
        batch_data: List[Tuple[Union["torch.Tensor", np.ndarray], List[Point]]],
        rotate_180: bool = True,
        return_gpu: bool = False,
    ) -> Union[List[np.ndarray], "torch.Tensor"]:
        """Warp from GPU-resident uint8 (H,W,3) BGR tensors or numpy arrays.
        Highly optimized for zero-download 4K pipelines: groups crops by source 
        frame and uses tensor.expand() to avoid massive VRAM copies.
        """
        if not batch_data:
            return [] if not return_gpu else torch.empty(0).to(self.device)

        # Group corners by tensor to avoid duplicate float conversions
        # Use object data_ptr() for tensors, or id() for numpy arrays
        grouped = {}
        order_map = []  # To reconstruct the original output order
        for idx, (img, corners) in enumerate(batch_data):
            if img is None:
                continue
            if isinstance(img, np.ndarray):
                t = torch.from_numpy(img).to(self.device, non_blocking=True)
                key = t.data_ptr()
                grouped[key] = {'tensor': t, 'indices': [], 'mats': []}
            else:
                key = img.data_ptr()
                if key not in grouped:
                    grouped[key] = {'tensor': img, 'indices': [], 'mats': []}
            
            grouped[key]['indices'].append(idx)
            grouped[key]['mats'].append(self._perspective_matrix(corners))

        results = [None] * len(batch_data)
        for group in grouped.values():
            t_u8 = group['tensor'] # (H, W, 3) uint8 BGR
            # Convert to (1, 3, H, W) float RGB
            t_float = t_u8.permute(2, 0, 1)[[2, 1, 0], :, :].unsqueeze(0).float() / 255.0
            
            n_crops = len(group['mats'])
            # Expand to (N, 3, H, W) without allocating memory!
            t_expanded = t_float.expand(n_crops, -1, -1, -1)
            
            mats = torch.from_numpy(np.stack(group['mats'], axis=0).astype(np.float32)).to(
                self.device, non_blocking=True
            )
            
            warped = kornia.geometry.transform.warp_perspective(
                t_expanded, mats, (self.height, self.width)
            )
            
            # Back to (N, H, W, 3) uint8 BGR
            out = (warped[:, [2, 1, 0], :, :] * 255.0).clamp_(0, 255).to(torch.uint8)
            out = out.permute(0, 2, 3, 1).contiguous()
            
            if rotate_180:
                out = torch.rot90(out, 2, dims=[1, 2])
                
            if not return_gpu:
                out = out.cpu().numpy()
                
            for i, orig_idx in enumerate(group['indices']):
                results[orig_idx] = out[i]

        valid_results = [r for r in results if r is not None]
        if return_gpu:
            return torch.stack(valid_results) if valid_results else torch.empty(0).to(self.device)
        return valid_results
