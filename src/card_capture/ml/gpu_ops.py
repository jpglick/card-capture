"""Batched GPU image primitives for quality scoring. Inputs are torch tensors
(N,H,W,3) uint8 BGR on any device; ops stay on that device. cv2-equivalent
within float tolerance (numerics intentionally differ — see plan)."""
from __future__ import annotations
from typing import List
import torch
import torch.nn.functional as F

_DCT_CACHE: dict = {}

def _dct_matrix(n: int, device, dtype) -> torch.Tensor:
    key = (n, str(device), str(dtype))
    if key not in _DCT_CACHE:
        k = torch.arange(n, device=device, dtype=dtype)
        i = k.view(n, 1)
        # Use torch.pi if available, otherwise 3.141592653589793
        pi = getattr(torch, 'pi', 3.141592653589793)
        D = torch.cos((2 * k.view(1, n) + 1) * i * pi / (2 * n))
        D *= torch.sqrt(torch.tensor(2.0 / n, device=device, dtype=dtype))
        D[0] *= 1.0 / torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))
        _DCT_CACHE[key] = D
    return _DCT_CACHE[key]

def gpu_dct2(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal 2D DCT-II of a batch (N,n,n), matching cv2.dct on n×n floats."""
    n = x.shape[-1]
    D = _dct_matrix(n, x.device, x.dtype)
    return D @ x @ D.transpose(-1, -2)

def rgb_gray_batch(bgr_u8: torch.Tensor) -> torch.Tensor:
    """(N,H,W,3) BGR uint8 → (N,H,W) float gray, matching cv2 BGR2GRAY weights."""
    b, g, r = bgr_u8[..., 0].float(), bgr_u8[..., 1].float(), bgr_u8[..., 2].float()
    return 0.114 * b + 0.587 * g + 0.299 * r

def phash_batch(bgr_u8: torch.Tensor) -> List[str]:
    """16-char hex perceptual hashes, byte-compatible with
    VisualDeduplicator.compute_phash / .hamming_distance (which does int(h,16)).
    Mirrors it exactly: 20% margin inner-crop → gray → 32×32 area-resize → DCT →
    8×8 low-freq vs median → 64 bits → hex."""
    n = bgr_u8.shape[0]
    h, w = bgr_u8.shape[1], bgr_u8.shape[2]
    # inner crop: drop 20% on each side (matches compute_phash: int(h*0.2))
    by, bx = int(h * 0.2), int(w * 0.2)
    inner = bgr_u8[:, by:h - by, bx:w - bx, :]
    gray = rgb_gray_batch(inner).unsqueeze(1)                       # (N,1,h',w')
    
    # MPS workaround: adaptive_avg_pool2d (used by F.interpolate area) requires 
    # input size to be divisible by output size on MPS. 
    # Fall back to CPU for the resize if on MPS.
    if gray.device.type == "mps":
        orig_device = gray.device
        resized = F.interpolate(gray.cpu(), size=(32, 32), mode="area").to(orig_device).squeeze(1)
    else:
        resized = F.interpolate(gray, size=(32, 32), mode="area").squeeze(1)
        
    dct = gpu_dct2(resized)
    low = dct[:, :8, :8].reshape(n, 64)
    med = low.median(dim=1, keepdim=True).values
    bits = (low > med)
    out = []
    for i in range(n):
        bitstr = "".join("1" if b else "0" for b in bits[i].tolist())
        out.append(f"{int(bitstr, 2):016x}")                       # 16-char hex, like compute_phash
    return out

def glare_mask_batch(bgr_u8: torch.Tensor) -> torch.Tensor:
    """(N,H,W,3)→(N,H,W) bool, gray>200 (cv2 THRESH_BINARY @200)."""
    return rgb_gray_batch(bgr_u8) > 200

def glare_centroid_batch(bgr_u8: torch.Tensor):
    """Returns list of (x,y) or None — centroid of gray>200 mask (== cv2.moments)."""
    mask = (rgb_gray_batch(bgr_u8) > 200).float()
    n, h, w = mask.shape
    ys = torch.arange(h, device=mask.device).view(1, h, 1).float()
    xs = torch.arange(w, device=mask.device).view(1, 1, w).float()
    area = mask.sum(dim=[1, 2])
    cx = (mask * xs).sum(dim=[1, 2]) / area.clamp(min=1)
    cy = (mask * ys).sum(dim=[1, 2]) / area.clamp(min=1)
    out = []
    for i in range(n):
        out.append((float(cx[i]), float(cy[i])) if area[i] > 0 else None)
    return out

def laplacian_var_batch(gray_f: torch.Tensor) -> torch.Tensor:
    """(N,H,W) float → (N,) variance of 3×3 Laplacian (kornel matches cv2's)."""
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                     device=gray_f.device, dtype=gray_f.dtype).view(1, 1, 3, 3)
    lap = F.conv2d(gray_f.unsqueeze(1), k, padding=1).squeeze(1)
    return lap.reshape(lap.shape[0], -1).var(dim=1, unbiased=False)

def spatial_glare_batch(bgr_u8: torch.Tensor) -> torch.Tensor:
    """GPU APPROXIMATION of scoring._spatial_glare_score.

    cv2 used connectedComponents to find the LARGEST saturated blob. No clean GPU
    CC exists, so we approximate "largest dense blob fraction" via the product
    of total saturated fraction and maximum local density squared. This proxy
    penalizes large contiguous regions (clumps) while ignoring scattered noise.
    Returns (N,) in [0,1], 1=no glare. Documented divergence."""
    # saturation ~ V channel high; approximate V with max(B,G,R)
    v = bgr_u8.float().amax(dim=-1)                                 # (N,H,W)
    sat = (v > 240).float().unsqueeze(1)                            # (N,1,H,W)
    n, _, h, w = sat.shape

    # Total saturated fraction
    total_frac = sat.mean(dim=(2, 3)).squeeze(1)

    # Max local density via moderate kernel pooling
    ksz = max(1, min(h, w) // 8)
    dens = F.avg_pool2d(sat, kernel_size=ksz, stride=max(1, ksz // 2))
    max_dens = dens.amax(dim=[2, 3]).squeeze(1)

    # Proxy: total_frac scaled by clumpiness (max_dens^2).
    # For one large blob, max_dens=1.0, blob_frac=total_frac.
    # For scattered pixels, max_dens is low, blob_frac is negligible.
    blob_frac = total_frac * (max_dens ** 2)
    return (1.0 - blob_frac * 10.0).clamp(0.0, 1.0)

def complexity_batch(gray_f: torch.Tensor) -> torch.Tensor:
    """(N,H,W) float → (N,) complexity score (gray std-dev / 80)."""
    n = gray_f.shape[0]
    return (gray_f.reshape(n, -1).std(dim=1, unbiased=False) / 80.0).clamp(0.0, 1.0)

def border_purity_batch(gray_f: torch.Tensor) -> torch.Tensor:
    """(N,H,W) float → (N,) border purity score.
    Approximate cv2 _border_purity_score. Calculates std of outer 3% ring
    vs interior. Returns (N,) in [0,1]."""
    n, h, w = gray_f.shape
    bw = max(2, int(round(min(h, w) * 0.03)))
    if h <= 2 * bw or w <= 2 * bw:
        return torch.full((n,), 0.5, device=gray_f.device)

    # Interior
    interior = gray_f[:, bw:h-bw, bw:w-bw]
    interior_std = interior.reshape(n, -1).std(dim=1, unbiased=False)

    # Ring - calculate sum and sum of squares for the whole image and the interior
    # std = sqrt(mean(x^2) - mean(x)^2)
    full_sum = gray_f.reshape(n, -1).sum(dim=1)
    full_sq_sum = (gray_f**2).reshape(n, -1).sum(dim=1)
    full_count = h * w

    int_sum = interior.reshape(n, -1).sum(dim=1)
    int_sq_sum = (interior**2).reshape(n, -1).sum(dim=1)
    int_count = (h - 2*bw) * (w - 2*bw)

    ring_sum = full_sum - int_sum
    ring_sq_sum = full_sq_sum - int_sq_sum
    ring_count = full_count - int_count

    ring_mean = ring_sum / ring_count
    ring_var = (ring_sq_sum / ring_count) - (ring_mean ** 2)
    ring_std = torch.sqrt(ring_var.clamp(min=0))

    ratio = ring_std / (interior_std + 1e-3)
    return (1.0 - ratio).clamp(0.0, 1.0)


def gpu_roi_mean_abs_diff(frame_t: torch.Tensor, bg_t: torch.Tensor, bbox: Tuple[int, int, int, int]) -> torch.Tensor:
    """Mean absolute difference between frame and background within a bounding box.
    Used for fast GPU-resident novelty scoring.
    frame_t, bg_t are (C, H, W) tensors. bbox is (x0, y0, x1, y1).
    Returns a 0-dim tensor.
    """
    x0, y0, x1, y1 = bbox
    # Ensure within bounds
    h, w = frame_t.shape[1], frame_t.shape[2]
    x0 = max(0, min(x0, w - 1)); x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1)); y1 = max(y0 + 1, min(y1, h))
    
    roi_f = frame_t[:, y0:y1, x0:x1]
    roi_b = bg_t[:, y0:y1, x0:x1]
    return torch.abs(roi_f - roi_b).mean()


def yuv_nv12_to_rgb_batch(nv12_u8: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(N, 1.5*H, W) uint8 NV12 tensor → (N,H,W,3) uint8 RGB.
    Performs hardware-accelerated color conversion on GPU (MPS/CUDA).
    Returns RGB for direct YOLO tensor inference.
    """
    n = nv12_u8.shape[0]
    # Slicing the Y and interleaved UV planes
    y = nv12_u8[:, :h, :].unsqueeze(1).float()      # (N, 1, H, W)
    uv = nv12_u8[:, h:, :].float()                   # (N, H/2, W)
    
    # De-interleave UV plane: U is at even cols, V is at odd cols
    u_plane = uv[:, :, 0::2].unsqueeze(1)            # (N, 1, H/2, W/2)
    v_plane = uv[:, :, 1::2].unsqueeze(1)            # (N, 1, H/2, W/2)
    
    # Upsample chroma to match luma resolution
    u = F.interpolate(u_plane, size=(h, w), mode='nearest')
    v = F.interpolate(v_plane, size=(h, w), mode='nearest')
    
    # BT.601 coefficients
    y_shifted = (y - 16.0) * 1.164
    
    # R = 1.164(Y-16) + 1.596(V-128)
    r = y_shifted + 1.596 * (v - 128.0)
    # G = 1.164(Y-16) - 0.813(V-128) - 0.391(U-128)
    g = y_shifted - 0.813 * (v - 128.0) - 0.391 * (u - 128.0)
    # B = 1.164(Y-16) + 2.018(U-128)
    b = y_shifted + 2.018 * (u - 128.0)
    
    # Combine to RGB and clamp to valid uint8 range
    # Final shape: (N, H, W, 3)
    return torch.cat([r, g, b], dim=1).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def yuv_nv12_to_bgr_batch(nv12_u8: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(N, 1.5*H, W) uint8 NV12 tensor → (N,H,W,3) uint8 BGR.
    Same as above but returns BGR. Used for legacy paths.
    """
    rgb = yuv_nv12_to_rgb_batch(nv12_u8, h, w)
    return rgb[:, :, :, [2, 1, 0]].contiguous()
