"""Resource monitoring: host fingerprinting and per-run CPU/RAM/GPU sampling."""
from __future__ import annotations

import json
import platform
import socket
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional


def get_host_info() -> dict:
    """Fingerprint the current machine. Never raises."""
    info: dict = {}
    try:
        info["hostname"] = socket.gethostname()
    except Exception:
        info["hostname"] = "unknown"

    try:
        info["platform"] = f"{platform.system()} {platform.machine()}"
    except Exception:
        info["platform"] = "unknown"

    # CPU
    try:
        import psutil
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
    except Exception:
        info["cpu_count_physical"] = None
        info["cpu_count_logical"] = None

    # RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["mem_total_gb"] = round(mem.total / 1e9, 2)
    except Exception:
        info["mem_total_gb"] = None

    # GPU device
    gpu_device = "cpu"
    try:
        from card_capture.detectors import probe_torch_device_status
        status = probe_torch_device_status()
        gpu_device = status.resolved
    except Exception:
        pass
    info["gpu_device"] = gpu_device

    # GPU name and VRAM
    gpu_name: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_is_unified = False

    if gpu_device == "cuda":
        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            vram_total_gb = round(props.total_memory / 1e9, 2)
        except Exception:
            pass
    elif gpu_device == "mps":
        try:
            raw = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                timeout=5,
            )
            data = json.loads(raw)
            gpu_name = data["SPDisplaysDataType"][0].get("sppci_model", "Apple GPU")
        except Exception:
            gpu_name = "Apple GPU (MPS)"
        # MPS uses unified memory
        vram_total_gb = info.get("mem_total_gb")
        vram_is_unified = True

    info["gpu_name"] = gpu_name
    info["vram_total_gb"] = vram_total_gb
    info["vram_is_unified"] = vram_is_unified

    return info


class ResourceSampler:
    """Samples CPU/RAM/GPU every 2 seconds and writes to run_resource_samples."""

    INTERVAL_S = 2.0

    def __init__(self, run_id: str, db_path: Path, start_time: float) -> None:
        self.run_id = run_id
        self.db_path = db_path
        self.start_time = start_time
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop,
            name="resource-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def _loop(self) -> None:
        # Prime cpu_percent so the first real reading isn't 0.0
        try:
            import psutil
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

        while not self._stop_event.wait(timeout=self.INTERVAL_S):
            self._sample()

    def _sample(self) -> None:
        elapsed_s = time.monotonic() - self.start_time

        cpu_pct: Optional[float] = None
        mem_used_mb: Optional[float] = None
        mem_pct: Optional[float] = None
        gpu_pct: Optional[float] = None
        vram_used_mb: Optional[float] = None

        try:
            import psutil
            cpu_pct = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            mem_pct = vm.percent
            mem_used_mb = vm.used / 1e6
        except Exception:
            pass

        # NVIDIA via pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_pct = float(util.gpu)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_used_mb = mem_info.used / 1e6
        except Exception:
            pass

        # Apple Silicon via ioreg (no sudo required)
        if gpu_pct is None:
            try:
                import re
                out = subprocess.check_output(
                    ["ioreg", "-r", "-c", "IOAccelerator", "-d", "2"],
                    text=True, timeout=2, stderr=subprocess.DEVNULL,
                )
                m = re.search(r'"PerformanceStatistics"\s*=\s*\{([^}]+)\}', out)
                if m:
                    pairs = dict(re.findall(r'"([^"]+)"=(\d+)', m.group(1)))
                    if "Device Utilization %" in pairs:
                        gpu_pct = float(pairs["Device Utilization %"])
                    if "In use system memory" in pairs:
                        vram_used_mb = int(pairs["In use system memory"]) / 1e6
            except Exception:
                pass

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    "INSERT INTO run_resource_samples "
                    "(run_id, elapsed_s, cpu_pct, mem_used_mb, mem_pct, gpu_pct, vram_used_mb, stage) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (self.run_id, elapsed_s, cpu_pct, mem_used_mb, mem_pct, gpu_pct, vram_used_mb, self.current_stage),
                )
        except Exception:
            pass
 pass
