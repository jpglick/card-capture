"""Generate synthetic foil and non-foil test fixtures for threshold calibration."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def generate_non_foil_fixture(output_path: Path) -> None:
    """Generate a synthetic non-foil card fixture (stable Laplacian across frames).

    Non-foil cards have consistent edge structures that don't shift much across frames,
    resulting in low Laplacian variance.
    """
    # Create a base card-like image: rectangular shape with some edge texture
    frame_size = (750, 1050)

    # Create 4 frames with slight variations but stable edge structure
    for i in range(4):
        # Start with a card-like base: mid-gray rectangle on darker background
        frame = np.ones((*frame_size, 3), dtype=np.uint8) * 40

        # Card area: mid-gray with subtle gradient
        card_area = frame[100:650, 150:900]
        base_color = 120 + (i % 2) * 3  # Very slight variation per frame
        card_area[:] = base_color

        # Add subtle edge texture (consistent across frames): slight border darkening
        card_area[0:5, :] = base_color - 30
        card_area[-5:, :] = base_color - 30
        card_area[:, 0:5] = base_color - 30
        card_area[:, -5:] = base_color - 30

        # Add fine-grained noise (same pattern every frame - non-foil)
        np.random.seed(42)  # Fixed seed for reproducibility
        noise = np.random.normal(0, 3, card_area.shape).astype(np.int16)
        card_area[:] = np.clip(card_area.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        frame[100:650, 150:900] = card_area

        # Save frame
        frame_path = output_path / f"non_foil_{i}.png"
        cv2.imwrite(str(frame_path), frame)
        print(f"Generated {frame_path}")


def generate_foil_fixture(output_path: Path) -> None:
    """Generate a synthetic foil card fixture (shifting Laplacian across frames).

    Foil cards have high-frequency content (holographic shimmer) that shifts
    between frames, resulting in high Laplacian variance.
    """
    frame_size = (750, 1050)

    # Create 4 frames with different high-frequency patterns (simulating holographic shift)
    for i in range(4):
        # Start with a card-like base: mid-gray rectangle on darker background
        frame = np.ones((*frame_size, 3), dtype=np.uint8) * 40

        # Card area: mid-gray base
        card_area = frame[100:650, 150:900]
        base_color = 120
        card_area[:] = base_color

        # Add subtle edge texture
        card_area[0:5, :] = base_color - 30
        card_area[-5:, :] = base_color - 30
        card_area[:, 0:5] = base_color - 30
        card_area[:, -5:] = base_color - 30

        # Add **different** high-frequency patterns per frame (simulating holographic shimmer)
        # Use frame index in random seed to get different patterns
        np.random.seed(42 + i)  # Different seed per frame

        # Generate high-frequency pattern with larger amplitude than non-foil
        high_freq = np.random.normal(0, 12, card_area.shape).astype(np.int16)

        # Also add some periodic patterns to simulate reflective surface
        y_idx = np.arange(card_area.shape[0])[:, np.newaxis]
        x_idx = np.arange(card_area.shape[1])[np.newaxis, :]

        # Offset the patterns per frame to simulate movement
        offset = (i * 3) % 20
        periodic = 15 * np.sin(2 * np.pi * (y_idx + offset) / 30) * np.cos(2 * np.pi * (x_idx + offset) / 50)

        # Convert periodic to (H, W, 3) to match card_area shape
        periodic_3d = np.stack([periodic, periodic, periodic], axis=2)

        card_area[:] = np.clip(
            card_area.astype(np.int16) + high_freq + periodic_3d.astype(np.int16),
            0,
            255
        ).astype(np.uint8)

        frame[100:650, 150:900] = card_area

        # Save frame
        frame_path = output_path / f"foil_{i}.png"
        cv2.imwrite(str(frame_path), frame)
        print(f"Generated {frame_path}")


def main():
    """Generate all foil/non-foil fixtures."""
    # Create directories
    foil_dir = Path("tests/fixtures/foil/foil")
    non_foil_dir = Path("tests/fixtures/foil/non_foil")

    foil_dir.mkdir(parents=True, exist_ok=True)
    non_foil_dir.mkdir(parents=True, exist_ok=True)

    print("Generating non-foil fixtures...")
    for i in range(3):
        non_foil_subdir = non_foil_dir / f"non_foil_{i}"
        non_foil_subdir.mkdir(parents=True, exist_ok=True)
        generate_non_foil_fixture(non_foil_subdir)

    print("\nGenerating foil fixtures...")
    for i in range(3):
        foil_subdir = foil_dir / f"foil_{i}"
        foil_subdir.mkdir(parents=True, exist_ok=True)
        generate_foil_fixture(foil_subdir)

    print("\nAll fixtures generated!")


if __name__ == "__main__":
    main()
