import cv2
import os
import sys
import numpy as np

from card_capture.detectors import CardcaptorUltralyticsDetector
from card_capture.sampler import AdaptivePresenceSampler
from card_capture.models import FramePacket

def main():
    video_path = os.path.expanduser("~/IMG_5872.MOV")
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    sampler = AdaptivePresenceSampler(video_path=video_path)
    detector = CardcaptorUltralyticsDetector(confidence_threshold=0.5)
    model = detector._load_model()
    
    print("Running sampler to get frames...")
    frames_iter = sampler.sample()
    
    sampled_frames = []
    for i, frame in enumerate(frames_iter):
        sampled_frames.append(frame)
        if i >= 4: # Just check first 5 frames
            break
    
    print(f"Got {len(sampled_frames)} frames to test")
    
    detect_images = []
    scale_factors = []
    frames = []
    for i, f in enumerate(sampled_frames):
        packet = FramePacket(
            frame_index=f.frame_index,
            timestamp_ms=f.timestamp_ms,
            image=f.image,
            width=f.width,
            height=f.height,
            triage_metrics={}
        )
        frames.append(packet)
        original_h, original_w = f.image.shape[:2]
        if original_w > detector.detection_width:
            scaled_w = detector.detection_width
            scaled_h = max(1, int(round(original_h * detector.detection_width / original_w)))
            detect_image = cv2.resize(f.image, (scaled_w, scaled_h))
            scale_x = original_w / scaled_w
            scale_y = original_h / scaled_h
        else:
            detect_image = f.image
            scale_x = 1.0
            scale_y = 1.0
        detect_images.append(detect_image)
        scale_factors.append((scale_x, scale_y, f.width, f.height))

    results = model(detect_images, conf=0.5, verbose=False)
    for frame, result, (scale_x, scale_y, frame_width, frame_height) in zip(frames, results, scale_factors):
        print(f"\nFrame {frame.frame_index}: {frame.width}x{frame.height}")
        obb = getattr(result, "obb", None)
        if obb is None or obb.conf is None:
            print("No OBB found.")
            continue
        polygons = obb.xyxyxyxy.cpu().numpy()
        confidences = obb.conf.cpu().numpy()
        for polygon_array, confidence in zip(polygons, confidences):
            confidence_float = float(confidence)
            print(f"  Confidence: {confidence_float}")
            if confidence_float < 0.5:
                print("  Failed confidence check")
                continue
            polygon = tuple(
                (float(point[0]) * scale_x, float(point[1]) * scale_y)
                for point in polygon_array
            )
            if len(polygon) != 4:
                print("  Failed len(polygon) == 4")
                continue

            poly_arr = np.array(polygon, dtype=np.float32)
            poly_area = cv2.contourArea(poly_arr)
            frame_area = frame.width * frame.height
            
            print(f"  Area check: poly_area={poly_area}, frame_area={frame_area}, min={0.1*frame_area}, max={0.8*frame_area}")
            if not (0.1 * frame_area <= poly_area <= 0.8 * frame_area):
                print(f"  Failed area check: {poly_area} vs {frame_area} (needs to be between 10% and 80%)")
                continue

            width_top = np.linalg.norm(poly_arr[1] - poly_arr[0])
            width_bottom = np.linalg.norm(poly_arr[2] - poly_arr[3])
            height_right = np.linalg.norm(poly_arr[2] - poly_arr[1])
            height_left = np.linalg.norm(poly_arr[3] - poly_arr[0])
            w = max(width_top, width_bottom)
            h = max(height_right, height_left)
            ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            
            print(f"  Ratio check: w={w}, h={h}, ratio={ratio}")
            if not (0.50 <= ratio <= 0.95):
                print(f"  Failed ratio check: {ratio} not in 0.50-0.95")
                continue
            
            print("  Passed all checks!")

if __name__ == '__main__':
    main()
