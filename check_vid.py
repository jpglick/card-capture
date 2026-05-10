import cv2
import sys
import os

video_path = os.path.expanduser("~/IMG_5872.MOV")
cap = cv2.VideoCapture(video_path)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Video width: {width}, height: {height}")

cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
if ret:
    print(f"Read frame shape: {frame.shape}")
else:
    print("Could not read frame")
