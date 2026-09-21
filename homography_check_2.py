import json
import numpy as np
import cv2

from apps.matches.pitch_landmarks_32 import (
    ROBOFLOW_KEYPOINTS_32_BY_INDEX,
    PITCH_LENGTH_M, PITCH_WIDTH_M,
    PENALTY_BOX_LENGTH_M, PENALTY_BOX_WIDTH_M,
    GOAL_BOX_LENGTH_M, GOAL_BOX_WIDTH_M,
    CENTRE_CIRCLE_RADIUS_M,
)

CONF_THRESHOLD = 0.5

with open("inference_result_2.json") as f:
    result = json.load(f)

keypoints = result["predictions"][0]["keypoints"]

real_pts, pixel_pts, used_indices = [], [], []

for kp in keypoints:
    idx = kp["class_id"] + 1
    conf = kp["confidence"]
    if conf < CONF_THRESHOLD:
        continue
    entry = ROBOFLOW_KEYPOINTS_32_BY_INDEX.get(idx)
    if entry is None:
        continue
    real_pts.append([entry["x"], entry["y"]])
    pixel_pts.append([kp["x"], kp["y"]])
    used_indices.append(idx)

print(f"Using {len(real_pts)} correspondences: {used_indices}")

real_pts = np.array(real_pts, dtype=np.float32)
pixel_pts = np.array(pixel_pts, dtype=np.float32)

H, mask = cv2.findHomography(real_pts, pixel_pts, method=0)
print("Homography matrix:")
print(H)

def project(x, y):
    pt = np.array([[[x, y]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)
    return tuple(out[0, 0].astype(int))

def draw_line(frame, p1, p2, color=(0, 255, 255), thickness=2, n=20):
    for i in range(n):
        t0, t1 = i / n, (i + 1) / n
        x0 = p1[0] + (p2[0] - p1[0]) * t0
        y0 = p1[1] + (p2[1] - p1[1]) * t0
        x1 = p1[0] + (p2[0] - p1[0]) * t1
        y1 = p1[1] + (p2[1] - p1[1]) * t1
        cv2.line(frame, project(x0, y0), project(x1, y1), color, thickness)

frame = cv2.imread("calibration_test_frame_2.jpg")

HL, HW = PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2
PBL, PBW = PENALTY_BOX_LENGTH_M, PENALTY_BOX_WIDTH_M / 2
GBL, GBW = GOAL_BOX_LENGTH_M, GOAL_BOX_WIDTH_M / 2
CCR = CENTRE_CIRCLE_RADIUS_M

draw_line(frame, (-HL, HW), (HL, HW))
draw_line(frame, (-HL, -HW), (HL, -HW))
draw_line(frame, (-HL, HW), (-HL, -HW))
draw_line(frame, (HL, HW), (HL, -HW))

draw_line(frame, (0, HW), (0, -HW), color=(255, 0, 255))

draw_line(frame, (-HL, PBW), (-HL + PBL, PBW))
draw_line(frame, (-HL, -PBW), (-HL + PBL, -PBW))
draw_line(frame, (-HL + PBL, PBW), (-HL + PBL, -PBW))

draw_line(frame, (HL, PBW), (HL - PBL, PBW))
draw_line(frame, (HL, -PBW), (HL - PBL, -PBW))
draw_line(frame, (HL - PBL, PBW), (HL - PBL, -PBW))

draw_line(frame, (-HL, GBW), (-HL + GBL, GBW), color=(0, 165, 255))
draw_line(frame, (-HL, -GBW), (-HL + GBL, -GBW), color=(0, 165, 255))
draw_line(frame, (-HL + GBL, GBW), (-HL + GBL, -GBW), color=(0, 165, 255))

draw_line(frame, (HL, GBW), (HL - GBL, GBW), color=(0, 165, 255))
draw_line(frame, (HL, -GBW), (HL - GBL, -GBW), color=(0, 165, 255))
draw_line(frame, (HL - GBL, GBW), (HL - GBL, -GBW), color=(0, 165, 255))

n_circle = 60
circle_pts_real = [
    (CCR * np.cos(2 * np.pi * i / n_circle), CCR * np.sin(2 * np.pi * i / n_circle))
    for i in range(n_circle + 1)
]
for i in range(n_circle):
    p1 = project(*circle_pts_real[i])
    p2 = project(*circle_pts_real[i + 1])
    cv2.line(frame, p1, p2, (0, 255, 0), 2)

for (x, y), idx in zip(pixel_pts, used_indices):
    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.putText(frame, str(idx), (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imwrite("homography_check_2.jpg", frame)
print("Saved homography_check_2.jpg")
