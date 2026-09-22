import cv2
import numpy as np
from apps.matches.models import Match, MatchCalibration
from apps.matches.pitch_landmarks_32 import (
    PITCH_LENGTH_M, PITCH_WIDTH_M,
    PENALTY_BOX_LENGTH_M, PENALTY_BOX_WIDTH_M,
    GOAL_BOX_LENGTH_M, GOAL_BOX_WIDTH_M,
    CENTRE_CIRCLE_RADIUS_M,
)

def verify_match_11():
    match = Match.objects.get(pk=11)
    video_path = match.video.original_video.path
    cap = cv2.VideoCapture(video_path)

    calibrations = list(match.calibrations.order_by("calibration_frame"))
    print(f"Match 11 has {len(calibrations)} anchors.")

    HL, HW = PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2
    PBL, PBW = PENALTY_BOX_LENGTH_M, PENALTY_BOX_WIDTH_M / 2
    GBL, GBW = GOAL_BOX_LENGTH_M, GOAL_BOX_WIDTH_M / 2
    CCR = CENTRE_CIRCLE_RADIUS_M

    for cal in calibrations:
        frame_idx = cal.calibration_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"Could not read frame {frame_idx}")
            continue

        image_pts = np.array([[p["pixel_x"], p["pixel_y"]] for p in cal.points], dtype=np.float32)
        pitch_pts = np.array([[p["pitch_x"], p["pitch_y"]] for p in cal.points], dtype=np.float32)

        # H maps image (pixel) -> pitch (m)
        H_img2pitch, mask = cv2.findHomography(image_pts, pitch_pts, cv2.RANSAC, 5.0)
        # H_pitch2img maps pitch (m) -> image (pixel)
        H_pitch2img, _ = cv2.findHomography(pitch_pts, image_pts, cv2.RANSAC, 5.0)

        if H_pitch2img is None:
            print(f"Anchor {frame_idx}: Homography computation failed!")
            continue

        # Compute reprojection errors
        reprojected_pixels = cv2.perspectiveTransform(pitch_pts.reshape(-1, 1, 2), H_pitch2img).reshape(-1, 2)
        errors = np.linalg.norm(image_pts - reprojected_pixels, axis=1)
        mean_err = np.mean(errors)
        max_err = np.max(errors)
        print(f"Anchor frame {frame_idx}: {len(cal.points)} points | Mean reprojection error: {mean_err:.2f}px | Max: {max_err:.2f}px")

        def project(x, y):
            pt = np.array([[[x, y]]], dtype=np.float32)
            out = cv2.perspectiveTransform(pt, H_pitch2img)
            return tuple(out[0, 0].astype(int))

        def draw_line(img, p1, p2, color=(0, 255, 255), thickness=2, n=20):
            for i in range(n):
                t0, t1 = i / n, (i + 1) / n
                x0 = p1[0] + (p2[0] - p1[0]) * t0
                y0 = p1[1] + (p2[1] - p1[1]) * t0
                x1 = p1[0] + (p2[0] - p1[0]) * t1
                y1 = p1[1] + (p2[1] - p1[1]) * t1
                pt0 = project(x0, y0)
                pt1 = project(x1, y1)
                # Check boundaries to avoid drawing crazy wild lines
                if -1000 < pt0[0] < 3000 and -1000 < pt0[1] < 3000 and -1000 < pt1[0] < 3000 and -1000 < pt1[1] < 3000:
                    cv2.line(img, pt0, pt1, color, thickness)

        # Pitch boundaries
        draw_line(frame, (-HL, HW), (HL, HW), color=(0, 255, 255))
        draw_line(frame, (-HL, -HW), (HL, -HW), color=(0, 255, 255))
        draw_line(frame, (-HL, HW), (-HL, -HW), color=(0, 255, 255))
        draw_line(frame, (HL, HW), (HL, -HW), color=(0, 255, 255))

        # Halfway line
        draw_line(frame, (0, HW), (0, -HW), color=(255, 0, 255), thickness=3)

        # Left penalty box & goal box
        draw_line(frame, (-HL, PBW), (-HL + PBL, PBW), color=(255, 255, 0))
        draw_line(frame, (-HL, -PBW), (-HL + PBL, -PBW), color=(255, 255, 0))
        draw_line(frame, (-HL + PBL, PBW), (-HL + PBL, -PBW), color=(255, 255, 0))
        draw_line(frame, (-HL, GBW), (-HL + GBL, GBW), color=(0, 165, 255))
        draw_line(frame, (-HL, -GBW), (-HL + GBL, -GBW), color=(0, 165, 255))
        draw_line(frame, (-HL + GBL, GBW), (-HL + GBL, -GBW), color=(0, 165, 255))

        # Right penalty box & goal box
        draw_line(frame, (HL, PBW), (HL - PBL, PBW), color=(255, 255, 0))
        draw_line(frame, (HL, -PBW), (HL - PBL, -PBW), color=(255, 255, 0))
        draw_line(frame, (HL - PBL, PBW), (HL - PBL, -PBW), color=(255, 255, 0))
        draw_line(frame, (HL, GBW), (HL - GBL, GBW), color=(0, 165, 255))
        draw_line(frame, (HL, -GBW), (HL - GBL, -GBW), color=(0, 165, 255))
        draw_line(frame, (HL - GBL, GBW), (HL - GBL, -GBW), color=(0, 165, 255))

        # Center circle
        n_circle = 60
        circle_pts_real = [
            (CCR * np.cos(2 * np.pi * i / n_circle), CCR * np.sin(2 * np.pi * i / n_circle))
            for i in range(n_circle + 1)
        ]
        for i in range(n_circle):
            pt0 = project(*circle_pts_real[i])
            pt1 = project(*circle_pts_real[i + 1])
            if -1000 < pt0[0] < 3000 and -1000 < pt0[1] < 3000 and -1000 < pt1[0] < 3000 and -1000 < pt1[1] < 3000:
                cv2.line(frame, pt0, pt1, (0, 255, 0), 2)

        # Draw detected keypoints as red dots
        for p in cal.points:
            px, py = int(p["pixel_x"]), int(p["pixel_y"])
            cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(frame, p["landmark_id"], (px + 6, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        out_name = f"match11_anchor{frame_idx}_overlay.jpg"
        cv2.imwrite(out_name, frame)
        print(f"Saved {out_name}")

    cap.release()

if __name__ == "__main__":
    verify_match_11()
