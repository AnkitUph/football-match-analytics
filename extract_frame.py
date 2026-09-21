import cv2

cap = cv2.VideoCapture("media/test_clips/test_11.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Couldn't read frame — check the video path/frame number")

cv2.imwrite("calibration_test_frame.jpg", frame)
print("Saved calibration_test_frame.jpg", frame.shape)
