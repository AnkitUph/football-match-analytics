from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2

model_path = hf_hub_download(repo_id='Adit-jain/Soccana_Keypoint', filename='Model/weights/best.pt')
print('Model path:', model_path)

model = YOLO(model_path)

frame = cv2.imread('/tmp/test_frame.jpg')
results = model.predict(frame, conf=0.5)
r = results[0]

if r.keypoints is not None and len(r.keypoints.xy) > 0:
    print('Detected', r.keypoints.xy.shape[1], 'keypoints total')
    for i in range(r.keypoints.xy.shape[1]):
        x, y = r.keypoints.xy[0][i].tolist()
        conf = r.keypoints.conf[0][i].item() if r.keypoints.conf is not None else None
        if conf and conf >= 0.5:
            print(f'  HIGH CONF kp[{i}] = ({x:.1f}, {y:.1f})  conf={conf:.3f}')
else:
    print('No keypoints detected on this frame')

annotated = r.plot()
cv2.imwrite('/tmp/keypoint_test_output.jpg', annotated)
print('Saved visualization to /tmp/keypoint_test_output.jpg')
