import os
import base64
import requests
import json

API_KEY = os.environ["ROBOFLOW_API_KEY"]
MODEL_ID = "football-field-detection-f07vi/14"

with open("calibration_test_frame_2.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

resp = requests.post(
    f"https://detect.roboflow.com/{MODEL_ID}",
    params={"api_key": API_KEY},
    data=img_b64,
    headers={"Content-Type": "application/x-www-form-urlencoded"},
)

result = resp.json()

with open("inference_result_2.json", "w") as f:
    json.dump(result, f, indent=2)

keypoints = result["predictions"][0]["keypoints"]
print(f"Total keypoints returned: {len(keypoints)}")
print()
print(f"{'class_id':>8}  {'class':>6}  {'x':>8}  {'y':>8}  {'confidence':>10}")
for kp in keypoints:
    print(f"{kp['class_id']:>8}  {kp['class']:>6}  {kp['x']:>8.1f}  {kp['y']:>8.1f}  {kp['confidence']:>10.4f}")
