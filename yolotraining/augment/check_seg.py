import cv2
import numpy as np
import json

path = "/home/toi/myproject/yolotraining/data/augment_data/segment_parts/images/train_0_0.jpg"
img = cv2.imread(path)
h, w = img.shape[:2]
ratiow = 640/w
ratioh = 640/h

img = cv2.resize(img, (640, 640))

print(ratiow, ratioh)
segment_path = path.replace("images", "labels").replace("jpg", "json")
with open(segment_path, "r") as f:
    segment = json.loads(f.read())["segmentation"]
                        
seg_numpy = np.array([segment]).reshape(-1, 2)
seg_numpy[:, 0] = seg_numpy.copy()[:, 0]*ratiow
seg_numpy[:, 1] = seg_numpy.copy()[:, 1]*ratioh
cv2.polylines(img, [seg_numpy.astype(np.int32)],True, (255, 255, 255), 1)
cv2.imshow("img", img)

if cv2.waitKey(0) & 0xFF == ord("q"):
    cv2.destroyAllWindows()