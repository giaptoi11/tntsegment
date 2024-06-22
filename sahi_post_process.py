import tifffile
import cv2 
import numpy as np
import glob
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict

detection_model_seg = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path="/home/toi/myproject/best.pt",
    confidence_threshold=0.5,
    device="cpu", # or 'cuda:0'
)

import os
path_folder_rgb = "/home/toi/myproject/test_images_rgb"
path_folder_rgb_result = "/home/toi/myproject/test_images_rgb_sahi_result"
os.makedirs(path_folder_rgb, exist_ok=True)
os.makedirs(path_folder_rgb_result, exist_ok=True)
# for path in glob.glob("/home/toi/myproject/test_images/images/*"):
#     filename = os.path.basename(path)
#     image = tifffile.imread(path)
#     img = cv2.normalize(image[:,: , 1:4], None, 0, 255, cv2.NORM_MINMAX, ).astype(np.uint8)
#     image = img.copy()
#     cv2.imwrite(f"{path_folder_rgb}/{filename}.jpg", img)
data = {"images": []}
for index, path in enumerate(glob.glob(f"{path_folder_rgb}/*")):
    filename = os.path.basename(path).split(".jpg")[0]
    
    result = get_sliced_prediction(
        path,
        detection_model_seg,
        slice_height = 320,
        slice_width = 320,
        overlap_height_ratio = 0.2,
        overlap_width_ratio = 0.2
    )
    
    object_prediction_list = result.object_prediction_list
    
    img_result = cv2.imread(path)
    
    data_seg = []
    for obj in object_prediction_list:
        mask = np.array(obj.mask.segmentation[0], dtype=np.int32)
        data_seg.append({
            "class": "field", 
            "segmentation": mask.copy().reshape(-1).tolist()
            })
        # mask = np.array([0, 0, 100, 100, 20, 20], dtype=np.int32)
        mask = mask.reshape(-1, 2)
        cv2.polylines(img_result, [mask], True, (0, 255, 0), 2)
    
    cv2.imwrite(f"{path_folder_rgb_result}/{os.path.basename(path)}", img_result)
    data["images"].append({
        "file_name": filename,
        "annotations": data_seg
    })
    print(index, os.path.basename(path))
    # break
    
import json
with open("submitsion_sh.json", "w") as f:
    json.dump(data, f)