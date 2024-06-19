from PIL import Image
import numpy as np
import os
from glob import glob
import cv2




output_dir = '/home/toi/myproject/yolotraining/data/data_split640'
os.makedirs(output_dir, exist_ok=True)
for image_path in glob('/home/toi/myproject/yolotraining/data/data/images/*'):
    # Load the image
    image_np = cv2.imread(image_path)
    h,w = image_np.shape[:2]
    # Load the annotations
    annotations_path = image_path.replace('.jpg', '.txt').replace("images", "labels")
    with open(annotations_path, 'r') as f:
        annotations = f.readlines()
    '''
    -----------------
    |       |       |
    |  tl   |  tr   |
    |       |       |
    -----------------
    |       |       |
    |  bl   |  br   |
    |       |       |
    -----------------
    '''
    
    tile_tl = image_np[0:h//2, 0:w//2]
    tile_tr = image_np[0:h//2, w//2:]
    tile_bl = image_np[h//2:, 0:w//2]
    tile_br = image_np[h//2:, w//2:]
    split_part = [tile_tl, tile_tr, tile_bl, tile_br]
    part = [(0, h//2, 0, w//2), (0, h//2, w//2, w), (h//2, h, 0, w//2), (h//2, h, w//2, h)]  
    for idx, (y1, y2, x1, x2) in enumerate(part):
        cv2.imwrite(f'{output_dir}/{idx}_{x1}_{y1}_{x2}_{y2}.jpg', split_part[idx])
        new_polygon = open(f'{output_dir}/{idx}_{x1}_{y1}_{x2}_{y2}.txt', 'w')
        mask = np.zeros_like(image_np)
        mask[y1:y2, x1:x2] = np.ones_like(mask[y1:y2, x1:x2])
        
        for line in annotations:
            debug_pts = []
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            polygon_norm = np.array(parts[1:])
            polygon = []
            for i in range(0, len(polygon_norm), 2):
                mask_poly = np.zeros_like(image_np)
                x = polygon_norm[i]*w
                y = polygon_norm[i+1]*h
                polygon.append((x, y))
            polygon = np.array(polygon)
            mask_poly = cv2.fillPoly(mask_poly, [polygon.reshape((-1, 1, 2)).astype(np.int32)], (1, 1, 1))

            intersect = mask*mask_poly
            if 10 < intersect.sum():
                crop_mask_poly = (mask_poly[y1:y2, x1:x2, 0]).astype(np.uint8)
                contours, hierarchy = cv2.findContours(crop_mask_poly*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours = contours[0] 
                anno_txt = '0'
                for (x, y) in contours.reshape((-1, 2)):
                    anno_txt += f' {x} {y}'

                debug_pts.append(contours)

            if len(anno_txt.split()) > 6:
                new_polygon.write(anno_txt + '\n')
                debug_pts = np.array(debug_pts)
                split_part[idx] = cv2.polylines(split_part[idx], [debug_pts.reshape((-1, 1, 2)).astype(np.int32)], True, (np.random.random((3))*255).astype(np.uint8).tolist(), 2)