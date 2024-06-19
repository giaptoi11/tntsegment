import numpy as np
import glob
import cv2
from PIL import Image
import json
import tifffile
import shapely
import shapely.geometry
import uuid
import os

class Process():
    def __init__(self) -> None:
        self.image_folder = "/home/toi/myproject/yolotraining/data/raw_data/images"
        
        # self.image_rgb_folder = "/home/toi/myproject/yolotraining/data/data/images"
        # self.label_rgb_folder = "/home/toi/myproject/yolotraining/data/data/labels"
        
        # self.training_folder = "/home/toi/myproject/yolotraining/data/training_data"
        
        self.image_paths = glob.glob("/home/toi/myproject/yolotraining/data/raw_data/images/*")
        self.label_json_path = "/home/toi/myproject/yolotraining/data/raw_data/train_annotation.json"
        
        self.back_ground_folder = "/home/toi/myproject/yolotraining/data/augment_data/background"
        
        self.all_parts_folder  = "/home/toi/myproject/yolotraining/data/augment_data/segment_parts"
        os.makedirs(f"{self.all_parts_folder}/labels", exist_ok=True)
        os.makedirs(f"{self.all_parts_folder}/images", exist_ok=True)
    def load_json(self):        
        with open(self.label_json_path, "r") as f:
            self.data = json.loads(f.read())["images"]
            
    def read_image(self, image_name: str):
        img = tifffile.imread(f"{self.image_folder}/{image_name}")
        img = img[:, :, 1:4]
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return img
    
    def write_segment_path(self, img, img_name:str):
        cv2.imwrite(f"{self.all_parts_folder}/images/{img_name.split('.')[0]}.jpg", img)
        
    
    def convert_cv2_to_pil(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        return im_pil
    
    def box_from_polygon(self, polygon: np.array):
        """
        Return:
            x_min, y_min, x_max, y_max
        """
        x_min = int(np.min(polygon[:, 0]))
        x_max = int(np.max(polygon[:, 0]))
        y_min = int(np.min(polygon[:, 1]))
        y_max = int(np.max(polygon[:, 1]))

        bbox = [x_min, y_min, x_max, y_max]
        return bbox
    def save_part_images(self):
        """
        -------------
        |   |       |
        |----       |
        |           |
        |     ----  |
        |     |  |  |
        |     ----  |   
        -------------
        
        """
        file_name: str
        for dt in self.data:
            file_name = dt["file_name"]
            annotations = dt["annotations"]
            
            # f = open(f"{self.label_rgb_folder}/{file_name.split('.')[0]}.txt", "w")
            img = self.read_image(file_name)
            img_zeros = np.zeros_like(img)
            height, width, channels = img.shape
            
            # img = self.convert_cv2_to_pil(img.copy())
            for idx, annotation in enumerate(annotations):
                class_name = annotation["class"]
                segmentation = annotation["segmentation"]
                array_segmentation = np.array([segmentation], dtype=np.int32).reshape(-1, 2)
                
                img_with_mask = cv2.fillPoly(img.copy(), [array_segmentation], (0, 0, 0))
                segment_img = cv2.bitwise_xor(img, img_with_mask)
                bbox = self.box_from_polygon(array_segmentation)
                
                new_segment = array_segmentation.copy()
                new_segment[:, 0] = array_segmentation[:, 0] - bbox[0]
                new_segment[:, 1] = array_segmentation[:, 1] - bbox[1]
                
                # bbox = self.box_from_polygon(array_segmentation)
                img_crop = segment_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                # img_crop_int = (img_crop!=0).astype(int)
                # cv2.bitwise_xor()
                self.write_segment_path(img_crop, f"{file_name.split('.')[0]}_{idx}.jpg")
                with open(f"{self.all_parts_folder}/labels/{file_name.split('.')[0]}_{idx}.json", "w") as f:
                    new_segment_list = new_segment.reshape(-1,  ).tolist()
                    json.dump({"class": class_name, "segmentation": new_segment_list}, f)
                    # f.write(f"{class_name} {' '.join(map(str, new_segment_list))}\n")
                    
                print(f"{file_name.split('.')[0]}_{idx}.jpg")
                # cv2.imshow('Result', img_crop)
                # if cv2.waitKey(0) & 0xFF == ord('q'):
                #     cv2.destroyAllWindows()
                #     exit()
                # if cv2.waitKey(1) & 0xFF == ord('n'):
                #     cv2.destroyAllWindows()
    
process = Process()
process.load_json()
process.save_part_images()