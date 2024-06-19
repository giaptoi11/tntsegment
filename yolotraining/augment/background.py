import numpy as np
import glob
import cv2
import json
import tifffile
class Background():
    def __init__(self) -> None:
        self.image_folder = "/home/toi/myproject/yolotraining/data/raw_data/images"
        
        # self.image_rgb_folder = "/home/toi/myproject/yolotraining/data/data/images"
        # self.label_rgb_folder = "/home/toi/myproject/yolotraining/data/data/labels"
        
        # self.training_folder = "/home/toi/myproject/yolotraining/data/training_data"
        
        self.image_paths = glob.glob("/home/toi/myproject/yolotraining/data/raw_data/images/*")
        self.label_json_path = "/home/toi/myproject/yolotraining/data/raw_data/train_annotation.json"
        
        self.back_ground_folder = "/home/toi/myproject/yolotraining/data/augment_data/background"
    def load_json(self):        
        with open(self.label_json_path, "r") as f:
            self.data = json.loads(f.read())["images"]
            
    def read_image(self, image_name: str):
        img = tifffile.imread(f"{self.image_folder}/{image_name}")
        img = img[:, :, 1:4]
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return img
        
    def remove_segment(self, img, segmentation):
        img = cv2.fillPoly(img, [segmentation.reshape((-1, 2)).astype(np.int32)], (0, 0, 0))
        return img
    
    def fill_background(self, img, segmentation, average_color):
        img = cv2.fillPoly(img, [segmentation.reshape((-1, 2)).astype(np.int32)], (average_color[0], average_color[1], average_color[2]))
        return img
    
    def save_background(self, img, img_name: str):
        cv2.imwrite(f"{self.back_ground_folder}/{img_name.split('.')[0]}.jpg", img)
        
    def process(self):
        file_name: str
        
        for dt in self.data:
            file_name = dt["file_name"]
            annotations = dt["annotations"]
            
            # f = open(f"{self.label_rgb_folder}/{file_name.split('.')[0]}.txt", "w")
            img = self.read_image(file_name)
            height, width, channels = img.shape
            
            for idx, annotation in enumerate(annotations):
                class_name = annotation["class"]
                segmentation = annotation["segmentation"]
                
                array_segmentation = np.array([segmentation]).reshape(-1, 2)
                img = self.remove_segment(img, np.array(array_segmentation))
                
                
                # yolo_segmentation = array_segmentation.reshape(-1,).tolist()
            
                # f.write(f"0 {' '.join(map(str, yolo_segmentation))}\n")
                # print(file_name, idx,"/", len(annotations))
                
            # avg_value = [int(np.mean(img[:, :, i])) for i in range(3)]
            channel1 = np.mean(img[:, :, 0][img[:, :, 0]!=0])
            channel2 = np.mean(img[:, :, 1][img[:, :, 1]!=0])
            channel3 = np.mean(img[:, :, 2][img[:, :, 2]!=0])
            img = np.full_like(img, (channel1, channel2, channel3))
            # for idx, annotation in enumerate(annotations):
            #     class_name = annotation["class"]
            #     segmentation = annotation["segmentation"]
                
            #     array_segmentation = np.array([segmentation]).reshape(-1, 2)
            #     # img = self.remove_segment(img, np.array(array_segmentation))
            #     img = self.fill_background(img, np.array(array_segmentation), [channel1, channel2, channel3])
                
            #     # yolo_segmentation = array_segmentation.reshape(-1,).tolist()
            
            #     # f.write(f"0 {' '.join(map(str, yolo_segmentation))}\n")
            #     print(file_name, idx,"/", len(annotations))
            self.save_background(img, file_name)
            
            # f.close()
            print("+"*50)
            
    
    
    def run(self):
        self.load_json()
        self.process()
        
        
        
# data = Background()
# data.run()