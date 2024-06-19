import cv2
import tifffile
import numpy as np
import glob
import json
import os
import shutil
import random
class Data():
    def __init__(self) -> None:
        self.image_folder = "/home/toi/myproject/yolotraining/data/raw_data/images"
        
        self.image_rgb_folder = "/home/toi/myproject/yolotraining/data/data/images"
        self.label_rgb_folder = "/home/toi/myproject/yolotraining/data/data/labels"
        
        self.training_folder = "/home/toi/myproject/yolotraining/data/training_data"
        
        self.image_paths = glob.glob("/home/toi/myproject/yolotraining/data/raw_data/images/*")
        self.label_json_path = "/home/toi/myproject/yolotraining/data/raw_data/train_annotation.json"
        
    
    def run(self):
        self.load_json()
        self.convert_json_to_yolo()
        pass
    
    def read_images(self):
        for image_path in self.image_paths:
            img = tifffile.imread(image_path)
            img = img[:, :, 1:4]
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imshow("image", img)

            if cv2.waitKey(0) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
            break

    def read_image(self, image_name: str):
        img = tifffile.imread(f"{self.image_folder}/{image_name}")
        img = img[:, :, 1:4]
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        cv2.imwrite(f"{self.image_rgb_folder}/{image_name.split('.')[0]}.jpg", img)
        return img

        
    
    def load_json(self):        
        with open(self.label_json_path, "r") as f:
            self.data = json.loads(f.read())["images"]
            
        
    def convert_json_to_yolo(self):
        file_name: str
        
        for dt in self.data:
            file_name = dt["file_name"]
            annotations = dt["annotations"]
            
            f = open(f"{self.label_rgb_folder}/{file_name.split('.')[0]}.txt", "w")
            img = self.read_image(file_name)
            
            for idx, annotation in enumerate(annotations):
                class_name = annotation["class"]
                segmentation = annotation["segmentation"]
                
                height, width, channels = img.shape
                
                array_segmentation = np.array([segmentation]).reshape(-1, 2)
                array_segmentation[:, 0] = array_segmentation.copy()[:, 0]/width
                array_segmentation[:, 1] = array_segmentation.copy()[:, 1]/height
                yolo_segmentation = array_segmentation.reshape(-1,).tolist()
                
                f.write(f"0 {' '.join(map(str, yolo_segmentation))}\n")
                print(file_name, idx,"/", len(annotations))
            
            f.close()
            print("+"*50)    
            #     break
            # break
    def split_train_test(self):
        paths = glob.glob(f"{self.image_rgb_folder}/*")
        total = random.shuffle(paths)
        train_paths = paths[:int(len(paths)*0.8)]
        test_paths = paths[int(len(paths)*0.8):]
        
        os.makedirs(f"{self.training_folder}/train/images", exist_ok=True)
        os.makedirs(f"{self.training_folder}/train/labels", exist_ok=True)
        os.makedirs(f"{self.training_folder}/test/images", exist_ok=True)
        os.makedirs(f"{self.training_folder}/test/labels", exist_ok=True)
        for train_path in train_paths:
            train_label_path = train_path.replace("/images", "/labels").replace(".jpg", ".txt")
            
            shutil.copy(train_path,       f"{self.training_folder}/train/images/{train_path.split('/')[-1]}")
            shutil.copy(train_label_path, f"{self.training_folder}/train/labels/{train_label_path.split('/')[-1]}")
            
        for test_path in test_paths:
            
            test_label_path = test_path.replace("/images", "/labels").replace(".jpg", ".txt")
            
            shutil.copy(test_path,       f"{self.training_folder}/test/images/{test_path.split('/')[-1]}")
            shutil.copy(test_label_path, f"{self.training_folder}/test/labels/{test_label_path.split('/')[-1]}")
        
# process_data = Data()
# process_data.run()
# process_data.split_train_test()

