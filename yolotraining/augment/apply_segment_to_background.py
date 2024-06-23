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
import random
from worker import worker
from concurrent.futures import ProcessPoolExecutor
import datetime
class Apply():
    def __init__(self) -> None:
        self.image_folder = "/home/toi/myproject/yolotraining/data/raw_data/images"
        
        # self.image_rgb_folder = "/home/toi/myproject/yolotraining/data/data/images"
        # self.label_rgb_folder = "/home/toi/myproject/yolotraining/data/data/labels"
        
        # self.training_folder = "/home/toi/myproject/yolotraining/data/training_data"
        
        self.image_paths = glob.glob("/home/toi/myproject/yolotraining/data/raw_data/images/*")
        self.label_json_path = "/home/toi/myproject/yolotraining/data/raw_data/train_annotation.json"
        
        self.back_ground_folder = "/home/toi/myproject/yolotraining/data/augment_data/background"
        self.all_parts_folder  = "/home/toi/myproject/yolotraining/data/augment_data/segment_parts"
        
        self.augment_folder  = "/home/toi/myproject/yolotraining/data/augment_data/augment_images"
        os.makedirs(f"{self.augment_folder}", exist_ok=True)
        os.makedirs(f"{self.augment_folder}/images", exist_ok=True)
        os.makedirs(f"{self.augment_folder}/labels", exist_ok=True)
    def load_json(self):        
        with open(self.label_json_path, "r") as f:
            self.data = json.loads(f.read())["images"]
            
    def read_image(self, image_name: str):
        img = tifffile.imread(f"{self.image_folder}/{image_name}")
        img = img[:, :, 1:4]
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return img
    
    def save_background(self, img, img_name: str):
        cv2.imwrite(f"{self.back_ground_folder}/{img_name.split('.')[0]}.jpg", img)
    
    def write_segment_path(self, img, img_name:str):
        cv2.imwrite(f"{self.all_parts_folder}/{img_name.split('.')[0]}.jpg", img)
        
    
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
    
    def get_mini_parts(self):
        paths = glob.glob(self.all_parts_folder + "/images/*")
        return paths
    def get_background(self):
        paths = glob.glob(self.back_ground_folder + "/*")
        return paths
    
    # Hàm kiểm tra vị trí có hợp lệ không (không bị chồng chéo)
    def is_valid_position(self, positions, x, y, w, h):
        for (px, py, pw, ph) in positions:
            if (x < px + pw and x + w > px) and (y < py + ph and y + h > py):
                return False
        return True
    
    def apply_segment_to_background(self, i):
        file_name: str
        
        mini_paths = self.get_mini_parts()
        background_paths = self.get_background()
        
        for bg_index, background_path in enumerate(background_paths):
            # print(f"{datetime.datetime.now()}", "="*50)
            print(f"From thread: {i%8} - {bg_index}/{len(background_paths)}")
            label_f= open(f"{self.augment_folder}/labels/{background_path.split('/')[-1].split('.')[0]}_{i}.txt", "w")
            
            background = cv2.imread(background_path)
            background = background[:640, :640, :]
            bg_height, bg_width, _ = background.shape
            
            
            #get random from mini_paths
            new_mini_paths  = random.sample(mini_paths, 600)
            positions = []
            for index, mini_path in enumerate(new_mini_paths):
                # print(mini_path,index, "/", len(new_mini_paths))
                segment_path = mini_path.replace("images", "labels").replace("jpg", "json")
                with open(segment_path, "r") as f:
                    segment = json.loads(f.read())["segmentation"]
                    
                mini_part = cv2.imread(mini_path)
                
                # mini_part = cv2.cvtColor(mini_part, cv2.COLOR_BGR2BGRA)
                mini_part_height, mini_part_width = mini_part.shape[:2]
                
                placed = False
                
                for y in range(0, bg_height - mini_part_height, 10): 
                    for x in range(0, bg_width - mini_part_width, 10):
                        
                        if self.is_valid_position(positions, x, y, mini_part_width, mini_part_height):
                            positions.append((x, y, mini_part_width, mini_part_height))
                            placed = True
                        else:
                            placed = False
                            
                        if placed:
                            if mini_part.shape[2] == 4:  
                                alpha_s = mini_part[:, :, 3] / 255.0
                                alpha_b = 1.0 - alpha_s
                                for c in range(0, 3):
                                    background[y:y+mini_part_height, x:x+mini_part_width, c] = (
                                    alpha_s * mini_part[:, :, c] +
                                    alpha_b * background[y:y+mini_part_height, x:x+mini_part_width, c]
                            )
                            else:
                                seg_numpy = np.array([segment]).reshape(-1, 2)
                                seg_numpy[:, 0] = seg_numpy.copy()[:, 0] + x
                                seg_numpy[:, 1] = seg_numpy.copy()[:, 1] + y
                                # cv2.polylines(background, [seg_numpy.astype(np.int32)],True, (255, 255, 255), 1)
                                new_seg = seg_numpy.reshape(-1).tolist()
                                label_f.write(f"0 {' '.join(map(str, new_seg))}\n")
                                
                                bg = background.copy()[y:y+mini_part_height, x:x+mini_part_width]
                                # mask1 = (mini_part.copy()==0).astype(int)
                                # bg2 = bg*mask1
                                bg2 = cv2.fillPoly(bg, [seg_numpy.astype(np.int32)], (0, 0, 0))
                                background[y:y+mini_part_height, x:x+mini_part_width] = bg2 + mini_part
                                
                            break
                    
                    if placed:
                        placed = False
                        break 
                    
            cv2.imwrite(f"{self.augment_folder}/images/{background_path.split('/')[-1].split('.')[0]}_{i}.jpg", background)
            label_f.close()
        return f"{datetime.datetime.now()} -  finished {i}"
                # cv2.imshow("background", background)
                # cv2.imshow("minipart", mini_part)
                # if cv2.waitKey(0):
                #     if 0xFF == ord("b"):
                #         cv2.destroyAllWindows()
                #         exit(0)
                        
                #     if 0xFF == ord("q"):
                #         cv2.destroyAllWindows()
    
    def main(self):
        for index_pool in np.arange(start = 0, stop = 1000, step = 8):
            hour = datetime.datetime.now().hour
            if int(hour) == 7:
                break
            else:
                print(f"{datetime.datetime.now()}, hour = {hour}","="*50)
            with ProcessPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(self.apply_segment_to_background, index_pool+i) for i in range(8)]


                results = [future.result() for future in futures]
                for result in results:
                    print(result)
            # runs = [threading.Thread(target=run, args=(i+index,)) for index in range(8)]
            
            # for r in runs:
            #     r.start()
            
            # for r in runs:
            #     r.join()

            
                    
                
                
            
if __name__ == "__main__":
    
    process = Apply()
    process.load_json()
    # process.apply_segment_to_background()
    process.main()
# process.save_part_images()