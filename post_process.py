import tifffile
import json
import cv2
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from shapely.geometry import Polygon, LineString, GeometryCollection, MultiPolygon, MultiLineString
from shapely.geometry import Polygon, LineString, GeometryCollection

def imshow(image):
    plt.imshow(image)
    plt.show()

# Hàm vẽ polygon
def get_only_polygon(polygon):
    if isinstance(polygon, MultiPolygon):
        # If it's a MultiPolygon, get the largest polygon by area
        largest_poly = max(polygon.geoms, key=lambda poly: poly.area)
        return largest_poly 
    
    elif isinstance(polygon, LineString):
        return None
    
    elif isinstance(polygon, GeometryCollection):
        result = []
        for geom in polygon.geoms:
            result.append(get_only_polygon(geom))
        
        
        for rs in result.copy():
            if rs == None:
                result.remove(None)
                
        largest_poly = max(result, key=lambda poly: poly.area)
        
    else:
        return polygon
    
def polygon_to_list(polygon):
    return np.array(list(polygon.exterior.coords), np.int32).reshape(-1, )

import math
def euclidean_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)
def separate_polygon(polygon1, polygon2):
    polygon1 = get_only_polygon(polygon1)
    polygon2 = get_only_polygon(polygon2)
    
    if polygon1 == None or polygon2 == None:
        return None, None
    
    overlap = polygon1.intersection(polygon2)
    if isinstance(overlap, MultiLineString):
        return None, None
    centroid1 = polygon1.centroid
    centroid2 = polygon2.centroid
    overlap_centroid = overlap.centroid
    
    try:
        # Tính khoảng cách giữa centroid của overlap và centroid của hai polygon
        distance_to_polygon1 = euclidean_distance(overlap_centroid, centroid1)
        distance_to_polygon2 = euclidean_distance(overlap_centroid, centroid2)
    except:
        return None, None
    
    # union = polygon1.union(polygon2)
    if distance_to_polygon1 <= distance_to_polygon2:
        polygon2 = polygon2.difference(polygon1)
        # print("Overlap belongs to Polygon 1")
    else:
        polygon1 = polygon1.difference(polygon2)
        # print("Overlap belongs to Polygon 2")
    return polygon1, polygon2
    
def draw_polygon(image, polygon, color, fill = False):
    pts = np.array(list(polygon.exterior.coords), np.int32)
    pts = pts.reshape((-1, 1, 2))
    if not fill:
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
    else:
        cv2.fillPoly(image, [pts], color=color, lineType=cv2.LINE_AA)
    return image

# Hàm vẽ LineString
def draw_linestring(image, linestring, color):
    pts = np.array(list(linestring.coords), np.int32)
    for i in range(len(pts) - 1):
        cv2.line(image, tuple(pts[i]), tuple(pts[i+1]), color=color, thickness=2)
    return image
# prompt: hãy vẽ polygon trên trên vào ảnh img
import random
def random_color():
  """
  Generates a random RGB color.
  """
  red = random.randint(0, 255)
  green = random.randint(0, 255)
  blue = random.randint(0, 255)
  return (red, green, blue)
    
all_data = json.loads(open('/home/toi/myproject/submitsion_sh.json').read())
new_data = {"images": []}
for data in all_data["images"]:
    filename = data["file_name"]
    seg = data["annotations"][0]["segmentation"]
    image = tifffile.imread(f"/home/toi/myproject/test_images/images/{filename}")
    img = cv2.normalize(image[:,: , 1:4], None, 0, 255, cv2.NORM_MINMAX, ).astype(np.uint8)
    image = img.copy()
    print(filename)
    


    polygons = [] 
    for sg in data["annotations"]:
        segmentation = sg["segmentation"]
        segmentation_array = np.array(segmentation, dtype=np.int32)
        segmentation_array = segmentation_array.reshape((-1, 2))
        # if len(segmentation_array.tolist()) < 5:
        #     print(segmentation_array.tolist())
        polygons.append(Polygon(segmentation_array.tolist()))

    polygon_info = []
    tree = STRtree(polygons)
    overlaps = []

    for index, polygon in enumerate(polygons):
        possible_overlaps_indices = tree.query(polygon)  # Tìm các chỉ số polygon có thể overlap
        for idx in possible_overlaps_indices:
            other = polygons[idx]  # Lấy polygon từ chỉ số
            if polygon != other:  # Loại bỏ chính polygon đó
                try:
                    overlap = polygon.intersection(other)
                    if not overlap.is_empty:
                        overlaps.append(overlap)
                        polygon_info.append((index, idx, overlap))
                except:
                    continue
                        
                        
    #Write folder before filter
    for poly in polygons:
        image = draw_polygon(image, poly, (255, 255, 255))  # Màu xanh dương
        
    for overlap in overlaps:
        # Vẽ các thành phần của GeometryCollection
        if isinstance(overlap, GeometryCollection):
            for geom in overlap.geoms:
                if isinstance(geom, Polygon):
                    draw_polygon(image, geom, (0, 0, 255), True)  
                elif isinstance(geom, LineString):
                    draw_linestring(image, geom, (255, 0, 0)) 
        else:
            if isinstance(overlap, Polygon):
                draw_polygon(image, overlap, (0, 0, 255), True)  
            elif isinstance(overlap, LineString):
                draw_linestring(image, overlap, (255, 0, 0))  

        # imshow(image)
    cv2.imwrite(f"/home/toi/myproject/before_filter/{filename}.jpg", image)
    
    ################################################################################################33
    
    image = img.copy()
    for polygon_index1, polygon_index2, overlap in polygon_info.copy():
        try:
            polygon1, polygon2 = separate_polygon(polygons[polygon_index1], polygons[polygon_index2])
            if polygon1 == None or polygon2 == None:
                continue
            polygons[polygon_index1] = polygon1
            polygons[polygon_index2] = polygon2
        except Exception as e:
            print(type(polygons[polygon_index1]), type(polygons[polygon_index2]))
            print(polygon_index1, polygon_index2)
            print(e)
            print("="*50)
        
        
    index_to_remove = []
    for idx, polygon in enumerate(polygons.copy()):
        r = get_only_polygon(polygon)
        if r == None:
            index_to_remove.append(idx)
        else:
            polygons[idx] = r

    for i in sorted(index_to_remove, reverse=True):
        # print(i, len(polygons))
        del polygons[i]



    # for idx, polygon in enumerate(polygons.copy()):
    #     if isinstance(polygon, MultiPolygon):
    #         # If it's a MultiPolygon, get the largest polygon by area
    #         largest_poly = max(polygon.geoms, key=lambda poly: poly.area)
    #         polygons[idx] = largest_poly
            
    #     elif isinstance(polygon, GeometryCollection):
    #         for geom in polygon.geoms:
    #             print(type(geom))
    #             # if isinstance(geom, LineString):
    #             #     draw_linestring(image, overlap, (255, 0, 0))  
    #         print("="*50)


            
    tree = STRtree(polygons)
    overlaps = []

    for index, polygon in enumerate(polygons):
        possible_overlaps_indices = tree.query(polygon)  # Tìm các chỉ số polygon có thể overlap
        for idx in possible_overlaps_indices:
            other = polygons[idx]  # Lấy polygon từ chỉ số
            if polygon != other:  # Loại bỏ chính polygon đó
                try:
                    overlap = polygon.intersection(other)
                    if not overlap.is_empty:
                        overlaps.append(overlap)
                except:
                    print(idx, type(other))
                    continue
                    
    data_seg = []
    for poly in polygons:
        if isinstance(poly, GeometryCollection):
            for geom in poly.geoms:
                if isinstance(geom, Polygon):
                    draw_polygon(image, geom, (255, 255, 255))  
                    data_seg.append({
                        "class": "field", 
                        "segmentation": polygon_to_list(geom).tolist()
                    })
                    
                elif isinstance(geom, LineString):
                    draw_linestring(image, geom, (255, 255, 255))
        else:
            data_seg.append({
                        "class": "field", 
                        "segmentation": polygon_to_list(poly).tolist()
                    })
            draw_polygon(image, poly, (255, 255, 255)) 
        
            
        

        
    for overlap in overlaps:
        # Vẽ các thành phần của GeometryCollection
        if isinstance(overlap, GeometryCollection):
            for geom in overlap.geoms:
                if isinstance(geom, Polygon):
                    draw_polygon(image, geom, (0, 0, 255), True)  
                elif isinstance(geom, LineString):
                    draw_linestring(image, geom, (255, 0, 0)) 
        else:
            if isinstance(overlap, Polygon):
                draw_polygon(image, overlap, (0, 0, 255), True)  
            elif isinstance(overlap, LineString):
                draw_linestring(image, overlap, (255, 0, 0))  
                
    new_data["images"].append({
        "file_name": filename,
        "annotations": data_seg
    })
    cv2.imwrite(f"/home/toi/myproject/after_filter/{filename}.jpg", image)

with open("/home/toi/myproject/submissionsh2.json", "w") as f:
    json.dump(new_data, f)
    
    
