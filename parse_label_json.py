import json
import os
import numpy as np
import xml.etree.ElementTree as ET 
from shapely.geometry import Polygon, box  
import shutil

def float2int(points):
    points_int = []
    for point in points:
        point_int = []
        for item in point:
            point_int.append(int(round(item)))
        points_int.append(point_int)
    return points_int

def get_info_of_rect(points):
    points_int = float2int(points)
    p1 = points_int[0]
    p2 = points_int[1]
    delta_x = abs(p1[0] - p2[0])
    delta_y = abs(p1[1] - p2[1])
    center = list((np.asarray(p1) + np.asarray(p2)) / 2)
    return delta_x, delta_y, center

def align_rect_background(background, rect):
    origin = np.asarray(background[0])
    rect_aligned = [list(np.asarray(rect[0]) - origin), list(np.asarray(rect[1]) - origin)]
    return rect_aligned

def get_resolution_background(shapes):
    for shape in shapes:
        label_name = shape["label"]
        if label_name[0] == "p":
            act_dist = float(label_name.split("_")[1])
            pix_dist,_,_ = get_info_of_rect(shape["points"])
            resolution = act_dist/pix_dist
        if label_name == "B":
            background = shape["points"]
            background_W, background_H,_ = get_info_of_rect(background)
    try:
        resolution == 5
    except:
        resolution = 5
        print("*********WARNING: MISSING RESOLUTION, USING DEFUALT 5m***********")
    return resolution, background, background_W, background_H

def json4aug(inputpath, outputpath):
    with open(inputpath, encoding='utf-8') as f:
        data = json.load(f)
    shapes = data["shapes"]
    resolution, background, background_W, background_H = get_resolution_background(shapes)
    outjson = {"resolution": resolution, "W": background_W, "H":background_H}
    for shape in shapes:
        label_name = shape["label"]     
        if label_name[0] == "p" or label_name == "B": continue
        outjson[label_name] = []
    global_idx = 0
    for shape in shapes:
        label_name = shape["label"]     
        if label_name[0] == "p" or label_name == "B": continue
        rect_aligned = align_rect_background(background, shape["points"])
        rect_int = float2int(rect_aligned)
        rect_x, rect_y, rect_c = get_info_of_rect(rect_aligned)
        outjson[label_name].append({"left_top":rect_int[0], "right_bottom":rect_int[1],"xwidth":rect_x, "yheight":rect_y, "c":rect_c, "global_idx":global_idx})
        global_idx += 1
    with open(outputpath, "w") as fw:
        json.dump(outjson, fw, sort_keys=True)


def generate_xml_template(scenename, W, H, element_list): 
    root = ET.Element("annotation")  
    category = ET.SubElement(root, "category")  
    category.text = "site"  
    filename = ET.SubElement(root, "filename")  
    filename.text = f"{category.text}_{scenename}"  
    size = ET.SubElement(root, "size")  
    width = ET.SubElement(size, "width")  
    width.text = str(W) 
    height = ET.SubElement(size, "height")  
    height.text = str(H) 
    layout = ET.SubElement(root, "layout")  
    for item in element_list:
        element1 = ET.SubElement(layout, "element")  
        element1.set("label", item["label"])  
        element1.set("polygon_x", item["polygon_x"])  
        element1.set("polygon_y", item["polygon_y"])  
        element1.set("global_idx", item["global_idx"])  
    text = ET.SubElement(root, "text")  
    keyword = ET.SubElement(text, "keyword")  
    keyword.text = "shop"  
    tree = ET.ElementTree(root)  

    return tree

def split_points_xy(points, convertstr=True):
    polygon_x = []
    polygon_y = []
    left_top = points[0]
    right_bottom = points[1]  
    rect_polygon = Polygon([(left_top[0], left_top[1]),   
                         (right_bottom[0], left_top[1]),   
                         (right_bottom[0], right_bottom[1]),   
                         (left_top[0], right_bottom[1])])
    for x, y in rect_polygon.exterior.coords:  
        polygon_x.append(int(x))  
        polygon_y.append(int(y))  
    if convertstr:
        polygon_x = "".join([str(num)+" " for num in polygon_x])[:-1]
        polygon_y = "".join([str(num)+" " for num in polygon_y])[:-1]
    return polygon_x, polygon_y
def split_center_xy(center, xwidth, yheight):
    points = [[center[0]-xwidth/2, center[1]-yheight/2], [center[0]+xwidth/2, center[1]+yheight/2]]
    polygon_x, polygon_y = split_points_xy(points)
    return polygon_x, polygon_y

    
def json2xml(inputpath, outputpath):
    with open(inputpath, encoding='utf-8') as f:
        data = json.load(f)
    shapes = data["shapes"]
    resolution, background, background_W, background_H = get_resolution_background(shapes)

    scenename = os.path.basename(outputpath).split(".")[0]
    element_list = []
    global_idx = 0
    for shape in shapes:
        cur_element={}
        label_name = shape["label"]     
        if label_name[0] == "p" or label_name == "B": continue
        cur_element["label"] = label_name
        rect_aligned = align_rect_background(background, shape["points"])
        rect_int = float2int(rect_aligned)
        polygon_x, polygon_y = split_points_xy(rect_int)
        cur_element["polygon_x"] = polygon_x
        cur_element["polygon_y"] = polygon_y
        cur_element["global_idx"] = str(global_idx)
        element_list.append(cur_element)
        global_idx += 1
    tree = generate_xml_template(scenename, background_W, background_H, element_list)
    tree.write(outputpath, encoding="utf-8", xml_declaration=True)



if __name__ == "__main__":
    dataroot = r""
    annotation_dir = os.path.join(dataroot, "annotation")
    ann_aug_dir = os.path.join(dataroot, "ann_aug")
    originxml_dir = os.path.join(dataroot, "origin_xml")
    optxml_dir = os.path.join(dataroot, "opt_xml")
    os.makedirs(ann_aug_dir, exist_ok=True)
    os.makedirs(originxml_dir, exist_ok=True)
    os.makedirs(optxml_dir, exist_ok=True)

    scene_files = os.listdir(annotation_dir)

    for scene_file in scene_files:
        scene_name = os.path.basename(scene_file).split(".")[0]
        annotationpath = os.path.join(annotation_dir,scene_file)
        ann_aug_path = os.path.join(ann_aug_dir, f"{scene_name}.json")
        originxml_path = os.path.join(originxml_dir, f"{scene_name}.xml")
        optxml_path = os.path.join(optxml_dir, f"{scene_name}.xml")

        json4aug(annotationpath, ann_aug_path)
        print(f"saved ann_aug {ann_aug_path}")
        json2xml(annotationpath, originxml_path)
        shutil.copy(originxml_path, optxml_path)
        print(f"saved xml {originxml_path}")