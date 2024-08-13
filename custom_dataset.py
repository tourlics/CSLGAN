import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from tqdm import tqdm
def vis_xml(inputpath, outputpath, max_width, max_height, showlabel=True, choose_sub=None):
    pastel_colormap = [
        "#C5E3ED", 
        "#CCEBC5", 
        "#EEDDEE", 
        "#FFD1DC", 
        "#FFC8A2", 
        "#FFFFCD",
        "#EAEAEA", 
        "#EED8AE", 
        "#F0FFFF", 
        "#767171", 
        "#ECA8A9", 
        "#F0F0FF", 
        "#FAEBD7"
    ]
    
    tree = ET.parse(inputpath)
    root = tree.getroot()
    labels = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13']

    colormap = dict(zip(labels, pastel_colormap))
    colormap["background"] = "#FFFFFF"

    def darken_color(hex_color, factor=0.9):
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
        darker_rgb = tuple(int(c * factor) for c in rgb)
        return '#{:02x}{:02x}{:02x}'.format(*darker_rgb)

    W = int(root.findall('size')[0].findall('width')[0].text)
    H = int(root.findall('size')[0].findall('height')[0].text)

    max_background_color = "#C4C4C4"
    background = Image.new('RGBA', (max_width, max_height), max_background_color)
    
    poly = Image.new('RGBA', (W, H), colormap["background"])
    pdraw = ImageDraw.Draw(poly)

    for layout in root.findall('layout'):
        for element in layout.findall('element'):
            label = element.get('label')
            if choose_sub is not None:
                if not (label in choose_sub):
                    continue
            px = [int(i) for i in element.get('polygon_x').split(" ")]
            py = [int(i) for i in element.get('polygon_y').split(" ")]
            polygon_color = colormap[label]
            outline_color = darken_color(polygon_color)
            pdraw.polygon(list(zip(px, py)), fill=polygon_color, outline=outline_color)
            if showlabel:
                pdraw.text((min(px), min(py)), label + f"  #{element.get('global_idx')}", fill="black")
    
    offset_x = (max_width - W) // 2
    offset_y = (max_height - H) // 2

    background.paste(poly, (offset_x, offset_y), mask=poly)
    background.save(outputpath)


def find_max_dimensions(input_folder):
    max_width = 0
    max_height = 0
    xml_files = glob.glob(os.path.join(input_folder, "*.xml"))
    
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        W = int(root.findall('size')[0].findall('width')[0].text)
        H = int(root.findall('size')[0].findall('height')[0].text)
        if W > max_width:
            max_width = W
        if H > max_height:
            max_height = H
    
    return max_width, max_height

def process_folder(input_folder, pngall_folder, pngcon_folder, showlabel=False, choose_sub=None):
    
    max_width, max_height = find_max_dimensions(input_folder)
    xml_files = glob.glob(os.path.join(input_folder, "*.xml"))
    
    for xml_file in tqdm(xml_files):
        output_file1 = os.path.join(pngall_folder, os.path.basename(xml_file).replace(".xml", ".png"))
        vis_xml(xml_file, output_file1, max_width, max_height, showlabel, choose_sub)


xml_folder = r''
pngall_folder = r''
pngcon_folder = r''

os.makedirs(pngall_folder, exist_ok=True)
os.makedirs(pngcon_folder, exist_ok=True)
process_folder(xml_folder, pngall_folder, pngcon_folder)