import xml.etree.ElementTree as ET
from PIL import Image
from PIL import ImageDraw
import os
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm


def vis_xml(inputpath, outputpath, showlabel=True, choose_sub = None):
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
    # Parse XML to find out unique labels
    tree = ET.parse(inputpath)
    root = tree.getroot()
    labels = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13']

    # Map labels to colors
    colormap = dict(zip(labels, pastel_colormap))
    colormap["background"] = "#FFFFFF"  # "#fcf5ee"white background

    # Function to darken a color
    def darken_color(hex_color, factor=0.9):
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
        darker_rgb = tuple(int(c * factor) for c in rgb)
        return '#{:02x}{:02x}{:02x}'.format(*darker_rgb)

    W = int(root.findall('size')[0].findall('width')[0].text)
    H = int(root.findall('size')[0].findall('height')[0].text)

    background = Image.new('RGBA', (W, H), colormap["background"])
    poly = Image.new('RGBA', (W, H))
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
    
    background.paste(poly, mask=poly)
    background.save(outputpath)


if __name__ == "__main__":
    dataroot = r""
    label_png_dir = os.path.join(dataroot, "select_xml_png")
    os.makedirs(label_png_dir, exist_ok=True)

    # # vis origin xml
    xml_dir = os.path.join(dataroot, "select_xml")
    xmls = os.listdir(xml_dir)
    for xml in xmls:
        xmlpath = os.path.join(xml_dir, xml)
        scene_name = os.path.basename(xml).split(".")[0]
        pngpath = os.path.join(label_png_dir, f"{scene_name}.png")
        vis_xml(xmlpath, pngpath, showlabel=False)