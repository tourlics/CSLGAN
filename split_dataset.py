import os
import random
import shutil

def write_txt_files(root_image_dir, split_txt_dir):
    image_names = os.listdir(root_image_dir)

    total_num = len(image_names)
    train_num = int(0.85*total_num)
    val_num = int(0.05* total_num)
    test_ratio = total_num - train_num - val_num

    train_names = random.sample(image_names, train_num)

    remaining_elements = [x for x in image_names if x not in train_names]
    val_names = random.sample(remaining_elements, val_num)

    test_names = [x for x in image_names if (x not in train_names) and (x not in val_names)]

    with open(os.path.join(split_txt_dir, "train.txt"), 'w', encoding='utf-8') as file:  
        for item in train_names:  
            file.write(item + '\n')  

    with open(os.path.join(split_txt_dir, "val.txt"), 'w', encoding='utf-8') as file:  
        for item in val_names: 
            print(item) 
            file.write(item + '\n')  

    with open(os.path.join(split_txt_dir, "test.txt"), 'w', encoding='utf-8') as file:  
        for item in test_names:  
            file.write(item + '\n')  

def copy_images_txt(split_txt_dir, root_image_dir, out_img_dir, mode):    
    os.makedirs(os.path.join(out_img_dir, mode), exist_ok=True)
    with open(os.path.join(split_txt_dir, f"{mode}.txt"), 'r', encoding='utf-8') as file:  
        for line in file:  
            line = line.replace("\n", "")
            shutil.copy(os.path.join(root_image_dir, line), os.path.join(out_img_dir, mode,line))
            
if __name__ == "__main__":

    image_types = "pngall_xml"
    split_txt_dir = f""
    os.makedirs(split_txt_dir, exist_ok=True)
    root_image_dir = f""
    write_txt_files(root_image_dir, split_txt_dir)

    root_image_dir = f""
    out_img_dir = f""
    os.makedirs(out_img_dir, exist_ok=True)
    copy_images_txt(split_txt_dir, root_image_dir, out_img_dir, mode="train")
    copy_images_txt(split_txt_dir, root_image_dir, out_img_dir, mode="val")
    copy_images_txt(split_txt_dir, root_image_dir, out_img_dir, mode="test")


