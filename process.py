import numpy as np
import cv2
import math
import xml.etree.ElementTree as ET
import os
import time
from eprogress import LineProgress
import argparse

line_progress = LineProgress(title='Progress:')

target_w = 416
target_h = 416

img_path = "images/"
out_img_path = "out_images/"
xml_path = "labelxml/"
out_txt_path = "txt/"

classes = ["ring"]

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def cv2_letterbox_image(image, expected_size, box):
    
    ih, iw = image.shape[0:2]
    ew, eh = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    for i in range(4):
        box[i] = box[i] * scale
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    top = (eh - nh) // 2
    box[2] = box[2] + top
    box[3] = box[3] + top
    bottom = eh - nh - top
    left = (ew - nw) // 2
    box[0] = box[0] + left
    box[1] = box[1] + left
    right = ew - nw - left
    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return new_img, box

def convert_annotation(image_id):
    in_file = open(os.path.join(xml_path,image_id + ".xml")) 
    img = cv2.imread(os.path.join(img_path,image_id + ".png"))
    out_img  = img.copy()
    out_file = open(os.path.join(out_txt_path,image_id + ".txt"), 'w')  
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if int(cls) != 9 or int(difficult)==1:
            continue
        cls_id = 0
        xmlbox = obj.find('bndbox')
        box = np.array([float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)])
        dst_img, box_new = cv2_letterbox_image(img, (target_w, target_h), box)
        b = tuple(box_new)
        b_scale = convert((target_w,target_h), b)
        out_img = dst_img
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in b_scale]) + '\n')
    in_file.close()
    out_file.close()
    return out_img

if __name__ == "__main__":
    files = os.listdir(img_path)  
    files_num = len(files)
    parser = argparse.ArgumentParser()
    parser.add_argument('img_input_path')
    parser.add_argument('xml_input_path')
    parser.add_argument('img_output_path')
    parser.add_argument('txt_output_path')
    parser.add_argument('target_width')
    parser.add_argument('target_height')
    opt = parser.parse_args()
    img_path = opt.img_input_path
    xml_path = opt.xml_input_path
    out_img_path = opt.img_output_path
    out_txt_path = opt.txt_output_path
    target_w = int(opt.target_width)
    target_h = int(opt.target_height)
    
    if not os.path.exists(out_img_path):
        os.makedirs(out_img_path)
    if not os.path.exists(out_txt_path):
        os.makedirs(out_txt_path)

    for i in range(1,files_num):
        out_img = convert_annotation("Screenshot_"+str(i))
        cv2.imwrite(os.path.join('./',out_img_path,"Screenshot_"+ str(i) + ".png"),out_img)
        line_progress.update(int(i*100/files_num))
    