#!/usr/bin/python
"""
Developer: Adam Terwilliger
Version: 1.0
Purpose: Segmentation Prediction
Details: Given n RGB frames of a video, predict
         the next m segmentation frames
         Currently focused on Cityscapes dataset
Contact: adamtwig@msu.edu
"""

import os
import numpy as np

def gen_img_set(data_path):
    
    label_path = os.path.join(os.path.realpath(data_path), "gtFine", "val")
    
    labels = []
    for root, dirs, files in os.walk(label_path):
        labels += [root +"/"+ s for s in files]

    names = sorted([im for im in labels if "labelIds" in im])

    return names

def check_add_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def setup_output_dirs(output_path, color, hybrid):
    check_add_dir(output_path)
    paths = ["pred"]
    if color:
        paths.append("color")
    if hybrid:
        paths.append("hybrid")
    for tmp_path in paths:
        check_add_dir(os.path.join(output_path, tmp_path))
        for city in ["frankfurt", "lindau", "munster"]:
            check_add_dir(os.path.join(output_path, tmp_path, city))

def get_cityscapes_filename(data_path, ref_frame, frame_num):
    
    rgb_path = os.path.join(os.path.realpath(data_path), "leftImg8bit_sequence", "val")
    label_parts = ref_frame.split("_")
    label_prefix = label_parts[-5]
    curr_city = label_prefix.split("/")[-1]
    label_id = int(label_parts[-3])
    data_str_id =  str(label_id - 19 + frame_num).zfill(6)
    img_name = curr_city+"_"+label_parts[-4]+"_"+data_str_id+"_leftImg8bit.png"
    data_file = os.path.join(rgb_path,curr_city,img_name)
    
    return data_file

def get_segfeat_filename(frame_name):

    output_name = frame_name.replace("leftImg8bit.png","segFeat.npy")
    output_name = output_name.replace("/cityscapes/leftImg8bit_sequence", "/cityscapes/segFeatures/pspnet")

    return output_name

def get_pred_filename(output_path, ref_frame):

    label_parts = ref_frame.split("_")
    label_prefix = label_parts[-5]
    curr_city = label_prefix.split("/")[-1]
    img_name = curr_city+"_"+label_parts[-4]+"_"+label_parts[-3]+"_pred.png"
    out_pred_filename = os.path.join(output_path, "pred", curr_city, img_name)

    return out_pred_filename


def get_label_color_pred(pred):

    trainID_map = {0:7, 1:8, 2:11, 3:12, 4:13, 5:17, 6:19, 7:20, 8:21, 9:22, 10:23, 11:24, 12:25, 13:26, 14:27, 15:28, 16:31, 17:32, 18:33}

    color_dict = {0 : (0,0,0),1 : (0,0,0),2 : (0,0,0),3 : (0,0,0),4 : (0,0,0),5 : (0,74,111),6 : (81,0,81),7 : (128,64,128),8 : (232,35,244),9 : (160,170,250),10 : (140,150,230),11 : (70,70,70),12 : (156,102,102),13 : (153,153,190),14 : (180,165,180),15 : (100,100,150),16 : (90,120,150),17 : (153,153,153),18 : (153,153,153),19 : (30,170,250),20 : (0,220,220),21 : (35,142,107),22 : (152,251,152),23 : (180,130,70),24 : (60,20,220),25 : (0,0,255),26 : (142,0,0),27 : (70,0,0),28 : (100,60,0),29 : (90,0,0),30 : (110,0,0),31 : (100,80,0),32 : (230,0,0),33 : (32,11,119),-1 : (142,0,0)}

    label_pred = np.array(pred)
    color_pred = np.zeros([pred.shape[0], pred.shape[1], 3])
    unique_trainIDs = np.unique(pred)
    
    for trainID in unique_trainIDs:
        label_pred[pred == trainID] = trainID_map[trainID]
        color_pred[pred == trainID] = color_dict[trainID_map[trainID]]

    return label_pred, color_pred

