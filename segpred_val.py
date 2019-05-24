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

from __future__ import print_function

import caffe
import numpy as np
import cv2
import sys
import os
import utils.utils_segpred as utils
import utils.eval.cityscapesEvaluation as city_eval
np.set_printoptions(precision=3, suppress=True)

# parameters: 
#   short_mid = "t3" or "t10"
#   gpu_id = 0, 1, ... any gpu 
#   (nvidia-smi to see available gpus)
def main(short_mid="t10", gpu_id=0):

    # default directories
    # recommend either softlink to desired directories
    #   or renaming these variables
    data_path = "data/"
    models_path = "models/"
    results_path = "results/"

    output_path = os.path.join(results_path, "segpred_"+short_mid, "val")

    # top performing network is released: t3-67.2 t10-52.5 mIOU
    segpred_net_path = os.path.join(models_path, "segpred.prototxt")
    segpred_weights_path = os.path.join(models_path, "segpred_"+short_mid+".caffemodel")

    caffe.set_mode_gpu()
    caffe.set_device(int(gpu_id))
    
    val_net = caffe.Net(segpred_net_path, segpred_weights_path, caffe.TEST)

    # generates list of filenames for validation images
    img_set = utils.gen_img_set(data_path)
    
    # output predictions as color images or rgb/color hybrid
    color = False
    hybrid = False
    
    # creates nested directories for cityscapes
    utils.setup_output_dirs(output_path, color, hybrid)
    
    print("---------------------------------------------------------------------------------------------")
    print("Output path:", os.path.realpath(output_path))
    print("---------------------------------------------------------------------------------------------")
    print("\nValidation Phase")
    print("---------------------")
    
    num_iter = 1
    
    for ref_frame in img_set:
   
        print("\rImages Processed: {}".format(num_iter), end=" ")
        sys.stdout.flush()
    
        if short_mid == "t3":
            input_frames = [7, 10, 13, 16]
        elif short_mid == "t10":
            input_frames = [0, 3, 6, 9]

        frame19_name = utils.get_cityscapes_filename(data_path, ref_frame, 19)
        frame19_full = cv2.imread(frame19_name)
        frame19 = cv2.resize(frame19_full, (1024, 512))
   
        # normalize rgb
        frame19 = frame19 * 0.00392156862745098  
    
        frame0 = np.zeros((len(input_frames)-1, 512, 1024, 3))
        frame1 = np.zeros((len(input_frames)-1, 512, 1024, 3))
        
        seg_feat = np.zeros((len(input_frames), 19, 64, 128))
        
        for past in range(len(input_frames)):
            
            frame0_name = utils.get_cityscapes_filename(data_path, ref_frame, input_frames[past])
            frame0_full = cv2.imread(frame0_name)

            # this approach uses cached segmentation features from pspnet
            # must first run ___.py to preprocess and save the seg feats
            seg_feat0_name = utils.get_segfeat_filename(frame0_name)
            seg_feat0 = np.load(seg_feat0_name)
            seg_feat[past] = seg_feat0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        
            if past < len(input_frames)-1:            
                
                frame1_name = utils.get_cityscapes_filename(data_path, ref_frame, input_frames[past+1])
                frame1_full = cv2.imread(frame1_name)
                        
                frame0[past] = cv2.resize(frame0_full, (1024, 512))
                frame1[past] = cv2.resize(frame1_full, (1024, 512))
        
        sf_s0, sf_s1, sf_s2, sf_s3 = seg_feat.shape
        val_net.blobs["seg_feat"].data[...] = seg_feat

        tFrame0 = frame0.transpose(0, 3, 1, 2)
        tFrame1 = frame1.transpose(0, 3, 1, 2)
        tF_s0, tF_s1, tF_s2, tF_s3 = tFrame0.shape
        val_net.blobs["rgb0"].reshape(tF_s0, tF_s1, tF_s2, tF_s3)
        val_net.blobs["rgb1"].reshape(tF_s0, tF_s1, tF_s2, tF_s3)
        val_net.blobs["rgb0"].data[...] = tFrame1
        val_net.blobs["rgb1"].data[...] = tFrame0

        flow_seq = np.ones((len(input_frames)-1,1))
        flow_seq[0] = 0
        val_net.blobs["flow_seq"].reshape(flow_seq.shape[0], flow_seq.shape[1])
        val_net.blobs["flow_seq"].data[...] = flow_seq

        out = val_net.forward()

        tOut = np.array(out["seg_softmax"][0].transpose([1,2,0]))
        
        # output is upsampled before evaluation
        re_seg = cv2.resize(tOut, (2048, 1024))
        
        pred = np.argmax(re_seg, axis=2)

        label_pred, color_pred = utils.get_label_color_pred(pred)

        out_pred_filename = utils.get_pred_filename(output_path, ref_frame)
        cv2.imwrite(out_pred_filename, label_pred)

        if color:
            out_color_filename = out_pred_filename.replace("/pred", "/color")
            out_color_filename = out_color_filename.replace("_pred.png", "_color.png")
            cv2.imwrite(out_color_filename, color_pred)
        
        if hybrid:    
            out_hybrid_filename = out_pred_filename.replace("/pred", "/hybrid")
            out_hybrid_filename = out_hybrid_filename.replace("_pred.png", "_hybrid.png")
            hybrid_pred = 0.5 * frame19_full + 0.5 * color_pred        
            cv2.imwrite(out_hybrid_filename, hybrid_pred)
        
        num_iter +=1
   
    print("\n\nEvaluation Phase")
    print("---------------------")
    city_eval.main("segpred_"+short_mid, data_path, os.path.join(output_path,"pred"))
        

if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], int(sys.argv[2]))
    else:
        print("Running defaults: short_mid = t10, gpu_id = 0")
        main()
