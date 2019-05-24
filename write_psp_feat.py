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
#   gpu_id = 0, 1, ... any gpu 
#   (nvidia-smi to see available gpus)
def main(gpu_id=0):

    # default directories
    # recommend either softlink to desired directories
    #   or renaming these variables
    data_path = "data/"
    models_path = "models/"

    # custom version of pspnet is included which achieves 74.3 mIOU on current frame
    # this network does not utilize the sliding window approach for testing
    # rather it completes a single forward pass at 512x1024 (half resolution)
    psp_net_path = os.path.join(models_path, "pspnet101_cityscapes_update.prototxt")
    psp_weights_path = os.path.join(models_path, "pspnet101_cityscapes.caffemodel")

    caffe.set_mode_gpu()
    caffe.set_device(int(gpu_id))
    
    psp_net = caffe.Net(psp_net_path, psp_weights_path, caffe.TEST)

    # generates list of filenames for validation images
    img_set = utils.gen_img_set(data_path)
    
    print("---------------------------------------------------------------------------------------------")
    print("Output path:", os.path.realpath(data_path)+"/segFeatures/pspnet/val/")
    print("---------------------------------------------------------------------------------------------")
    print("\nPre-processing Phase")
    print("----------------------")
    
    num_iter = 1
    
    for ref_frame in img_set:
   
        # short_mid == "t10":
        input_frames = [0, 3, 6, 9]
        # short_mid == "t3":
        input_frames += [7, 10, 13, 16]

        for past in range(len(input_frames)):
            
            frame0_name = utils.get_cityscapes_filename(data_path, ref_frame, input_frames[past])
            seg_feat0_name = utils.get_segfeat_filename(frame0_name)
       
            # if seg feat already there, then skip
            if not os.path.isfile(seg_feat0_name):
        
                frame0_full = cv2.imread(frame0_name)

                psp_mean = np.array([103.939, 116.779, 123.68])
                psp_frame0 = cv2.resize(frame0_full, (1017, 505)) 

                psp_rgb = psp_frame0 - psp_mean

                t_psp_rgb = psp_rgb[np.newaxis,:,:,:].transpose(0, 3, 1, 2)
                t_psp_s0, t_psp_s1, t_psp_s2, t_psp_s3 = t_psp_rgb.shape
                psp_net.blobs["psp_rgb"].reshape(t_psp_s0, t_psp_s1, t_psp_s2, t_psp_s3)
                psp_net.blobs["psp_rgb"].data[...] = t_psp_rgb

                psp_out = psp_net.forward()

                seg_feat = np.array(psp_net.blobs['conv4_23'].data[...])

                #print("Writing:", seg_feat0_name)

                np.save(seg_feat0_name, seg_feat)

            #else:
            #    print("Already completed:", seg_feat0_name)
            
            print("\rFeatures Processed: {}".format(num_iter), end=" ")
            sys.stdout.flush()

            num_iter += 1
    
    print("\n")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(int(sys.argv[1]))
    else:
        print("Running default: gpu_id = 0")
        main()
