# Recurrent Flow-Guided Semantic Forecasting
Adam M. Terwilliger, Garrick Brazil, Xiaoming Liu

## Overview
We are releasing our segmentation prediction framework for the Cityscapes dataset as detailed in our [arXiv report](https://arxiv.org/abs/1809.08318), accepted at WACV 2019. 
![Segpred Overview](http://cvlab.cse.msu.edu/images/segpred/overview.jpg)
Our proposed approach aggregates past optical flow features using a convolutional LSTM to predict future optical flow, which is used by an learnable warp layer to produce future segmentation.

For additional details, please visit our [project page](http://cvlab.cse.msu.edu/project-segpred.html).

If you utilize our framework in your work, please include this citation:

    @inproceedings{recurrent-flow-guided-semantic-forecasting, 
      author = {Adam M. Terwilliger and Garrick Brazil and Xiaoming Liu},
      title = {Recurrent Flow-Guided Semantic Forecasting},
      booktitle = {Proc. IEEE Winter Conference on Application of Computer Vision},
      month = {January},
      year = {2019},
    }
    
## Implementation
Our framework was tested with Debian GNU/Linux 8, Python 2.7, CUDA 8.0, cuDNN v5.1, OpenCV 2.4.9.1, NVIDIA Titan X GPUs, and a modified version of Caffe as provided. The included Caffe contains custom layers for PSPNet, ConvLSTM, FlowNet, and our warp layer, so it is necessary to utilize this version to successfully reproduce our results.

## Setup
For specific details regarding setup, please refer to [Installation.md](docs/Installation.md).

## Results
We release our top performing models for short-term and mid-term prediction, which achieve 67.2 mIOU and 52.5 mIOU, respectively on Cityscapes validation set. These results can be reproduced, after completing the setup, as follows:

    python segpred_val.py t3 0
    python segpred_val.py t10 0

The pre-trained models are available for download [here](https://www.cse.msu.edu/computervision/segpred-release.zip).

## Contact
For any questions regarding the framework, feel free to post an issue on this github. 

With any follow-up questions regarding the paper, don't hesitate to contact the authors at {adamtwig, brazilga, liuxm}@msu.edu.

