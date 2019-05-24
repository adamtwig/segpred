## Installation

- **Clone the repo**
    ``` bash
    git clone https://github.com/adamtwig/segpred.git
    ```
- **Install dependencies**
    - Operating System: our framework has been tested on multiple versions of Linux, but we cannot attest to its support on Windows or Mac. We recommend a recent version of Ubuntu. 
    <!---- Python: both Python 2.7 and Python 3.6 were tested.--->
    - Python 2.7.9 was tested for this framework. We are in the process of testing Python 3.6+. 
    - CUDA / cuDNN: Only the specified versions of CUDA and cuDNN are supported, so these should be installed via [CUDA 8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive) and [cuDNN v5.1](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/prod_20161129/8.0/cudnn-8.0-linux-x64-v5.1-tgz) where you will have to create a NVIDIA Developer account, if you don't already have it.
    
     - Update ~/.bashrc with respect to CUDA and paths, as an example:
    ``` bash
    export LD_PRELOAD=$LD_PRELOAD:/usr/local/cuda-8.0/lib64/libcudnn.so.5.1.10
    export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/lib64/libcudnn.so.5.1.10:/user/adamtwig/hdf5_dir/hdf5/lib
    export CUDA_HOME=/usr/local/cuda-8.0
    ```
    - Getting dependencies setup correctly can be a frustrating process, so feel free to post an issue detailing any specifics errors you are finding.

- **Install Caffe**
    - Update makefile.config for any system specifics.
    
    - Make Caffe and pycaffe:
    ``` bash
    cd external/caffe
    make -j8 && make pycaffe
    ```
    
    - If you encounter any errors during this process, additional installation instructions can be found at the [Caffe website](http://caffe.berkeleyvision.org/installation.html) and specific issues may be found at the [Caffe Github](https://github.com/BVLC/caffe/issues).
    
    - Update ~/.bashrc to include Caffe in Python path, as an example:
    ``` bash
    export PYTHONPATH=/scratch/adamtwig/segpred/external/caffe/python:$PYTHONPATH
    ```

- **Download Cityscapes video dataset**
    - First an account must be created, if not already created, at [Cityscapes](https://www.cityscapes-dataset.com/login/).
    - Next download the label dataset named gtFine_trainvaltest.zip (241MB).
    - Finally download the full sequence dataset named leftImg8bit_sequence_trainvaltest.zip (324GB). Note this dataset contains >300 GB, so make sure to have a large enough hard-drive or server capacity. Since we only care about validation sequences, we can delete the other folders, after uncompressing the download, which will leave ~34GB.
    - Store both of these datasets in the same directory. For example, my data path looks like:
    
     /scratch/adamtwig/working/datasets/cityscapes
    
    which contains both:
    
    gtFine and leftImg8bit_sequence.

- **Setup segpred paths**
    - There are three custom paths we need: data, models, and results.
    - For data, I recommend creating a softlink to your actual data folder to "data" in the segpred repo. For example,
      ``` bash
      ln -s /scratch/adamtwig/working/datasets/cityscapes data
      ```
   - For models and results, if you have space on your hard drive, I recommend making these directories locally in segpred. If you don't have space, you can softlink them to another folder similar to the data example.

- **Download pre-trained models**
    - We have provided three pre-trained models and two model prototxts, accessible for download [here](https://www.cse.msu.edu/computervision/segpred-release.zip). First is pre-trained PSPNet on Cityscapes for convenience: pspnet101_cityscapes.caffemodel. Next are our top performing short-term (t3) and mid-term (t10) models appropriately named: segpred_t3.caffemodel and segpred_t10.caffemodel. Copy or move these models to your models directory, where the prototxt may already be, if the models directory is local within segpred.
    
- **Complete pre-processing**
    - Our framework requires that we pre-process and cache off segmentation features from PSPNet. We have included a custom version of PSPNet which does not utilize the sliding window approach for testing, but rather completes a single forward pass at half resolution (512x1024) and achieves 74.3 mean IOU on Cityscapes validation. We utilize four previous frames for both short-term and mid-term, so we provide a pre-processing script to cache these segmentation features ran as follows:
    ``` bash
    python write_psp_feat.py 0
    ```

- **Running validation script**
    - Now that we have installed our dependencies, built the custom version of Caffe, downloaded the Cityscapes video dataset, setup the segpred paths, downloaded the pre-trained models, and completed the PSPNet pre-processing, we are finally ready to run the segpred validation script as simply as follows:
    ``` bash
    python segpred_val.py t3 0
    python segpred_val.py t10 0
    ```   
   
- **Additional evaluation**
    - We provide a modified, more efficient version of the cityscapes evaluation script which outputs only the per-class results formatted for a LaTeX table. This script found in [utils/eval](../utils/eval) will be called when the validation script has processed all 500 validation images.
    
    - If you find the evaluation phase is very slow, you should run the cython setup.py script in the directory.
    
    - We also provide the official Cityscapes validation code, we can be built and run as follows, assuming we have already ran segpred_val.py successfully, as an example:
    ``` bash
    pip install cityscapesscripts --user
    cd external/cityscapesScripts
    python setup.py build_ext --inplace
    cd cityscapesscripts/evaluation
    export CITYSCAPES_DATASET=/scratch/adamtwig/working/datasets/cityscapes/
    export CITYSCAPES_RESULTS=/scratch/adamtwig/segpred/results/segpred_t3/val/pred/
    python evalPixelLevelsemanticLabeling.py
    ```
    Additional information can be found at [cityscapesScripts](https://github.com/mcordts/cityscapesScripts).
    
- **Additional visualization**
    - We provide a way to better visualize the segpred output. You can include either color images or a rgb / color hybrid images in your output by updating the "color" or "hybrid" variables in the segpred_val.py script to be "True". 
    <!---- An example of what each of these output images would look like is shown below:--->

