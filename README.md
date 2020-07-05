# TEM-Defect-Classification
Hi! You've reached the code repository corresponding to the paper "Learning-based Defect Recognition for Quasi-Periodic Microscope Images". This document will guide you through the setup and different steps of the code. You can either test the algorithm directly with one of your images, or train the model again with your own training data.


### Testing Pipeline
The following procedure works best if you have TEM images of cubic crystals, taken at image resolution of a few nanometers per side, or if you have your own trained classification model. If this is not the case, go directly to the training pipeline. 
1. Download repository and update your python environment.
   * If you are familiar with git, clone the repository with the following command:
     ```
     git clone ...
     ```
     Alternatively, on the top right page of this page, download the zipped version of the repository and unpack at your desired place.
   * Install the latest version of anaconda, then run the following command to set up the python environment:
     ```
     conda env create -f temdefect_env.yml
     conda activate temdefect_env
     ```
2. Update model and data directory. 
   * [Ask us](mailto:nik.dennler@uzh.ch) for the pre-trained tensor flow model or BYO (e.g. `vgg16_finetuned.h5`), then add it to the `models` directory. Pack your `tif` formatted images to a directory (e.g. `/test_images/`) and add such into the `data` directory. 
3. Run the Eigenfilter segmentation algorithm
   * Open the repository with your terminal and run the following command:
     ```
     export PYTHONPATH=`pwd`
     python src/eigenfilter_segmentation.py --image_dir test_images --model `vgg16_finetuned.h5`
     ```
    In case you are using an IDE such as Spyder or PyCharm, make sure that the interpreter takes the repository as your root directory. 

### Training Pipeline
If you have different images than described above, it's best to train the classification model again. Thus you need to have a bunch of training images. In our case we were successful with 13 unique images (don't add augmentation, this will be done automatically!), but depending on the complexity of the crystal class this number might be higher. As always, the more data the better.  
1. Download repository and update your python environment.
   * If you are familiar with git, clone the repository with the following command:
     ```
     git clone ...
     ```
     Alternatively, on the top right page of this page, download the zipped version of the repository and unpack at your desired place.
   * Install the latest version of anaconda, then run the following command to set up the python environment:
     ```
     conda env create -f temdefect_env.yml
     conda activate temdefect_env
     ```
2. Prepare your data
  * Collect images and safe in `tif` format.
  * Annotate training images.
    * ATTENTION: Annotations must ...
      * ... be of the color RED ([255, 0, 0]), GREEN ([0, 255, 0]) or BLUE ([0, 0, 255])
      * ... be continuous lines (no dashes!) that form a closed loop
      * ... not touch the walls or be so close to the walls that there is no space from them to flow. 
    * Let me explain this further. You can imagine the algorithm to work like water flowing from the walls, only halting of borders of a specific color. Everything covered by the water will be black (zeros), everything else white (ones). If step 2 doesn't work for some of your annotated images, try again for them fulfilling above points.

3. Run setup script
 * Decide if you want to UNZIP, produce LABELS and/or do PREPROCESSING (each either `True` or `False`). Run the setup file with the following command:
  ```
  python setup.py --unzip True --labels True --preprocessing False 
  ```
  * The preprocessing can take very long (depending on your parameters several hours). It can make sense to first only do UNZIP and LABELS, then check the results of the labels. If you're not satisfied, delete the directory, do the annotations again, and run again the setup file. If you're satisfied, run again with only PREPROCESSING activated.

4. Train model
 * The following is best done on a GPU device. Basically one has to run the `train_cnn.py` file with the desired arguments. For example, the models for six different splits have been trained with the following command: 
 
 ```
 python3 train_cnn.py -train_dir "../data/cubic/cnn_kfold_128_more/fold$i/train/" -val_dir "../data/cubic/cnn_kfold_128_more/fold$i/test/" -output_dir "../output/vgg16_128_more_finetune/" -name cnn_transfer_imagenet_fold$i -base VGG16 -weights 'imagenet' -train_all True -e 100 -fc 1024 1024 512 -d 0.5 -b 8 -l 0.00001
 ```
 Next to the obvious file paths for training data, validation data and output directory, we specify different parameters:
 * -name: name of the experiment
 * -base: base network. use one available from keras.applications
 * -weights: use weights either trained on ImageNet ('imagenet') or randomly initialized ('None')
 * -fc: fine-tuning layers, fully connected. each entry is the number of neurons per layer
 * -d: dropout fraction
 * -l: learning rate
 * -e: epochs
 * -b: batch size
 
5. Run segmentation

 * See in the above section on how to run the testing pipeline.
