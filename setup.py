import os
import sys
import argparse
import re
from zipfile import ZipFile
from pathlib import Path

from src.utils.get_labelsmask import get_labels
from src.utils.json2mask import get_masks, plot_labels
from src.utils.run_preprocessing import run
from src.train_cnn import train_cnn

# Aliasing  os.path.join fuction
join = os.path.join

# Data structure arguments
# Create Parser object
parser = argparse.ArgumentParser()
# add arguments:
# a. name of the file
parser.add_argument('-file', '--file', type=str, default='all_data.zip',
                    help='Name of zipped data file.')
# b. does it need to be unzipped
parser.add_argument('-unzip', '--unzip', type=bool, default=True,
                    help='Unzip?')
#???? c. does it need to convert annotations to labels 
parser.add_argument('-labels', '--labels', type=bool, default=True,
                    help='Convert annotations to labels?')
# d. does it need preprocessing?
parser.add_argument('-preprocessing', '--preprocessing', type=bool, 
                    default=False, help='Preprocessing?')
#???? e. 
parser.add_argument('-train_cnn', '--train_cnn', type=bool, default=False)

# Create a namespace object to hold the parser arguments as attributes
args = parser.parse_args()

# Aliasing of the Namespace attributes
DATA_ZIPFILE = f'data/{args.file}'
UNZIP = args.unzip
LABELS = args.labels
PREPROCESSING = args.preprocessing
TRAIN_CNN = args.train_cnn

# Save the path of the folder the file lives in, in a variable
dir_path = os.path.dirname(os.path.realpath(__file__)) 
# Save the Current Work Directory in a variable
cwd = os.getcwd() 

# Dictionary that holds all the training parameters
params = {
    'N_TRAIN': 22,
    'N_TEST': 6,
    'N_FOLDS': 10,
    'TARGET_SIZE': 128,
    'AUGMENTATION_FACTOR': 40,  
    # Multiplying n_images by this factor with rotating and flipping
    'BATCH_SIZE': 20000,  
    # Number of sampled patches
    'THRESHOLD_DEFECTIVE': 0.1,  
    # Lower limit for normalized defective area in order to be classified 
    # 'defective'
    'THRESHOLD_NONDEFECTIVE': 0.01, 
    # Upper limit for normalized defective area in order to be classified 
    # 'non_defective'
}

# Specify Directories where data is or is going to be
#???? doesn't this mean that the folder has to be inside a "data" folder?
parent_dir, name, image_format = re.split('\.|\/', DATA_ZIPFILE)
#this splits the path of the data into parent folder, name of the file and 
#format of the file, by splitting a string when either . or / are encountered

#???? Create directory paths for later use
dir_data = join(parent_dir, name)

dir_defective = join(dir_data, 'defective')
dir_defective_images = join(dir_defective, 'images')
dir_defective_annotations = join(dir_defective, 'annotations')
dir_defective_labels = join(dir_defective, 'labels')
dir_defective_json = join(dir_defective, 'json')

dir_nondefective = join(dir_data, 'non_defective')
dir_nondefective_images = join(dir_nondefective, 'images')
dir_nondefective_annotations = join(dir_nondefective, 'annotations')
dir_nondefective_labels = join(dir_nondefective, 'labels')

dir_folds = join(dir_data, f'n_train_{params["N_TRAIN"]}')
dir_output = join('output', name, f'n_train_{params["N_TRAIN"]}')

# Check if files are zipped and in case unzip
if UNZIP:
    if image_format == 'zip':
        # Unzip Data: ZipFile class object aliased as zip_ref
        with ZipFile(DATA_ZIPFILE, 'r') as zip_ref:
            # Extracts all members of the zipped archive 
            # (extractall class method)
            zip_ref.extractall(parent_dir)

if LABELS:
    # Get labels for non-defective images
    get_labels(
        dir_labels = dir_nondefective_labels,
        dir_annotated = dir_nondefective_images,
        no_label = True,
    )
    # Get labels for defective images
    get_masks(
        dir_labels = dir_defective_labels,
        dir_json = dir_defective_json,
        int_dict = {'B': 0, 'S': 0.5, 'A': 1},
    )

    # plot_labels(source_dir=Path(dir_data).joinpath("defective_json"),
    #             output_dir=Path(dir_data).joinpath("defective_labeled"),
    #             int_dict = {'B': 0, 'S': 1, 'A': 2},
    #             label_dict = {'B': 'primary symmetry', 
    #             'S': 'secondary symmetry', 'A': 'blurred'},show=False)

if PREPROCESSING:
    run(dir_data, dir_defective, dir_nondefective, dir_folds, params)

if TRAIN_CNN:
    train_cnn(dir_folds=dir_folds, output_dir=dir_output, 
              n_folds=params["N_FOLDS"])



