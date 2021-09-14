import os
import sys
import shutil
import tqdm

#???? sets the directory to the one "setup" is in
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.utils.preprocessing import preprocess_folder
from src.utils.augmentation import create_folds, create_folds_, augment_folds,\
                                   randomcrop_folds, get_spectrum_for_cnn
from src.utils.modify_folders import sort_folds_by_label, sort_folds_by_label_,\
                                     remove_redundant_dirs, make_cnn_structure


def run(DIR_PARENT, DIR_DEFECTIVE, DIR_NONDEFECTIVE, DIR_FOLDS, PARAMS):
    """

    :param DIR_PARENT:
    :param DIR_DEFECTIVE:
    :param DIR_NONDEFECTIVE:
    :param DIR_FOLDS:
    :param PARAMS:
    :return:
    """
    # %% Aliasing of params dictionary values
    N_TRAIN = PARAMS['N_TRAIN']
    N_TEST = PARAMS['N_TEST']
    N_FOLDS = PARAMS['N_FOLDS']
    TARGET_SIZE = PARAMS['TARGET_SIZE']
    # Multiplying n_images by this factor with rotating and flipping
    AUGMENTATION_FACTOR = PARAMS['AUGMENTATION_FACTOR']
    # Number of sampled patches
    BATCH_SIZE = PARAMS['BATCH_SIZE']
    # Lower limit for normalized defective area in order to be classified 
    # 'defective'
    THRESHOLD_DEFECTIVE = PARAMS['THRESHOLD_DEFECTIVE']
    # Upper limit for normalized defective area in order to be classified 
    # 'non_defective'
    THRESHOLD_NONDEFECTIVE = PARAMS['THRESHOLD_NONDEFECTIVE']
    
    # %% Preprocessing folders
    # Create a list of directories
    dirs = [DIR_DEFECTIVE]#, DIR_NONDEFECTIVE]
    for directory in dirs:
        dir_original = os.path.join(directory, 'images')
        dir_preprocessed = os.path.join(directory, 'preprocessed')
        dir_labels = os.path.join(directory, 'labels')
        dir_target = os.path.join(directory, 'folds')

        _nolabels = False
        if directory == DIR_NONDEFECTIVE:
            _nolabels = True

        # Preprocessing
        preprocess_folder(directory_original = dir_original, 
                          directory_preprocessed = dir_preprocessed)

        # Splitting into folds
        create_folds_(dir_images = dir_preprocessed, dir_labels = dir_labels, 
                      dir_target = dir_target, n_train = N_TRAIN, 
                      n_test = N_TEST, n_folds=N_FOLDS)

        # create_folds(dir_images=dir_preprocessed, dir_labels=dir_labels, 
        #              dir_target=dir_target, kfold=KFOLD,
        #               n_folds=N_SPLITS)
        # exit()

        # %% Augmenting
        print('AUGMENTING...')
        augment_folds(dir_data = dir_target, m = AUGMENTATION_FACTOR)
        print('RANDOM CROPPING...')
        randomcrop_folds(dir_data = dir_target, crop_target = TARGET_SIZE, 
                         batch_size = BATCH_SIZE, intensity_flip = True)

        # exit()
        # sort_folds_by_label(dir_target = dir_target, n_folds = N_SPLITS, 
        #                     thr_def = THRESHOLD_DEFECTIVE, 
        #                     thr_nondef = THRESHOLD_NONDEFECTIVE, 
        #                     nolabels=False)
        print('SORT FOLD BY LABEL...')
        sort_folds_by_label_(dir_target = dir_target, n_folds = N_FOLDS, 
                             threshold_defective = THRESHOLD_DEFECTIVE,
                             threshold_nondefective = THRESHOLD_NONDEFECTIVE, 
                             nolabels = False)

        # Deleting redundant files to save storage
        print('REMOVE REDUNDANT DIRS...')
        # remove_redundant_dirs(dir_target, N_FOLDS, "augmented", "patches")
    
    # %% making CNN structure
    # For each fold, make CNN structure, tqdm for progress bars
    for fold in tqdm.trange(N_FOLDS):
        make_cnn_structure(dir_target = DIR_PARENT,
                           dir_output = DIR_FOLDS,
                           fold = fold,
                           balanced = True)

    # %% Delete redundant
    for directory in dirs:
        dir_target = os.path.join(directory, 'folds')
        if os.path.exists(dir_target):
            print(f'Deleting directory at {dir_target}')
            # deletes dir_target and all the directories it contains
            shutil.rmtree(dir_target, ignore_errors=True)

#
# if __name__ == '__main__':
#     # Define directories
#     DIR_PARENT = "../../data/cubic"
#     DIR_DEFECTIVE = os.path.join(DIR_PARENT, "defective/")
#     DIR_NONDEFECTIVE = os.path.join(DIR_PARENT, "non_defective/")
#     DIR_FOLDS = os.path.join(DIR_PARENT, "6fold_128")
#     run(DIR_PARENT, DIR_DEFECTIVE, DIR_NONDEFECTIVE, DIR_FOLDS)