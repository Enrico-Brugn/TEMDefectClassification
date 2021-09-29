# %% Import block
import os
import sys
import time
import shutil
import random

# Set working directory as the same one as setup.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import tqdm


from keras.preprocessing.image import random_brightness
from sklearn.model_selection import KFold
from skimage import exposure

# from PIL import Image
import tensorflow as tf
import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array

from src.utils.modify_folders import sort_folds_by_label, remove_redundant_dirs

# %%
def create_folds_(dir_images, dir_labels, dir_target, n_train=1, n_test=4, 
                  n_folds=2, format='tif'):
    """
    From images and labels, creating n_split-times a fold, each containing the 
    structure
        -train
            -images
            -labels
        -test
            -images
            -labels
    and saving it im target directory.
    :param dir_images: Directory in which to find images
    :param dir_labels: Directory in which to find labels
    :param dir_target: Target directory
    :param kfold: ratio between size of train and test split.
    :param n_folds: Number of folds created. Needs to be equal or smaller than 
                    kfold
    :param format: Expected image format. Tested with 'tif'
    :return:
    """
    
    # Aliasing of frequently used functions
    join = os.path.join
    exists = os.path.exists
    
    # Save the current working directory in a varialbe, then print it and the
    # target directory.
    cwd = os.getcwd()
    print('cwd: ', cwd)
    print('target: ', dir_target)
    # TODO: Replace all assert statements with actual raised errors
    # Check if "dir_target" is an actual directory
    assert not os.path.isdir(dir_target)
    
    # Create a list of images and a list of labels and then print them both
    filenames_images = [
        file for file in os.listdir(dir_images) if file.endswith('.' + format)
    ]
    filenames_labels = [
        file for file in os.listdir(dir_labels) if file.endswith('.' + format)
    ]
    print(filenames_images)
    print(filenames_labels)
    
    # Make sure there are as many images as there are image labels 
    assert len(filenames_images) == len(filenames_labels)
    # Make sure we have enough pictures for training and for testing
    assert n_train + n_test <= len(filenames_images)
    
    
    for fold in range(n_folds):
        # Seed the random number generator
        # TODO Apparently np.random.seed is legacy, substitute?
        np.random.seed(fold)
        # Generate a 1D array (list) of randomly ordered numbers from 0 to
        # the number of items in "filename_images" - 1.
        all_idx = np.random.choice(range(len(filenames_images)), 
                                   len(filenames_images), replace=False)
        # Splice the list of numbers from position 0 to position "n_train"
        train_index = all_idx[:n_train]
        # Splice the list of numbers from position "n_train"+1 to position 
        # "n_train"+"n_test"
        test_index = all_idx[n_train:n_train + n_test]
        # Splice the list of numbers from position "n_train"+"n_test"+1 to the 
        # end of the list
        omitted_index = all_idx[n_train + n_test:]
        
        # Show which numbers ended up where
        print(f'Train indexes: {train_index}')
        print(f'Test indexes: {test_index}')
        print(f'Omitted indexes: {omitted_index}')
        
        # Create paths for the folders:
        # dir_data\f'fold{fold}'\'train'\'original\'images'\'0'
        # dir_data\f'fold{fold}'\'train'\'original\'labels'\'0'
        # dir_data\f'fold{fold}'\'test'\'original\'images'\'0'
        # dir_data\f'fold{fold}'\'test'\'original\'labels'\'0'
        dir_fold = join(dir_target, f'fold{fold}')
        dir_train = join(dir_fold, 'train')
        dir_test = join(dir_fold, 'test')
        dir_train_images = join(dir_train, 'original', 'images', '0')
        dir_train_labels = join(dir_train, 'original', 'labels', '0')
        dir_test_images = join(dir_test, 'original', 'images', '0')
        dir_test_labels = join(dir_test, 'original', 'labels', '0')
        
        # Create the folders (if they don't exist)
        if not exists(dir_train_images):
            os.makedirs(dir_train_images)
        if not exists(dir_train_labels):
            os.makedirs(dir_train_labels)
        if not exists(dir_test_images):
            os.makedirs(dir_test_images)
        if not exists(dir_test_labels):
            os.makedirs(dir_test_labels)
        
        # Make a list of files for training and testing, using the list of all
        # images "filename_images" and selecting the images indexed at the
        # numbers saved in "train_index" and "test_index" respectively.
        train_files = [
            str(i) + ': ' + filenames_images[i] for i in train_index]
        test_files = [
            str(i) + ': ' + filenames_images[i] for i in test_index]
        
        # Save the newly made lists in new .txt files in the folder "dir_folds"
        # TODO: is overwriting an issue?
        with open(join(dir_fold, 'train.txt'), 'w') as f:
            for item in train_files:
                f.write('%s\n' % item)
        with open(join(dir_fold, 'test.txt'), 'w') as f:
            for item in test_files:
                f.write('%s\n' % item)
        
        # Copy the files of the test batch and the train batch in the respective
        # folders. Both images and labels, correspondingly.
        for index in train_index:
            shutil.copy(
                join(dir_images, filenames_images[index]),
                join(dir_train_images))
            shutil.copy(
                join(dir_labels, filenames_labels[index]),
                join(dir_train_labels))
        for index in test_index:
            shutil.copy(
                join(dir_images, filenames_images[index]),
                join(dir_test_images))
            shutil.copy(
                join(dir_labels, filenames_labels[index]),
                join(dir_test_labels))

        print(
            f'Created fold{fold} with train/test: {train_index}/{test_index}')
        fold += 1 #???? why is this needed

    # else:
    #     fold = 0
    #     train_index = [i for i in range(len(filenames_images))]
    #     test_index = []
    #
    #     dir_fold = os.path.join(dir_target, "fold{}".format(fold))
    #     dir_train = os.path.join(dir_fold, "train")
    #     dir_test = os.path.join(dir_fold, "test")
    #     dir_train_images = os.path.join(dir_train, "original", "images", "0")
    #     dir_train_labels = os.path.join(dir_train, "original", "labels", "0")
    #     dir_test_images = os.path.join(dir_test, "original", "images", "0")
    #     dir_test_labels = os.path.join(dir_test, "original", "labels", "0")
    #
    #     if not os.path.exists(dir_train_images):
    #         os.makedirs(dir_train_images)
    #     if not os.path.exists(dir_train_labels):
    #         os.makedirs(dir_train_labels)
    #     if not os.path.exists(dir_test_images):
    #         os.makedirs(dir_test_images)
    #     if not os.path.exists(dir_test_labels):
    #         os.makedirs(dir_test_labels)
    #
    #     train_files = [str(i) + ": " + filenames_images[i] for i in train_index]
    #     test_files = [str(i) + ": " + filenames_images[i] for i in test_index]
    #
    #     with open(os.path.join(dir_fold, "train.txt"), "w") as f:
    #         for item in train_files:
    #             f.write("%s\n" % item)
    #
    #     with open(os.path.join(dir_fold, "test.txt"), "w") as f:
    #         for item in test_files:
    #             f.write("%s\n" % item)
    #
    #     for index in train_index:
    #         shutil.copy(
    #             os.path.join(dir_images, filenames_images[index]),
    #             os.path.join(dir_train_images),
    #         )
    #         shutil.copy(
    #             os.path.join(dir_labels, filenames_labels[index]),
    #             os.path.join(dir_train_labels),
    #         )
    #     for index in test_index:
    #         shutil.copy(
    #             os.path.join(dir_images, filenames_images[index]),
    #             os.path.join(dir_test_images),
    #         )
    #         shutil.copy(
    #             os.path.join(dir_labels, filenames_labels[index]),
    #             os.path.join(dir_test_labels),
    #         )
    #
    #     print(
    #         "Created fold{} with train/test: {}/{}".format(
    #             fold, train_index, test_index
    #         )
    #     )

# %%
#???? What's the difference with create_folds_?
def create_folds(dir_images, dir_labels, dir_target, kfold=1, n_folds=2, 
                 format='tif'):
    """
    From images and labels, creating n_split-times a fold, each containing the 
    structure
        -train
            -images
            -labels
        -test
            -images
            -labels
    and saving it im target directory.
    :param dir_images: Directory in which to find images
    :param dir_labels: Directory in which to find labels
    :param dir_target: Target directory
    :param kfold: ratio between size of train and test split.
    :param n_folds: Number of folds created. Needs to be equal or smaller than 
                    kfold
    :param format: Expected image format. Tested with 'tif'
    :return:
    """
    join = os.path.join
    exists = os.path.exists
    
    cwd = os.getcwd()
    print('cwd', cwd)
    print('target: ', dir_target)
    assert not os.path.isdir(dir_target) 
    # TODO: Replace all assert statements with actual raised errors
    assert n_folds <= kfold

    filenames_images = [
        file for file in os.listdir(dir_images) if file.endswith('.' + format)
    ]
    filenames_labels = [
        file for file in os.listdir(dir_labels) if file.endswith('.' + format)
    ]
    print(filenames_images)
    print(filenames_labels)

    assert len(filenames_images) == len(filenames_labels)


    if kfold != 1:
        fold = 0
        for train_index, test_index in KFold(kfold, random_state=42, 
                                             shuffle=True).split(
                                                 filenames_images):
            if n_folds <= fold:
                break

            dir_fold = join(dir_target, f'fold{fold}')
            dir_train = join(dir_fold, 'train')
            dir_test = join(dir_fold, 'test')
            dir_train_images = join(dir_train, 'original', 'images', '0')
            dir_train_labels = join(dir_train, 'original', 'labels', '0')
            dir_test_images = join(dir_test, 'original', 'images', '0')
            dir_test_labels = join(dir_test, 'original', 'labels', '0')

            if not exists(dir_train_images):
                os.makedirs(dir_train_images)
            if not exists(dir_train_labels):
                os.makedirs(dir_train_labels)
            if not exists(dir_test_images):
                os.makedirs(dir_test_images)
            if not exists(dir_test_labels):
                os.makedirs(dir_test_labels)

            train_files = [
                str(i) + ': ' + filenames_images[i] for i in train_index]
            test_files = [
                str(i) + ': ' + filenames_images[i] for i in test_index]

            with open(join(dir_fold, 'train.txt'), 'w') as f:
                for item in train_files:
                    f.write('%s\n' % item)

            with open(join(dir_fold, 'test.txt'), 'w') as f:
                for item in test_files:
                    f.write('%s\n' % item)

            for index in train_index:
                shutil.copy(
                    join(dir_images, filenames_images[index]),
                    join(dir_train_images),
                )
                shutil.copy(
                    join(dir_labels, filenames_labels[index]),
                    join(dir_train_labels),
                )
            for index in test_index:
                shutil.copy(
                    join(dir_images, filenames_images[index]),
                    join(dir_test_images),
                )
                shutil.copy(
                    join(dir_labels, filenames_labels[index]),
                    join(dir_test_labels),
                )

            print(
            f'Created fold{fold} with train/test: {train_index}/{test_index}')
            fold += 1

    else:
        fold = 0
        train_index = [i for i in range(len(filenames_images))]
        test_index = []

        dir_fold = join(dir_target, f'fold{fold}')
        dir_train = join(dir_fold, 'train')
        dir_test = join(dir_fold, 'test')
        dir_train_images = join(dir_train, 'original', 'images', '0')
        dir_train_labels = join(dir_train, 'original', 'labels', '0')
        dir_test_images = join(dir_test, 'original', 'images', '0')
        dir_test_labels = join(dir_test, 'original', 'labels', '0')

        if not exists(dir_train_images):
            os.makedirs(dir_train_images)
        if not exists(dir_train_labels):
            os.makedirs(dir_train_labels)
        if not exists(dir_test_images):
            os.makedirs(dir_test_images)
        if not exists(dir_test_labels):
            os.makedirs(dir_test_labels)

        train_files = [
            str(i) + ': ' + filenames_images[i] for i in train_index]
        test_files = [str(i) + ': ' + filenames_images[i] for i in test_index]

        with open(join(dir_fold, 'train.txt'), 'w') as f:
            for item in train_files:
                f.write('%s\n' % item)

        with open(join(dir_fold, 'test.txt'), 'w') as f:
            for item in test_files:
                f.write('%s\n' % item)

        for index in train_index:
            shutil.copy(
                join(dir_images, filenames_images[index]),
                join(dir_train_images),
            )
            shutil.copy(
                join(dir_labels, filenames_labels[index]),
                join(dir_train_labels),
            )
        for index in test_index:
            shutil.copy(
                join(dir_images, filenames_images[index]),
                join(dir_test_images),
            )
            shutil.copy(
                join(dir_labels, filenames_labels[index]),
                join(dir_test_labels),
            )

        print(
            f'Created fold{fold} with train/test: {train_index}/{test_index}')
        
# %%
def flip_intensity(image):
    """
    Flips intensity of image
    :param image: np.array with integers between 0 and 255
    :return:
    """
    image_flipped = image
    image_flipped[:, :, 0] = -1 * image[:, :, 0] + 255
    return image_flipped

# %%
def augment_folds(dir_data, m, random_state=2):
    """
    Randomly augmenting images contained in single or multiple folds.
    In each fold it creates a directory 'augmented' with the same structure as
    'original'
    :param dir_data: Directory where folds are located.
                     folds need to be directories called "fold*" with the 
                     following structure:
                     train
                        original
                            images
                            labels
                     test
                        original
                            images
                            labels

    :param m: factor of augmentation.
    :return: this function does not return anything
    """
    
    # Alias frequently used function
    join = os.path.join
    
    # Dictionary detailing image augmentation parameters to be fed to 
    # ImageDataGenerator class. Documentation at:
    # keras.preprocessing.image.ImageDataGenerator
    datagen_args_img = dict(
        rotation_range = 90,
        fill_mode = 'constant',
        cval = 255,
        # brightness_range=(0.5, 1),
        zoom_range = 0.05, 
## Seems to work now. TODO: -> fix keras_preprocessing library before running!
        horizontal_flip = True,
        vertical_flip = True,
        preprocessing_function = preprocess_input,
    )

    datagen_args_label = dict(
        rotation_range=90,
        fill_mode ='constant',
        cval = 255,
        # brightness_range=(0.5, 1),
        zoom_range = 0.05, ## Does not work if enabled!
        horizontal_flip = True,
        vertical_flip = True,
        preprocessing_function = preprocess_input,
    )
    
    # create 2 objects of the ImageDataGenerator class using the dictionaries
    # just created
    image_datagen = ImageDataGenerator(**datagen_args_img)
    label_datagen = ImageDataGenerator(**datagen_args_label)
    
    # Create a list of "fold" directories (folders) containing the data  
    # structure detailed in :param dir_data:
    folds = [
        folder for folder in os.listdir(dir_data) if folder.startswith('fold')]
    
    #???? Why do you redefine random_state?
    random_state = random_state
    
    # tdqm for progress bar
    for i in tqdm.trange(len(folds)):
        # fold is the i-eth element of the list folds
        fold = folds[i]
        
        # save the path to fold\'train' and fold\'test'
        dir_fold = join(dir_data, fold)
        dir_train = join(dir_fold, 'train')
        dir_test = join(dir_fold, 'test')
        
        for dir in [dir_train, dir_test]:
            # print("LISTDIR:")
            # print(os.listdir(os.path.join(dir, "original", "images", "0")))
            if os.listdir(join(dir, 'original', 'images', '0')) == []:
                print('Empty directory')
                continue

            random_state += 1
            # print("DIR: ", dir)

            # save the path to: fold\dir\'original' in a variable
            dir_original = join(dir, 'original')
            
            # create the paths to:
            # fold\dir\'augmented'\'images'\'0'
            # fold\dir\'augmented'\'labels'\'0'
            dir_augmented = join(dir, 'augmented')
            dir_augmented_images = join(dir_augmented, 'images', '0')
            dir_augmented_labels = join(dir_augmented, 'labels', '0')
            # delete the folders corresponding to the paths just created if 
            # they already exist
            if os.path.exists(dir_augmented):
                shutil.rmtree(dir_augmented, ignore_errors=True)
            # create the folders with the paths just made
            os.makedirs(dir_augmented_images)
            os.makedirs(dir_augmented_labels)
            
            # Create 2 DirectoryIterator objects for images and labels using
            # a common seed, these contain a tubple of (x,y) where x contains a
            # numpyarray of images and y a numpy of corresponding labels. Note
            # that in this case we don't use y, or, rather, all images are
            # labeled the same.
            # Documentation at: 
            # keras.preprocessing.image.ImageDataGenerator.flow_from_directory
            # keras.preprocessing.image.DirectoryIterator
            image_generator = image_datagen.flow_from_directory(
                join(dir, 'original', 'images'),
                target_size = (1024, 1024),
                color_mode = 'grayscale',
                batch_size = 1000,  # Max value, choose larger than n_samples
                class_mode = None,
                shuffle = True,
                seed = random_state,  
                # VERY IMPORTANT: NEEDS TO BE THE SAME FOR IMAGE_GENERATOR AND 
                # LABEL_GENERATOR
            )

            label_generator = label_datagen.flow_from_directory(
                join(dir, 'original', 'labels'),
                target_size = (1024, 1024),
                color_mode = 'grayscale',
                batch_size = 1000,  # Max value, choose larger than n_samples
                class_mode = None,
                shuffle = True,
                seed = random_state,  
                # VERY IMPORTANT: NEEDS TO BE THE SAME FOR IMAGE_GENERATOR AND
                # LABEL_GENERATOR
            )

            j_image = 0
            j_label = 0
            # Create m additional images and corresponding labels from the 
            # DirectoryIterator object. 
            for j in tqdm.trange(m):
                # print(j)
                
                # Iterate to the next element in the images DirectoryIterator 
                # object effectively saving a (image,label) tuple in X. This 
                # lable is not the same lable we use.
                X = image_generator.__next__()
                for element in X:
                    # save the image just estracted from the iterator in 
                    # dir_augmented_images\'j_image.tif'
                    save_img(
                        join(dir_augmented_images, f'{j_image}.tif'),
                        element,
                    )
                    j_image += 1
                
                # Iterate to the next element in the lable DirectoryIterator
                # object, effectively saving a (image,lable) tuple in y. The 
                # image part of the tuple is actually the lable corresponding 
                # image just saved from x.
                y = label_generator.__next__()
                for element in y:
                    # save the label just estracted from the iterator in 
                    # dir_augmented_images\'j_label.tif'
                    save_img(
                        join(dir_augmented_labels, f'{j_label}.tif'),
                        element,
                    )
                    j_label += 1

# %%
def randomcrop_folds(dir_data, crop_target=None, batch_size=50, 
                     intensity_flip=True, rescale_intensity=True):
    """
    Randomly crops patches from images. It validates if they are free of 
    border/white areas, then saves them.
    dir_data -- Directory where folds are located.
                     folds need to be directories called "fold*" with the 
                     following structure:
                     train
                        original
                            images
                            labels
                     test
                        original
                            images
                            labels
    crop_target --
    batch_size --
    rescale_intensity --
    remove_original --
    :return:
    """
    join = os.path.join
    
    # print("Cropping")
    # Create a list of "fold" directories (folders) containing the data  
    # structure detailed in :param dir_data:
    folds = [file for file in os.listdir(dir_data) if file.startswith('fold')]

    random_state = 2
    for k in tqdm.trange(len(folds)):
        fold = folds[k]
        dir_fold = join(dir_data, fold)
        dir_train = join(dir_fold, 'train')
        dir_test = join(dir_fold, 'test')

        print('Cropping fold')
        for dir in [dir_train, dir_test]:
            # print("Directory: " + dir)
            dir_original = join(dir, 'original')
            dir_augmented = join(dir, 'augmented')
            if os.path.exists(dir_augmented):
                if os.listdir(join(dir_augmented, 'images')) == []:
                    # print("Empty directory")
                    continue
            else:
                print('Directory does not exist.')
                continue

            # print("LISTDIR:")
            # print(os.listdir(os.path.join(dir_augmented, "images")))



            # dir_augmented_cropped = os.path.join(dir, "augmented_cropped")

            dir_augmented_images = join(dir_augmented, 'images', '0')
            dir_augmented_labels = join(dir_augmented, 'labels', '0')

            dir_augmented_cropped_images = join(dir, 'patches', 'images')
            dir_augmented_cropped_labels = join(dir, 'patches', 'labels')
            # if os.path.exists(dir_augmented_cropped):
            #     shutil.rmtree(dir_augmented_cropped, ignore_errors=True)
            os.makedirs(dir_augmented_cropped_images)
            os.makedirs(dir_augmented_cropped_labels)

            batch_images_crops = []
            batch_labels_crops = []

            i = 0
            j = 0
            random_state = 2
            while i < batch_size:
                random_state += 1
                j += 1
                random.seed(a = random_state)
                images_path = join(dir_augmented_images,
                    random.choice(sorted(os.listdir(dir_augmented_images))),
                )
                random.seed(a = random_state)
                labels_path = join(
                    dir_augmented_labels,
                    random.choice(sorted(os.listdir(dir_augmented_images))),
                )

                # print(images_path)
                images = img_to_array(load_img(images_path, 
                                               color_mode = 'grayscale'))
                labels = img_to_array(load_img(labels_path, 
                                               color_mode = 'grayscale'))

                # labels = labels.reshape([1024, 1024])
                # plt.imshow(labels)
                # plt.show()
                #
                # print(np.unique(labels))
                # exit()

                batch_images_temp = random_crop(
                    images, (crop_target, crop_target), seed = random_state
                )
                batch_labels_temp = random_crop(
                    labels, (crop_target, crop_target), seed = random_state
                )

                # batch_y_temp = batch_y[i]

                max = np.mean(
                    np.partition(
                        batch_images_temp[:, :, 0].flatten(), -10)[-10:]
                )

                if max != 255.0:  # Making sure to cover no edge!

                    if intensity_flip and random.randint(0, 1) == 1:
                        batch_images_temp = flip_intensity(batch_images_temp)

                    # if rescale_intensity:
                    #     p2, p98 = np.percentile(batch_images_temp, (2, 98))
                    #     batch_images_temp = exposure.rescale_intensity(
                    #                   batch_images_temp, in_range=(p2, p98))

                    save_img(
                        join(
                            dir_augmented_cropped_images, f'{i}.tif'
                        ),
                        batch_images_temp
                    )
                    save_img(
                        join(
                            dir_augmented_cropped_labels, f'{i}.tif'
                        ),
                        batch_labels_temp, scale = False
                    )

                    i += 1
                    #
                    # plt.imshow(batch_images_temp[:, :, 0])
                    # plt.imshow(batch_labels_temp[:, :, 0])
                    # plt.title("PASSED")
                    # plt.show()

                else:
                    # print("Skipped")
                    # plt.imshow(batch_images_temp[:, :, 0])
                    # plt.imshow(batch_labels_temp[:, :, 0])
                    # plt.title("SKIPPED")
                    # plt.show()
                    pass

            print(f'{i} of {j} passed')
            # batch_images_crops = np.array(batch_images_crops)
            # batch_labels_crops = np.array(batch_labels_crops)

# %%
def random_crop(img, random_crop_size, seed=None):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3 or img.shape[2] == 1
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    np.random.seed(seed = seed)
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y : (y + dy), x : (x + dx), :]

# %%
def preprocess_generator(X_dir, y_dir = None, channels = 1, batch_size = 30, 
                         seed = None):
    """
    Data generator, yields preprocessed images.
    TODO: Why do I need this?
    :param X_dir:
    :param channels:
    :param batch_size:
    :param seed:
    :return:
    """
    while True:
        batch_x_crops = []
        i = 0
        while i < batch_size:
            randomchoiceseed = random.randint(1, 10000)
            random.seed(a = randomchoiceseed)
            sample_path = random.choice(sorted(os.listdir(X_dir)))
            sample = load_img(
                os.path.join(X_dir, sample_path), color_mode = 'grayscale'
            )
            sample = img_to_array(sample)
            batch_x_crops.append(sample)
            i += 1
        batch_x_crops = np.array(batch_x_crops)

        preprocess_input(batch_x_crops)

        if y_dir is None:
            yield (batch_x_crops, batch_x_crops)
        else:
            pass

# %%
def get_spectrum(target_path, output_path):
    """
    Gets image, calculates 2dfft, shifts&computes magnitude, saves.
    :param target_path:
    :param output_path:
    :return:
    """
    img = cv2.imread(target_path, 0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    cv2.imwrite(output_path, magnitude_spectrum.astype(int))

    # plt.subplot(121)
    # plt.imshow(img, cmap="gray")
    # plt.subplot(122)
    # plt.imshow(magnitude_spectrum, cmap="gray")
    # plt.show()
    return
# %%
def get_spectrum_for_cnn(parent_dir, folds=1):
    """
    For cnn structure, as produced by modify_folders.make_cnn_structure, 
    calculate and save fourier spectras.
    :param parent_dir:
    :param folds:
    :return:
    """
    join = os.path.join
    
    for fold in range(folds):
        dir_fold = join(parent_dir, f'fold{fold}')

        train_nondefective = join(dir_fold, 'train', 'non_defective')
        train_defective = join(dir_fold, 'train', 'defective')
        test_nondefective = join(dir_fold, 'test', 'non_defective')
        test_defective = join(dir_fold, 'test', 'defective')
        cnn = [train_nondefective, train_defective, test_nondefective, 
               test_defective]

        spectraltrain_nondefective = join(dir_fold, 'spectral_train', 
                                          'non_defective')
        spectraltrain_defective = join(dir_fold, 'spectral_train', 'defective')
        spectraltest_nondefective = join(dir_fold, 'spectral_test', 
                                         'non_defective')
        spectraltest_defective = join(dir_fold, 'spectral_test', 'defective')
        cnn_spectral = [spectraltrain_nondefective, spectraltrain_defective, 
                        spectraltest_nondefective, spectraltest_defective]

        for i in range(len(cnn)):
            if os.path.exists(cnn_spectral[i]):
                shutil.rmtree(cnn_spectral[i])
            os.makedirs(cnn_spectral[i])
            for file in os.listdir(cnn[i]):
                target_path = join(cnn[i], file)
                output_path = join(cnn_spectral[i], file)
                get_spectrum(target_path, output_path)


# %%
# augmented_dir = "../data/training/temp/train/0"
#
# generator = crop_generator(augmented_dir, brightness_range=[0.1, 0.2])
# print("Done")
# j = 0
# for i in generator:
#     print(j)
#     j += 1
#     plt.imshow(i[0][0,:,:,0])
#     print(i[0][0,:,:,0])
#
#     plt.show()
#     exit()

if __name__ == "__main__":
    dir_images = "/home/nik/Documents/Defect Classification/Data/cubic/defective/preprocessed"
    dir_labels = "/home/nik/Documents/Defect Classification/Data/cubic/defective/labels"
    dir_target = "/home/nik/Documents/Defect Classification/Data/cubic/defective/folds"

    kfold = 2
    folds = 2

    create_folds(dir_images=dir_images, dir_labels=dir_labels, 
                 dir_target=dir_target, kfold=kfold, n_folds=folds)
    augment_folds(dir_data=dir_target, m=1)
    randomcrop_folds(dir_data=dir_target, crop_target=256, batch_size=50, 
                     intensity_flip=True)
    sort_folds_by_label(dir_target=dir_target, n_folds=folds, 
                        threshold_defective=0.1, threshold_nondefective=0.01, 
                        nolabels=False)
    remove_redundant_dirs(dir_target, folds, "augmented", "patches")
    #


















    # WORKED ONCE
    # dir_data = "../../data/cubic/non_defective"
    # kfold = 6
    # folds = 1
    # # create_folds(dir_data=dir_data, kfold=kfold, n_folds=folds)
    # # augment_folds(dir_data=dir_data, m=10, intensity_flip=True)
    # crop_folds(dir=dir_data,  crop_target=256, batch_size=5000, 
    #            remove_original=False)
    #
    # for fold in range(folds):
    #     print("fold{}".format(fold))
    #     print("train")
    #     sort_by_label(dir_data + "/fold{}/train/".format(str(fold)), 
    #                   threshold=0.05, nolabels=True)
    #     print("test")
    #     sort_by_label(dir_data + "/fold{}/test/".format(str(fold)), 
    #                   threshold=0.05, nolabels=True)
    # # exit()
    #
    # dir_data = "../../data/cubic/defective"
    # kfold = 1
    # folds = 1
    # # create_folds(dir_data=dir_data, kfold=kfold, n_folds=folds)
    # # augment_folds(dir_data=dir_data, m=10, intensity_flip=True)
    # crop_folds(dir=dir_data,  crop_target=256, batch_size=1000, 
    #            remove_original=False)
    #
    # for fold in range(folds):
    #     print("fold{}".format(fold))
    #     sort_by_label(dir_data + "/fold{}/train/".format(str(fold)), 
    #                   threshold=0.04)
    #     # sort_by_label(dir_data + "/fold{}/test/".format(str(fold)), 
    #                     threshold=0.05)
    #
    #
    # exit()

