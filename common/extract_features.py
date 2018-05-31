import scipy.spatial.distance as scidist
from keras.applications import *
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import glob
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
import tensorflow as tf
from Image_File_IO.extract_features_iterator import extract_features_iterator


def extract_features_white_box(DIRECTORY_PATH,
                               includedCategories=['Dress', 'Skirt', 'UpperBody', 'LowerBody'],
                               imageReshape = None,
                               extractor_functions = None):

    extract_features_iterator(DIRECTORY_PATH,
                              includedCategories = includedCategories,
                              imageReshape = imageReshape,
                              extractor_functions=extractor_functions,
                              isWhiteboxExtraction=True)



def extract_features_pre_trained(DIRECTORY_PATH,
                                 model,
                                 layer_name,
                                 includedCategories=['Dress', 'Skirt', 'UpperBody', 'LowerBody'],
                                 imageReshape = 224):
    extract_features_iterator(DIRECTORY_PATH,
                              model=model,
                              layer_name=layer_name,
                              includedCategories = includedCategories,
                              imageReshape = imageReshape,
                              isWhiteboxExtraction=False)





