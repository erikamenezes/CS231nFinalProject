import scipy.spatial.distance as scidist
from keras.applications import *
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import glob
import tensorflow as tf
from common.concatenate_extracted_features_with_feature_functions import extract_features_concat


def extract_features_iterator(DIRECTORY_PATH, base_model, layer_name, includedCategories=['Dress', 'Skirt', 'UpperBody', 'LowerBody'], imageSize = 224, isWhiteboxExtraction = True, extractor_functions = []):
    if(isWhiteboxExtraction):
        assert isinstance(extractor_functions, list)
        file_ext = "_whitebox_features"
    else:
        file_ext = "_" + base_model.name + "_features"
        model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

    for filename in glob.iglob(DIRECTORY_PATH + '/**/*_photos.npy', recursive=True):
        array = np.load(filename).transpose([0, 2, 3, 1])
        name = filename.split('/')
        photo_type = "consumer" if "consumer" in name[-1] else "shop"
        subCategory = name[-2]
        parentCategory = name[-3]
        if subCategory in includedCategories:
            newFileName = DIRECTORY_PATH + parentCategory + "/" + subCategory + "/" + photo_type + file_ext

            print("Input file {}, Array Shape: {}".format(filename, array.shape))

            if(imageSize is not None):
                array = resizeImage(array, imageSize)

            print(array.shape)
            print(newFileName)

            print("Extracting features")
            output = extract_features(array, model, extractor_functions, isWhiteboxExtraction)
            assert (array.shape[0] == output.shape[0])
            print(output.shape)
            np.save(newFileName, output)



def extract_features(images, model, extractor_functions, isWhiteboxExtraction):
    if(isWhiteboxExtraction):
        assert isinstance(extractor_functions, list)
        return extract_features_concat(images, extractor_functions)
    else:
        return model.predict(images, verbose=1)



def resizeImage(array, imageSize):
    resized_tesnor = tf.image.resize_images(array, (imageSize, imageSize),
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    with tf.Session().as_default():
        resized = resized_tesnor.eval()

    print("Resizing to {}".format(imageSize))
    return resized