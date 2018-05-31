
from keras.models import Model
import numpy as np
import glob
import tensorflow as tf
from common.concatenate_extracted_features_with_feature_functions import extract_features_concat
POSSIBLE_CATEGORIES = ['Dress', 'Skirt', 'UpperBody', 'LowerBody']


def extract_features_iterator(DIRECTORY_PATH, model = None, layer_name = None, includedCategories=['Dress', 'Skirt', 'UpperBody', 'LowerBody'], imageReshape = None, isWhiteboxExtraction = True, extractor_functions = None):
    if(isWhiteboxExtraction):
        print("Extracting Whitebox features")
        assert isinstance(extractor_functions, list)
        file_ext = "_whitebox_features"
    else:
        print("Extracting Pre-trained features")
        file_ext = "_" + model.name + "_features"
        model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    for filename in glob.iglob(DIRECTORY_PATH + '/**/*_photos.npy', recursive=True):
        array = np.load(filename).transpose([0, 2, 3, 1])
        name = filename.split('/')
        photo_type = "consumer" if "consumer" in name[-1] else "shop"
        subCategory = name[-2]
        if subCategory not in POSSIBLE_CATEGORIES or subCategory in includedCategories:
            newFileName = "/".join(name[:-1]) + "/" + photo_type + file_ext
            print(newFileName)
            print("Input file {}, Array Shape: {} \n".format(filename, array.shape))

            if(imageReshape is not None):
                array = resizeImage(array, imageReshape)

            print("Output file name {} \n".format(newFileName))

            print("Extracting features \n")
            output = extract_features(array, model, extractor_functions, isWhiteboxExtraction)
            assert (array.shape[0] == output.shape[0])
            print("Output Array Shape: {} \n".format(output.shape))
            np.save(newFileName, output)



def extract_features(images, model, extractor_functions, isWhiteboxExtraction):
    if(isWhiteboxExtraction):
        assert isinstance(extractor_functions, list)
        return extract_features_concat(images, extractor_functions)
    else:
        return model.predict(images, verbose=1)



def resizeImage(array, imageSize):
    print("Resizing to {}".format(imageSize))

    resized_tesnor = tf.image.resize_images(array, (imageSize, imageSize),
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    with tf.Session().as_default():
        resized = resized_tesnor.eval()

    return resized