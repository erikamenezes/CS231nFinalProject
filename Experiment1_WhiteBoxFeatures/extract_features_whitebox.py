import glob
from .feature_extraction_methods import *
from .concatenate_extracted_features_with_feature_functions import *

'''
This script extracts whitebox features for all photos in DIRECTORY_PATH
'''

DIRECTORY_PATH = "/Users/ckanitkar/Desktop/img_npy_final/"

TYPES = ["UpperBody", "LowerBody"]

for filename in glob.iglob(DIRECTORY_PATH + '/**/*_photos.npy', recursive=True):
    array = np.load(filename).transpose([0,2,3,1])
    name = filename.split('/')
    photo_type = "consumer" if "consumer" in name[-1] else "shop"
    category2 = name[-2]
    category1 = name[-3]
    if category2 in TYPES:
        newFileName = DIRECTORY_PATH + category1 + "/" + category2 + "/" + photo_type + "_features"
        print(filename)
        print (newFileName)
        print (array.shape)
        output = extract_features(array, [hog_feature, color_histogram_hsv])
        print (output.shape)
        assert (array.shape[0] == output.shape[0])
        np.save(newFileName, output)

