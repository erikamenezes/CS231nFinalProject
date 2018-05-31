from common.extract_features import *
from Experiment1_WhiteBoxFeatures.feature_extraction_methods import hog_feature
from keras.applications import ResNet50

directory = "/Users/ckanitkar/Desktop/img_npy_final_test/"

extract_features_white_box(directory, includedCategories=["Skirt"], extractor_functions=[hog_feature])


#extract_features_pre_trained(directory, includedCategories=["Skirt"], imageReshape = 224, model = ResNet50(), layer_name="avg_pool")