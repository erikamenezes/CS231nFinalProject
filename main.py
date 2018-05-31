from common.extract_features import extract_features_white_box
from Experiment1_WhiteBoxFeatures.feature_extraction_methods import hog_feature

directory = "/Users/ckanitkar/Desktop/img_npy_final_test/"

extract_features_white_box(directory, includedCategories=["Skirt"], extractor_functions=[hog_feature], imageReshape=224)