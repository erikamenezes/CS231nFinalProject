import numpy as np
import glob


# Script to consolidate data from sub categories into larger categories


directoryPath = "/Users/ckanitkar/Desktop/img_npy_final/"
type = 'shop'


consolidated_images = np.array([])
consolidated_labels = np.array([])


def concat(filename):
    array = np.load(filename)
    name = filename.split('/')
    photo_type = "consumer" if "consumer" in name[-1] else "shop"

    print(filename)
    print (newFileName)
    print (array.shape)
    consolidated_images


for filename in glob.iglob(directoryPath + '/**/*_photos.npy', recursive=True):
    concat(filename)
for filename in glob.iglob(directoryPath + '/**/*_labels.npy', recursive=True):
    concat(filename)

