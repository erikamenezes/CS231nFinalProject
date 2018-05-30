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



def extract_features(DIRECTORY_PATH, base_model, layer_name):
    model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)



#TYPES = ["Dress", "Skirt", "UpperBody", "LowerBody"]
TYPES = []
DIRECTORY_PATH = "/Users/ckanitkar/Desktop/img_npy_final/"


base_model = ResNet50()
#model = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)
model = Model(inputs=base_model.input, outputs=base_model.output)
print(model.summary())

for filename in glob.iglob(DIRECTORY_PATH + '/**/*_photos.npy', recursive=True):
    array = np.load(filename).transpose([0,2,3,1])
    name = filename.split('/')
    photo_type = "consumer" if "consumer" in name[-1] else "shop"
    category2 = name[-2]
    category1 = name[-3]
    if category2 in TYPES:
        newFileName = DIRECTORY_PATH + category1 + "/" + category2 + "/" + photo_type + "_ResNet50_features"
        print(filename)
        print (newFileName)
        print (array.shape)
        #plt.imshow(array[0])
        #plt.show()
        #esized = image.img_to_array(image.array_to_img(array)).resize((224,224))


        resized_tesnor = tf.image.resize_images(array, (224, 224), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        with tf.Session().as_default():
            resized = resized_tesnor.eval()

        print(resized.shape)
        #print(resized.dtype)
        #
        #print(resized.shape)

        print("Extracting features")
        output = model.predict(resized, verbose=1)
        print(output.shape)
        np.save(newFileName, output)





