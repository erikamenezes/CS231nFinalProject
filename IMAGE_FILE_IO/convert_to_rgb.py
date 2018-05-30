from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img, array_to_img
import numpy as np
import glob

# Script converting data from jpg to RGB vector

consumer_photos = []
consumer_labels = []
shop_photos = []
shop_labels = []

inputfile = "/Users/ckanitkar/Desktop/img/CLOTHING/Tank_Top"
absolutePath = "/Users/ckanitkar/Desktop/img_npy/"
inputFileNameSplit = inputfile.split('/')
parent1 = inputFileNameSplit[-2]
parent2 = inputFileNameSplit[-1]
parentPath = absolutePath + parent1 + "/" + parent2 + "/"

def load_data(data_dir):
	for filename in glob.iglob(data_dir + '/**/*.jpg', recursive=True):
		#print (filename)
		name = filename.split('/')
		name = name[-4:]
		photo_type = name[-1].split('.')[0].split('_')[0]
		id = name[-2]
		print (photo_type)
		print (id)
		category1 = (name[-4])
		category2 = (name[-3])
		img = load_img(filename)
		img = img.resize((128,128))
		img = img_to_array(img) # (1000,758,3)
		img = img.transpose(2, 0, 1) #(3, 1000, 758)
		if photo_type == "comsumer":
			consumer_photos.append(img)
			consumer_labels.append(id)
		elif photo_type == "shop":
			shop_photos.append(img)
			shop_labels.append(id)

	return np.asarray(consumer_photos, dtype='uint8'), np.asarray(consumer_labels), np.asarray(shop_photos, dtype='uint8'),np.asarray(shop_labels)


consumer_photos, consumer_labels, shop_photos, shop_labels = load_data(inputfile)

print("consumer_photos.shape", consumer_photos.shape)
print("consumer_labels.shape", consumer_labels.shape)
print("shop_photos.shape", shop_photos.shape)
print("shop_labels.shape", shop_labels.shape)

np.save(parentPath +"consumer_photos", consumer_photos)
np.save(parentPath + "consumer_labels", consumer_labels)
np.save(parentPath + "shop_photos", shop_photos)
np.save(parentPath + "shop_labels", shop_labels)