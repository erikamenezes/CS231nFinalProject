from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
import pickle
# import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.utils import shuffle

def load_pairs(consumer_features, consumer_labels, shop_features, shop_labels):
	N, dim = consumer_features.shape
	#pairs=[np.zeros((N, dim)) for i in range(2)]
	pairs_0 = []
	pairs_1 = []
	targets= [] #np.zeros((N,))

	i = 0
	for j,c in enumerate(consumer_features):
		# print ("i:", i)
		# print ("j:", j)
		shop_images_idx = np.where(shop_labels == consumer_labels[j])
		for s in shop_images_idx[0]:
			# print ("s:", s)
			#pairs[0][i,:] = c
			pairs_0.append(c)
			#print (pairs[1].shape)
			# print (shop_features[s].shape)
			#pairs[1][i,:] = shop_features[s]
			pairs_1.append(shop_features[s])
			targets.append(1)
			i +=1

		shop_images_idx_neg = np.where(shop_labels != consumer_labels[j])[0][0:10]
		# print ("shop_images_idx_neg", list(shop_images_idx_neg))
		# add neagtive samples
		for s in shop_images_idx_neg:
			#pairs[0][i,:] = c
			pairs_0.append(c)
			#pairs[1][i,:] = shop_features[s]
			pairs_1.append(shop_features[s])
			targets.append(0)
			i+=1

	print (np.asarray(pairs_0).shape)
	print (np.asarray(pairs_1).shape)
	return [np.asarray(pairs_0), np.asarray(pairs_1)], np.asarray(targets)


def convertToDistance(pairs):
	consumer = pairs[0]
	shop = pairs[1]
	assert consumer.shape == shop.shape

	difference = consumer - shop

	return np.abs(difference)



# def load_data(category_directories):
#
# 	#TODO: split train val test
# 	consumer_features, consumer_labels, shop_features, shop_labels = None, None, None, None
#
# 	for i, category_dir in enumerate(category_directories):
# 		if consumer_features is None:
# 			consumer_features = np.load(category_dir + "consumer_ResNet50_features.npy")
# 		else:
# 			consumer_features = np.vstack((consumer_features, np.load(filename)))
#
# 		if consumer_labels is None:
# 			consumer_labels = np.load(category_dir + "consumer_labels.npy")
#
# 		else:
# 			consumer_labels = np.vstack((consumer_labels, np.load(filename)))
#
# 		if shop_features is None:
# 			shop_features = np.load(category_dir + "shop_ResNet50_features.npy")
# 		else:
# 			shop_features = np.vstack((shop_features, np.load(filename)))
#
# 		if shop_labels is None:
# 			shop_labels = np.load(category_dir + "shop_labels.npy")
# 		else:
# 			shop_labels = np.vstack((shop_labels, np.load(filename)))
#
#
# 	return consumer_features, consumer_labels, shop_features, shop_labels


#category_directories = [ './img_npy_new/'] # './img_npy/SKIRT/Dress-Skirt/' ] #, './img_npy/TROUSER/' , './img_npy/DRESSES/' ,'./img_npy/CLOTHING/']


path = "/Users/ckanitkar/Desktop/img_npy_final_features_only/CLOTHING/LowerBody/"

consumer_features = np.load(path + "consumer_ResNet50_features.npy")
shop_features = np.load(path + "shop_ResNet50_features.npy")
consumer_labels = np.load(path + "consumer_labels.npy")
shop_labels = np.load(path + "shop_labels.npy")

print(consumer_features.shape)
print (consumer_labels.shape)
print (shop_features.shape)
print (shop_labels.shape)

pairs, targets = load_pairs(consumer_features, consumer_labels, shop_features, shop_labels)

print(pairs[0].shape)
print(pairs[1].shape)
print(targets.shape)


distance = convertToDistance(pairs)
print(distance.shape)

# input_shape = (105, 105, 1)
# left_input = Input(input_shape)
# right_input = Input(input_shape)


# #call the convnet Sequential model on each of the input tensors so params will be shared
# encoded_l = convnet(left_input)
# encoded_r = convnet(right_input)
# #layer to merge two encoded inputs with the l1 distance between them
# #call this layer on list of two input tensors.


input = Input(shape=(2048,))
output = Dense(2048, activation='relu')(input)
output = Dense(1, activation='sigmoid')(output)
siamese_net = Model(inputs=input, outputs=output)
siamese_net.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
siamese_net.fit(distance, targets, validation_split=.1)

# prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(L1_distance)
# siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

# optimizer = Adam(0.00006)
# #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
# siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)

# siamese_net.count_params()




