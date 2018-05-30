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

def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)
#//TODO: figure out how to initialize layer biases in keras.
def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

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
	return [pairs_0, pairs_1], np.asarray(targets)

def load_data(category_directories):
	
	#TODO: split train val test
	consumer_features, consumer_labels, shop_features, shop_labels = None, None, None, None
	
	for i, category_dir in enumerate(category_directories):
		if consumer_features is None:
			consumer_features = np.load(category_dir + "consumer_ResNet50_features.npy")
		else:
			consumer_features = np.vstack((consumer_features, np.load(filename)))

		if consumer_labels is None:
			consumer_labels = np.load(category_dir + "consumer_labels.npy")
				
		else:
			consumer_labels = np.vstack((consumer_labels, np.load(filename)))

		if shop_features is None:
			shop_features = np.load(category_dir + "shop_ResNet50_features.npy")
		else:
			shop_features = np.vstack((shop_features, np.load(filename)))

		if shop_labels is None:
			shop_labels = np.load(category_dir + "shop_labels.npy")
		else:
			shop_labels = np.vstack((shop_labels, np.load(filename)))

		
	return consumer_features, consumer_labels, shop_features, shop_labels	


category_directories = [ './img_npy_new/'] # './img_npy/SKIRT/Dress-Skirt/' ] #, './img_npy/TROUSER/' , './img_npy/DRESSES/' ,'./img_npy/CLOTHING/']

consumer_features, consumer_labels, shop_features, shop_labels = load_data(category_directories)
print (np.squeeze(consumer_features).shape)
print (consumer_labels.shape)
print (np.squeeze(shop_features).shape)
print (shop_labels.shape)

pairs, targets = load_pairs(np.squeeze(consumer_features), consumer_labels, np.squeeze(shop_features), shop_labels)


# input_shape = (105, 105, 1)
# left_input = Input(input_shape)
# right_input = Input(input_shape)
# #build convnet to use in each siamese 'leg'
# convnet = Sequential()
# convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,
#                    kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
# convnet.add(MaxPooling2D())
# convnet.add(Conv2D(128,(7,7),activation='relu',
#                    kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
# convnet.add(MaxPooling2D())
# convnet.add(Conv2D(128,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
# convnet.add(MaxPooling2D())
# convnet.add(Conv2D(256,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
# convnet.add(Flatten())
# convnet.add(Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init))

# #call the convnet Sequential model on each of the input tensors so params will be shared
# encoded_l = convnet(left_input)
# encoded_r = convnet(right_input)
# #layer to merge two encoded inputs with the l1 distance between them
# L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
# #call this layer on list of two input tensors.
# L1_distance = L1_layer([encoded_l, encoded_r])
# prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(L1_distance)
# siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

# optimizer = Adam(0.00006)
# #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
# siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)

# siamese_net.count_params()