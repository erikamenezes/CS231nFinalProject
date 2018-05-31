from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy


def GetSiameseNet(input_dim, hidden_dim, final_activation = 'sigmoid'):
	input = Input(shape=(input_dim,))
	output = Dense(hidden_dim, activation='relu')(input)

	if final_activation == 'sigmoid':
		output = Dense(1, activation='sigmoid')(output)
		siamese_net = Model(inputs=input, outputs=output)
		siamese_net.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])	

	elif final_activation == 'svm':
		output = Dense(1, activation='linear')(output)
		siamese_net = Model(inputs=input, outputs=output)
		siamese_net.compile(optimizer='rmsprop', loss='binary_hinge', metrics=['binary_accuracy'])	

	return siamese_net