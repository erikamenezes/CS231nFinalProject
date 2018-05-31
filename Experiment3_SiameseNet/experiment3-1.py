import numpy as np
from SiameseNetwork import GetSiameseNet
from SiameseDataUtil import LoadData, ComputeDistance

DATA_DIR = '.././img_npy_final_features_only/CLOTHING/LowerBody/'

consumer_features = np.load(DATA_DIR + 'consumer_ResNet50_features.npy')
consumer_labels = np.load(DATA_DIR + 'consumer_labels.npy')
shop_features = np.load(DATA_DIR + 'shop_ResNet50_features.npy')
shop_labels = np.load(DATA_DIR + 'shop_labels.npy')

print (consumer_features.shape)
print (consumer_labels.shape)
print (shop_features.shape)
print (shop_labels.shape)

pair, target = LoadData(consumer_features, consumer_labels, shop_features, shop_labels, pairs=True)
distance = ComputeDistance(pair, pairs=True)

input_dim = consumer_features.shape[-1]
hidden_dim = 2048
model = GetSiameseNet(input_dim,hidden_dim)

model.fit(distance, target, validation_split=.2)





