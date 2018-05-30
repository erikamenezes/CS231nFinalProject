import numpy as np
import scipy.spatial.distance as scidist


def computeDistances(consumer_features, shop_features, metric='euclidean', model = None):
    assert isinstance(consumer_features, np.ndarray), 'Consumer features must be an numpy array of size n * d'
    assert isinstance(shop_features, np.ndarray), 'Shop features must be a numpy array of size m * d'
    assert consumer_features.shape[1] == shop_features.shape[1], 'Distances must be a numpy array of consumer * shop'

    if model is not None:
      # TODO : Fill this out
      pass
    else:
      return scidist.cdist(consumer_features, shop_features, metric=metric)