import unittest
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from common.computeDistanceBetweenExtractedFeatures import *

class TestComputeDistance(unittest.TestCase):

    def createToyModel(self, dims):
        input = Input(shape=(dims,))
        output = Dense(1, bias_initializer='zeros', kernel_initializer='ones')(input)
        return Model(inputs=input, output = output)

    def setUp(self):
        self.consumer_features = np.array([
            [1, 3, 5, 2],
            [5, 10, 2, 4],
            [1, 1, 1, 4]
        ])
        self.shop_features = np.array([
            [1, 3, 5, 2],
            [5, 10, 2, 4],
            [1, 1, 1, 4],
            [12, 4, 2, 20]
        ])
        self.model = self.createToyModel(self.consumer_features.shape[1])

    def testComputeManhattanDistance(self):
        distances = computeDistances(self.consumer_features, self.shop_features, metric='cityblock')
        expected_distances = np.array([
            [0, 16, 8, 33],
            [16, 0, 14, 29],
            [8, 14, 0, 31]
        ])
        np.testing.assert_array_equal(distances, expected_distances)

    def testComputeDistanceWithTrainedModelCityBlock(self):
        # Toy model predicts distance to simply be sum of features passed in.

        expected_distances = -1 * np.array([
            [0, 16, 8, 33],
            [16, 0, 14, 29],
            [8, 14, 0, 31]
        ])
        output = computeDistances(self.consumer_features, self.shop_features, metric='cityblock', model=self.model, batchSize=2)

        np.testing.assert_array_equal(output, expected_distances)

    def testComputeDistanceWithTrainedModelEuclidean(self):
        # Toy model predicts distance to simply be sum of features passed in.

        expected_distances = -1 * np.array([
            [0, 78, 24, 455],
            [78, 0, 98, 341],
            [24, 98, 0, 387]
        ])
        output = computeDistances(self.consumer_features, self.shop_features, metric='euclidean', model=self.model, batchSize=2)

        np.testing.assert_array_equal(output, expected_distances)


if __name__ == '__main__':
    unittest.main()