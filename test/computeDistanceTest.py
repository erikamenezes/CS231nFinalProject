import unittest
import numpy as np
from common.computeDistanceBetweenExtractedFeatures import *

class TestComputeDistance(unittest.TestCase):

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
    def testComputeManhattanDistance(self):
        distances = computeDistances(self.consumer_features, self.shop_features, metric='cityblock')
        expected_distances = np.array([
            [0, 16, 8, 33],
            [16, 0, 14, 29],
            [8, 14, 0, 31]
        ])
        np.testing.assert_array_equal(distances, expected_distances)

    def testComputeDistanceWithTrainedModel(self):
        # TODO: Fill this out
        pass

if __name__ == '__main__':
    unittest.main()