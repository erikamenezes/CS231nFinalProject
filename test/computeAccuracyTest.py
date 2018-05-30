import unittest
import numpy as np
from common.computeAccuracy import computeAccuracy

class TestComputeAccuracy(unittest.TestCase):

    def setUp(self):
        self.distances = np.array([
            [4, 20, 2, 12],
            [36, 300, 232, 999],
            [44, 2000, 25, 23]
        ])
        self.shop_labels = np.array(['id_1', 'id_2', 'id_3', 'id_4'])
        self.consumer_labels = np.array(['id_2', 'id_1', 'id_3'])

    def testAccuracyKEquals1(self):
        correct, total, accuracy = computeAccuracy(self.distances, self.consumer_labels, self.shop_labels, k = 1)
        self.assertEqual((correct, total, accuracy), (1, 3, 1/3))

    def testAccuracyKEquals2(self):
        correct, total, accuracy = computeAccuracy(self.distances, self.consumer_labels, self.shop_labels, k = 2)
        self.assertEqual((correct, total, accuracy), (2, 3, 2/3))

    def testAccuracyKEquals4(self):
        correct, total, accuracy = computeAccuracy(self.distances, self.consumer_labels, self.shop_labels, k=4)
        self.assertEqual((correct, total, accuracy), (3, 3, 1))


if __name__ == '__main__':
    unittest.main()