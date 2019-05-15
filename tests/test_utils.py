import unittest
import numpy as np
from utils import _simple_bin, _simple_unbin, pre_process, process_output


class TestUtils(unittest.TestCase):

    def test_simple_bin_simple(self):
        # 4 1d_bins: (0, 63.75, 127.5, 191.25, 255)
        n_1d_bins = 4

        self.assertEqual(_simple_bin(np.array((50, 50)), n_1d_bins), 0)
        self.assertEqual(_simple_bin(np.array((100, 50)), n_1d_bins), 4)
        self.assertEqual(_simple_bin(np.array((100, 100)), n_1d_bins), 5)
        self.assertEqual(_simple_bin(np.array((50, 250)), n_1d_bins), 3)
        self.assertEqual(_simple_bin(np.array((255, 255)), n_1d_bins), 15)

    def test_simple_bin_broadcast(self):
        # 5 1d_bins: (0, 51, 102, 153, 204, 255)
        n_1d_bins = 5

        ab_values = np.array(
            [[[50, 50], [100, 100], [150, 210]],
             [[100, 0], [255, 255], [0, 0]]], dtype=np.uint8
        )
        expected_bins = np.array(
            [[0, 6, 14],
             [5, 24, 0]]
        )
        np.testing.assert_equal(_simple_bin(ab_values, n_1d_bins), expected_bins)

    def test_unbin_simple(self):
        # 3 1d_bins: (0, 85, 170, 255)
        # bin centers (integers): (42, 127, 212)
        n_1d_bins = 3

        np.testing.assert_equal(_simple_unbin(5, n_1d_bins), np.array([127, 212]))
        np.testing.assert_equal(_simple_unbin(0, n_1d_bins), np.array([42, 42]))
        np.testing.assert_equal(_simple_unbin(8, n_1d_bins), np.array([212, 212]))
        np.testing.assert_equal(_simple_unbin(7, n_1d_bins), np.array([212, 127]))

    def test_unbin_broadcast(self):
        # 10 1d_bins: (0, 25.5, 51, 76.5, 102, 127.5, 153, 178.5, 204, 229.5, 255)
        # bin centers (integers): (12, 38, 63, 89, 114, 140, 165, 191, 216, 242)
        n_1d_bins = 10

        bin_values = np.array(
            [[20, 23, 54],
             [99, 0, 82]]
        )

        expected_ab_values = np.array(
            [[[63, 12], [63, 89], [140, 114]],
             [[242, 242], [12, 12], [216, 63]]]
        )

        np.testing.assert_equal(_simple_unbin(bin_values, n_1d_bins),
                                expected_ab_values)

    def test_pre_process(self):
        # 3 1d_bins: (0, 85, 170, 255)
        n_1d_bins = 3
        resolution = 2

        image = np.array(
            [[[50, 50, 50], [0, 25, 100]],
             [[0, 50, 100], [100, 150, 200]]], dtype=np.uint8
        )

        # note the (width, height, 1) size instead of (width, height)
        expected_luminance = np.array([[[-75], [-94]],
                                       [[-75],  [26]]], dtype=int)

        expected_ab_bins = np.array([[[4], [3]],
                                     [[4], [4]]], dtype=int)

        luminance, ab_bins = pre_process(image, resolution, n_1d_bins)

        np.testing.assert_equal(luminance, expected_luminance)
        np.testing.assert_equal(ab_bins, expected_ab_bins)
        self.assertEqual(luminance.dtype, expected_luminance.dtype)
        self.assertEqual(ab_bins.dtype, expected_ab_bins.dtype)

    def test_process_output(self):
        # 3 1d_bins: (0, 85, 170, 255)
        n_1d_bins = 3
        original_shape = (2, 2)

        # note the (width, height, 1) size instead of (width, height)
        luminance = np.array([[[-75], [-94]],
                              [[-75],  [26]]], dtype=int)
        binned_ab_channels = np.array([[4, 3],
                                       [4, 4]], dtype=int)

        expected_output = np.array([[[48,  50,  51], [0,  49, 163]],
                                    [[48,  50,  51], [143, 146, 147]]],
                                   dtype=np.uint8)

        output = process_output(luminance, binned_ab_channels,
                                original_shape, n_1d_bins)
        np.testing.assert_equal(output, expected_output)


