import os
import cv2
import numpy as np
from skimage import color

def _simple_bin(ab_values, n_bins):
    """
    Given a 2 dimensions array (a, b), maps it to an integer
    in {0, 1, ..., 623, 624}. It uses a simple 2D square grid.
    (can be broadcasted)
    """
    a, b = ab_values[..., 0], ab_values[..., 1]
    # a_index = np.minimum(np.maximum(a//10, 0), 24)
    # b_index = np.minimum(np.maximum(b//10, 0), 24)
    # return a_index.astype(np.int) * 25.0 + b_index.astype(np.int)
    m = int(np.sqrt(n_bins))
    bin_size = np.ceil(255/m)
    a_index = np.minimum(np.maximum(a//bin_size, 0), m-1)
    b_index = np.minimum(np.maximum(b//bin_size, 0), m-1)
    return a_index.astype(np.int) * float(m) + b_index.astype(np.int)


def _simple_unbin(bin_integer, n_bins):
    """
    Given an integer, maps it back to the corresponding
    (a, b) values.
    (can be broadcasted)
    """
    list_shape = list(np.shape(bin_integer))
    ab_values = np.zeros(list_shape + [2])

    # a_index = bin_integer // 25
    # b_index = bin_integer % 25
    # a = a_index * 10 + 5
    # b = b_index * 10 + 5
    m = int(np.sqrt(n_bins))
    bin_size = np.ceil(255/m)

    a_index = bin_integer // m
    b_index = bin_integer % m
    a = a_index * bin_size + bin_size / 2
    b = b_index * bin_size + bin_size / 2

    ab_values[..., 0] = a
    ab_values[..., 1] = b
    return ab_values.astype(np.uint8)


def pre_process(image, n_bins):
    """
    rgb_image -> features, labels
    """
    resized_image = cv2.resize(image, (104, 104))
    #lab_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2LAB)
    bgr_image = np.array(resized_image)
    rgb_image = bgr_image[:,:,::-1]
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2LAB)
    luminance = lab_image[:, :, 0]
    ab_channels = lab_image[:, :, 1:]
    binned_ab_channels = _simple_bin(ab_channels, n_bins)

    return luminance, binned_ab_channels


def process_output(luminance, binned_ab_channels, original_shape, n_bins):
    """
    features, labels, shape -> rgb_image
    :param original_shape: np.shape(original_image)
    """
    ab_channels = _simple_unbin(binned_ab_channels, n_bins)
    lab_image = np.stack((luminance,
                          ab_channels[..., 0],
                          ab_channels[..., 1]), axis=2)
    rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    #rgb_image = bgr_image[:,:,::-1]
    original_size_rgb = cv2.resize(rgb_image, original_shape)

    return original_size_rgb

            
def data_generator(tensor_images, n_bins):
    for tensor_image in tensor_images:
        try:
            yield pre_process(tensor_image, n_bins)
        except cv2.error:
            print("/!\\ CV2 ERROR /!\\")
