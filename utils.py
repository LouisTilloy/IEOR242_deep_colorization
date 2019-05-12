import cv2
import os
import pickle
import numpy as np

DEFAULT_BIN_NUMBER = 13


def _simple_bin(ab_values, n_1d_bins):
    """
    Given a 2 dimensions array (a, b), maps it to an integer
    in {0, 1, ..., n_1d_bins^2 - 1}. It uses a simple 2D square grid.
    (can be broadcasted)
    """
    a, b = ab_values[..., 0].astype(int), ab_values[..., 1].astype(int)
    a_index = np.minimum(np.maximum((a * n_1d_bins)//255, 0), n_1d_bins - 1)
    b_index = np.minimum(np.maximum((b * n_1d_bins)//255, 0), n_1d_bins - 1)
    return a_index.astype(np.int) * n_1d_bins + b_index.astype(np.int)


def _simple_unbin(bin_integer, n_1d_bins):
    """
    Given an integer, maps it back to the corresponding
    (a, b) values.
    (can be broadcasted)
    """
    list_shape = list(np.shape(bin_integer))
    ab_values = np.zeros(list_shape + [2])

    a_index = bin_integer // n_1d_bins
    b_index = bin_integer % n_1d_bins
    a = ((a_index + 0.5) * 255)//n_1d_bins  # +0.5 to center the bins
    b = ((b_index + 0.5) * 255)//n_1d_bins

    ab_values[..., 0] = a
    ab_values[..., 1] = b
    return ab_values.astype(np.uint8)


def pre_process(image, resolution, n_1d_bins=DEFAULT_BIN_NUMBER):
    """
    rgb_image -> features, labels
    :param image: np.array
    :param resolution: (int, int)
    :param n_1d_bins: int
    """
    resized_image = cv2.resize(image, (resolution, resolution))
    lab_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2LAB)
    luminance = lab_image[:, :, 0].astype(int) - 128  # Center luminance
    ab_channels = lab_image[:, :, 1:]
    binned_ab_channels = _simple_bin(ab_channels, n_1d_bins)

    return luminance, binned_ab_channels


def process_output(luminance, binned_ab_channels, original_shape,
                   n_1d_bins=DEFAULT_BIN_NUMBER):
    """
    features, labels, shape -> rgb_image
    :param original_shape: np.shape(original_image)
    """
    ab_channels = _simple_unbin(binned_ab_channels, n_1d_bins)
    lab_image = np.stack(((luminance + 128).astype(np.uint8),
                          ab_channels[..., 0],
                          ab_channels[..., 1]), axis=2)
    rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    original_size_rgb = cv2.resize(rgb_image, original_shape[:2][::-1])

    return original_size_rgb


def data_generator(imagenet_folder, resolution):
    """
    Given the imagenet folder, returns a generator
    that goes through all the images (once) and
    pre process them.
    """
    # Step 0: get imagenet images path
    imagenet_paths = [imagenet_folder + "/" + name for name in next(os.walk(imagenet_folder))[2]]
    
    # Step 1: go randomly through the images
    shuffled_paths = np.array(imagenet_paths)
    np.random.shuffle(shuffled_paths)
    for path in shuffled_paths:
        bgr_image = cv2.imread(path)
        rgb_image = bgr_image[:, :, ::-1]
        try:
            yield pre_process(rgb_image, resolution)
        except cv2.error:
            print("/!\\ CV2 ERROR /!\\")


def cifar_10_train_data_generator(cifar_folder, resolution):
    """
    Given the folder where cifar-10 is stored,
    returns a generator over all training images.
    """
    # Step 0: retrieve cifar-10 batches
    cifar_batches = []
    for i in range(1, 6):
        with open(cifar_folder + "/data_batch_{}".format(i), "rb") as file:
            cifar_batches.append(pickle.load(file, encoding="latin"))

    # Step 1: construct list of images
    list_images = []
    for batch in cifar_batches:
        for raw_image in batch["data"]:
            image = np.transpose(raw_image.reshape((3, 32, 32)), [1, 2, 0])
            list_images.append(image)

    # Step 2: Go through the list randomly
    shuffled_images = np.array(list_images)
    np.random.shuffle(shuffled_images)
    for image in shuffled_images:
        yield pre_process(image, resolution)
