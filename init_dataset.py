import os
from os.path import join
from PIL import Image
import numpy as np
import random
from pathlib import Path

IMAGES_ROOT = Path('cityscapes')
TRAIN_DATA_ROOT = Path("data_dir//train")
VAL_DATA_ROOT = Path("data_dir//val")
CROPPING_SIZE = 81
TFL_LABEL = 19


def generate_random_coordinate(image):
    image_length = image.shape
    return random.randint(0, image_length[0] - 1), random.randint(0, image_length[1] - 1)


def get_cropping_edges(image, x, y, size):
    image_shape = image.shape
    left = max(0, x - size // 2)
    right = left + size
    if right > image_shape[0]-1:
        right = image_shape[0]-1
        left = right - size
    ceiling = max(y - size // 2, 0)
    bottom = ceiling + size
    if bottom > image_shape[1]-1:
        bottom = image_shape[1] - 1
        ceiling = bottom - size
    return left, right, ceiling, bottom


def crop_image(image, x, y, size):
    left, right, ceiling, bottom = get_cropping_edges(image, x, y, size)
    cropping_image = image[left:right, ceiling:bottom]
    return cropping_image


def write_to_binary_file(file_path, character):
    with open(file_path, mode='ab') as file:
        np.array(character, dtype=np.uint8).tofile(file)


def get_image(image_path):
    return np.array(Image.open(image_path)).astype(np.uint8)


def coordinate_is_ntfl(gt_image, x_random, y_random, CROPPING_SIZE):
    left, right, ceiling, bottom = get_cropping_edges(gt_image, x_random, y_random, CROPPING_SIZE)
    if gt_image[x_random][y_random] != TFL_LABEL and gt_image[left][y_random] != TFL_LABEL and gt_image[right][
        y_random] != TFL_LABEL and gt_image[x_random][ceiling] != TFL_LABEL and gt_image[x_random][bottom] != TFL_LABEL:
        return True
    return False


def get_cropping_coordinate(gt_image):
    # TODO return few tfl
    tfl_coordinates = np.where(gt_image == 19)
    if len(tfl_coordinates[0]) > 0:
        x_tfl = tfl_coordinates[0][5]
        y_tfl = tfl_coordinates[1][5]
    else:
        return [], [], [], []
    while True:
        x_random, y_random = generate_random_coordinate(gt_image)
        if coordinate_is_ntfl(gt_image, x_random, y_random, CROPPING_SIZE):
            break
    return [x_tfl], [y_tfl], [x_random], [y_random]


def treat_data(image, gt_image, x_tfl, y_tfl, x_ntfl, y_ntfl, root):
    for i, x in enumerate(x_tfl):
        tfl_image = crop_image(image, x, y_tfl[i], CROPPING_SIZE)
        write_to_binary_file(join(root, "data.bin"), tfl_image)
        write_to_binary_file(join(root, "labels.bin"), np.array([1]))
    for i, x in enumerate(x_ntfl):
        ntfl_image = crop_image(image, x, y_ntfl[i], CROPPING_SIZE)
        write_to_binary_file(join(root, "data.bin"), ntfl_image)
        write_to_binary_file(join(root, "labels.bin"), np.array([0]))


def init_dataset():
    for root, dirs, images in os.walk(join(IMAGES_ROOT, "leftImg8bit")):
        for image in images:
            image_path = join(root, image)
            gt_root = root.replace("leftImg8bit", "gtFine")
            gt_image_path = join(gt_root, image.replace("leftImg8bit.png", "gtFine_labelIds.png"))
            gt_image = get_image(gt_image_path)
            x_tfl, y_tfl, x_ntfl, y_ntfl = get_cropping_coordinate(gt_image)
            image = get_image(image_path)
            if "train" in image_path:
                treat_data(image, gt_image, x_tfl, y_tfl, x_ntfl, y_ntfl, TRAIN_DATA_ROOT)
            else:
                treat_data(image, gt_image, x_tfl, y_tfl, x_ntfl, y_ntfl, VAL_DATA_ROOT)


def main():
    init_dataset()


if __name__ == '__main__':
    main()
