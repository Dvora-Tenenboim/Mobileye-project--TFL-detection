import os
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = 81 * 81 * 3
def validate_data_set(index):
    root = "data_dir\\"
    data_path = os.path.join(root, "train\\data.bin")
    label_path = os.path.join(root, "train\\labels.bin")
    image = np.memmap(data_path, dtype='uint8', mode='r', offset=index * IMAGE_SIZE, shape=(81, 81, 3))
    is_tfl = np.memmap(label_path, dtype='uint8', mode='r', offset=index, shape=(1))
    if is_tfl:
        print("yes tfl")
    else:
        print("not tfl")

    plt.imshow(image)
    plt.show()


def main():
    for i in range(10,20):
        validate_data_set(i)


if __name__ == '__main__':
    main()
