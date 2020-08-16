try:
    print("Elementary imports: ")
    import os
    import json
    import glob
    import argparse
    #import cv2

    print("numpy/scipy imports:")
    import numpy as np
    from scipy import signal as sg
    from scipy import misc
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter

    print("PIL imports:")
    from PIL import Image

    print("matplotlib imports:")
    import matplotlib.pyplot as plt

except ImportError:
    print("Need to fix the installation")
    raise

print("All imports okay. Yay!")


def find_tfl_lights(c_image: np.ndarray, **kwargs):
#def find_tfl_lights(image_path, **kwargs):
    # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", help="path to the image file")
    # ap.add_argument("-r", "--radius", type=int,
    #                 help="radius of Gaussian blur; must be odd")
    # args = vars(ap.parse_args())
    # args["image"] = image_path
    # args["radius"] = 11
    # load the image and convert it to grayscale
    # image = cv2.imread(args["image"])
    # orig = image.copy()
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # perform a naive attempt to find the (x, y) coordinates of
    # the area of the image with the largest intensity value
    # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    # cv2.circle(image, maxLoc, 5, (255, 0, 0), 2)
    # display the results of the naive attempt
    # cv2.imshow("Naive", image)
    # apply a Gaussian blur to the image then find the brightest
    # region
    # gray = cv2.GaussianBlur(gray, (args["radius"], args["radius"]), 0)
    # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    # image = orig.copy()
    # cv2.circle(image, maxLoc, args["radius"], (255, 0, 0), 2)
    # display the results of our newly improved method
    # cv2.imshow("Robust", image)
    # cv2.waitKey(0)

    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """

    scharr = np.array([[0,0,0,0,0],
                       [0,1,1,1,0],
                       [0,1,1,1,0],
                       [0,1,1,1,0],
                       [0,0,0,0,0]])

    c_image = c_image/255
    grad = sg.convolve2d(c_image[:, :, 1], scharr, boundary='symm', mode='same')
    plt.imshow(grad)

    x = np.arange(-100, 100, 20) + c_image.shape[1] / 2
    y_red = [c_image.shape[0] / 2 - 120] * len(x)
    y_green = [c_image.shape[0] / 2 - 100] * len(x)
    return x, y_red, x, y_green


def show_image(image, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    pictures_path = 'pictures'
    flist = glob.glob(os.path.join(pictures_path, '*_leftImg8bit.png'))
    for image_path in flist:
        image = np.array(Image.open(image_path))
        red_x, red_y, green_x, green_y = find_tfl_lights(image)
        # show_image(image)

    plt.show(block=True)


if __name__ == '__main__':
    main()
