import cv2.cv2 as cv2
import glob
from copy import deepcopy
from scipy.signal import convolve
import numpy as np


def add_votes(point1, point2, xi1, xi2, votes_im):
    mid_point = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)

    intersection_point = (
        (point1[1] - point2[1] - (point1[0] * xi1) + (point2[0] * xi2)) / (xi2 - xi1),
        (xi1 * xi2 * (point2[0] - point1[0]) - (point2[1] * xi1) + (point1[1] * xi2))/(xi2 - xi1))

    x_values = np.arange(intersection_point[0], mid_point[0])
    y_values = np.arange(intersection_point[1], mid_point[1])

    # TODO: add votes to the matrix


def getGradient(edge_im):
    sobel_x_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])

    sobel_y_kernel = np.array([[1, -2, 1],
                               [0, 0, 0],
                               [-1, 2, -1]])

    sobel_x = convolve(sobel_x_kernel, edge_im, method='fft')
    sobel_y = convolve(sobel_y_kernel, edge_im, method='fft')
    return np.sqrt(np.power(sobel_x, 2) + np.power(sobel_y, 2))


if __name__ == "__main__":
    i_types = ('*.png', '*.webp', '*.jpg', '*.jpeg')  # tuple of all images types in the folder
    images_names = []
    for i_type in i_types:
        images_names.extend(glob.glob(i_type))  # each name includes the file extension

    temp_images = [cv2.imread(i_name) for i_name in images_names]
    images = []

    for i in range(len(temp_images)):
        images.append((images_names[i], deepcopy(temp_images[i])))

    for im_name, im_content in images:
        im = cv2.cvtColor(im_content, cv2.COLOR_RGB2GRAY)

        if im_name == "5da02f8f443e6-brondby-haveby-allotment-gardens-copenhagen-denmark-7.jpg.png":
            edge_map = cv2.Canny(im, 180, 240)
        elif im_name == "72384675-very-long-truck-trailer-for-exceptional-transport-with-many-sturdy-tires.webp":
            edge_map = cv2.Canny(im, 50, 255)
        elif im_name == "1271488188_2077d21f46_b.jpg":
            edge_map = cv2.Canny(im, 70, 150)
        elif im_name == "alvtd333_alvin_template_small_ellipse.png":
            edge_map = cv2.Canny(im, 100, 200)
        elif im_name == "gettyimages-1212455495-612x612.jpg":
            edge_map = cv2.Canny(im, 160, 180)
        elif im_name == "hammer-tissot-big.jpg":
            edge_map = cv2.Canny(im, 100, 200)
        elif im_name == "Headline-Pic.jpg":
            edge_map = cv2.Canny(im, 100, 200)
        elif im_name == "images.jpg":
            edge_map = cv2.Canny(im, 100, 200)
        elif im_name == "nEKGD2wNiwqrTOc63kiWZT7b4.png":
            edge_map = cv2.Canny(im, 150, 200)
        elif im_name == "s-l400.jpg":
            edge_map = cv2.Canny(im, 100, 180)
        elif im_name == "sewage-treatment-plant-wastewater-treatment-water-use-filtration-effluent-and-waste-water" \
                        "-industrial-solutions-for-sewerage-water-treatment-and-rec-2H4RX20.jpg":
            edge_map = cv2.Canny(im, 100, 200)
        else:  # "Traitement-dimage-drone-.jpeg":
            edge_map = cv2.Canny(im, 100, 200)

        gradient_map = getGradient(edge_map)

        cv2.imshow('image', im)
        cv2.imshow('edge map', edge_map)
        cv2.imshow('gradient', gradient_map)

        cv2.waitKey(0)
