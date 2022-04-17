import cv2.cv2 as cv2
import glob
from copy import deepcopy
from scipy.signal import convolve
import numpy as np


def draw_ellipse(ellipse_centers, target_im, ellipse_color):
    """
    Finds the parameters for the ellipses in the image and draws them.
    :param ellipse_centers: A list of tuples containing all ellipses' centers in the image.
    :param ellipse_color: The color to use in drawing the ellipses.
    :param target_im: The image in RGB to draw the detected ellipses on.
    :return: None
    """
    # TODO: find the remaining ellipse parameters (B, C, D) then draw the ellipse using cv2.ellipse)
    pass


def get_ellipse_centers(votes_im, thresh_param):
    """
    Detects ellipses' centers based on votes.
    :param votes_im: Matrix of votes for ellipses' centers in the image.
    :param thresh_param: Threshold parameter used to detect the centers.
    :return: A list of tuples where each tuple contains the row and col of the center.
    """
    el_centers = []
    for row in range(votes_im.shape[0]):
        for col in range(votes_im.shape[1]):
            if votes_im[row][col] >= thresh_param:
                el_centers.append((row, col))
    return el_centers


def add_votes(point_1, point_2, xi1, xi2, votes_im,edge_map):
    """
    Adds votes for finding the ellipses' centers based on the algorithm in paper for finding ellipses' centers.
    :param point_1: The first point in the algorithm (the point P).
    :param point_2: The last point in the algorithm (the point Q).
    :param xi1: The value of xi1 in the algorithm
    :param xi2: The value of xi2 in the algorithm.
    :param votes_im: The matrix containing all votes so far.
    :return: None
    """

    if xi2 != xi1:
        mid_point = ((point_1[0] + point_2[0]) // 2, (point_1[1] + point_2[1]) // 2) #M

        intersection_point = (
            (point_1[1] - point_2[1] - (point_1[0] * xi1) + (point_2[0] * xi2)) / (xi2 - xi1),
            (xi1 * xi2 * (point_2[0] - point_1[0]) - (point_2[1] * xi1) + (point_1[1] * xi2))/(xi2 - xi1)) #T

        # x_values = np.arange(intersection_point[0], mid_point[0])
        # y_values = np.arange(intersection_point[1], mid_point[1])

        temp = np.zeros(votes_im.shape)

        lines = cv2.HoughLines(edge_map, 1, np.pi / 180, 200)
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(temp, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow('full 255', temp)
        temp /= 255
        cv2.imshow('post processing', temp)



        # for x,y in
        # TODO: add votes to the matrix


def getGradient(edge_im):
    """
    Calculates the gradient map based on an edge map.
    :param edge_im: The edge map of the image.
    :return: The gradient map calculated based on Sobel kernel.
    """
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
            edge_map = cv2.Canny(im, 80, 200)
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

        ellipse_center_votes = np.zeros((im.shape[0], im.shape[1]))

        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                point1 = (i, j)

                for i2 in range(im.shape[0]):
                    for j2 in range(im.shape[1]):
                        point2 = (i2, j2)
                        add_votes(point1, point2, gradient_map[point1[0]][point1[1]],
                                  gradient_map[point2[0]][point2[1]], ellipse_center_votes,edge_map)

        ellipse_center_votes /= 2   # we counted the votes twice
        thresh = 10

        centers = get_ellipse_centers(ellipse_center_votes, thresh)

        # bonus:
        draw_ellipse(centers, im_content, (255, 255, 0))

        cv2.imshow('image', im)
        cv2.imshow('edge map', edge_map)
        cv2.imshow('gradient', gradient_map)
        cv2.imshow('Result', im_content)

        cv2.waitKey(0)

        break
