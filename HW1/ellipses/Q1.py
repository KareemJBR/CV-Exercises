import cv2.cv2 as cv2
from scipy.signal import convolve
import numpy as np


def add_votes(point_1, point_2, xi1, xi2, votes_im, elips_min_dis, elips_max_dis):
    """Adds votes for finding the ellipses' centers based on the algorithm in paper for finding ellipses' centers"""

    if xi2 != xi1:
        mid_point = (int((point_1[0] + point_2[0]) // 2), int((point_1[1] + point_2[1]) // 2))  # M

        intersection_point = (
            int((point_1[1] - point_2[1] - (point_1[0] * xi1) + (point_2[0] * xi2)) // (xi2 - xi1)) + 1,
            int((xi1 * xi2 * (point_2[0] - point_1[0]) - (point_2[1] * xi1) + (point_1[1] * xi2)) // (xi2 - xi1)))  # T

        # print(mid_point, '.........', intersection_point)
        temp_ = np.zeros(votes_im.shape)
        dis = ((((mid_point[0] - intersection_point[0]) ** 2) + ((mid_point[1] - intersection_point[1]) ** 2)) ** 0.5)
        if elips_min_dis < dis < elips_max_dis:
            votes_im = cv2.line(temp_, mid_point, intersection_point, (0.0001, 0, 0), 1) + votes_im

    return votes_im


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

    # 1-------------------------------------------- moon image -------------------------------------------- #
    moon = cv2.imread('moon.webp')
    moon_temp = cv2.resize(moon, (moon.shape[1] // 3, moon.shape[0] // 3), interpolation=cv2.INTER_CUBIC)

    edge_map = cv2.Canny(moon_temp, 100, 500, apertureSize=3, L2gradient=True)
    cv2.imwrite('moon_edge_map.jpg', edge_map)

    gradient_map = getGradient(edge_map)
    votes = np.zeros(edge_map.shape)

    for i in range(edge_map.shape[0]):
        for j in range(edge_map.shape[1]):
            if edge_map[i][j] != 0:
                p1 = (i, j)
                for row in range(edge_map.shape[0]):
                    for col in range(edge_map.shape[1]):
                        if edge_map[row][col] != 0:
                            p2 = (row, col)
                            xi_1 = gradient_map[i][j]
                            xi_2 = gradient_map[row][col]

                            votes = add_votes(p1, p2, xi_1, xi_2, votes_im=votes, elips_min_dis=0, elips_max_dis=1000)

    votes_temp = np.copy(votes)

    for i in range(votes.shape[0]):
        for j in range(votes.shape[1]):
            if votes[i][j] > np.max(np.max(votes)) - 0.000005:
                temp = cv2.circle(moon_temp, (i, j), 3, (0, 0, 255), -1)

    cv2.imwrite('moon_res.jpg', moon_temp)
    cv2.imshow('moon res', moon_temp)
    cv2.waitKey(0)

    # 2-------------------------------------------- wheels image -------------------------------------------- #
    wheels = cv2.imread('nEKGD2wNiwqrTOc63kiWZT7b4.png')
    wheels_temp = cv2.resize(wheels, (wheels.shape[1] // 3, wheels.shape[0] // 3), interpolation=cv2.INTER_CUBIC)
    kernel_size = 3
    wheels_temp = cv2.GaussianBlur(wheels_temp, (kernel_size, 1), sigmaX=15, sigmaY=20)
    edge_map = cv2.Canny(wheels_temp, 100, 500, apertureSize=3, L2gradient=True)
    cv2.imshow('wheels_temp', edge_map)
    cv2.waitKey(0)
    cv2.imwrite('wheels_edge_map.jpg', edge_map)

    gradient_map = getGradient(edge_map)
    votes = np.zeros(edge_map.shape)

    for i in range(edge_map.shape[0]):
        for j in range(edge_map.shape[1]):
            if edge_map[i][j] != 0:
                p1 = (i, j)
                for row in range(edge_map.shape[0]):
                    for col in range(edge_map.shape[1]):
                        if edge_map[row][col] != 0:
                            p2 = (row, col)
                            xi_1 = gradient_map[i][j]
                            xi_2 = gradient_map[row][col]

                            votes = add_votes(p1, p2, xi_1, xi_2, votes_im=votes, elips_min_dis=0, elips_max_dis=1000)

    votes_temp = np.copy(votes)

    for i in range(votes.shape[0]):
        for j in range(votes.shape[1]):
            if votes[i][j] > np.max(np.max(votes)) - 0.05:
                temp = cv2.circle(wheels_temp, (i, j), 2, (0, 0, 255), -1)

    cv2.imwrite('wheels_res.jpg', wheels_temp)
    cv2.imshow('wheels res', wheels_temp)
    cv2.waitKey(0)

    # # 3-------------------------------------------- cup image -------------------------------------------- # done
    cup = cv2.imread('s-l400.jpg')
    cup_temp = cv2.resize(cup, (cup.shape[1] // 3, cup.shape[0] // 3), interpolation=cv2.INTER_CUBIC)

    edge_map = cv2.Canny(cup_temp, 100, 500, apertureSize=3, L2gradient=True)
    cv2.imwrite('cup_edge_map.jpg', edge_map)
    cv2.imshow('edge_map', edge_map)
    # cv2.waitKey(0)

    gradient_map = getGradient(edge_map)
    votes = np.zeros(edge_map.shape)

    for i in range(edge_map.shape[0]):
        for j in range(edge_map.shape[1]):
            if edge_map[i][j] != 0:
                p1 = (i, j)
                for row in range(edge_map.shape[0]):
                    for col in range(edge_map.shape[1]):
                        if edge_map[row][col] != 0:
                            p2 = (row, col)
                            xi_1 = gradient_map[i][j]
                            xi_2 = gradient_map[row][col]

                            votes = add_votes(p1, p2, xi_1, xi_2, votes_im=votes, elips_min_dis=0, elips_max_dis=1000)

    votes_temp = np.copy(votes)

    for i in range(votes.shape[0]):
        for j in range(votes.shape[1]):
            if votes[i][j] > np.max(np.max(votes)) - 0.005:
                temp = cv2.circle(cup_temp, (i, j), 2, (0, 0, 255), -1)

    cv2.imwrite('cup_res.jpg', cup_temp)
    cv2.imshow('cup res', cup_temp)
    cv2.waitKey(0)

    # # 4-------------------------------------------- box image -------------------------------------------- #
    box = cv2.imread('images.jpg')
    box_temp = cv2.resize(box, (box.shape[1] // 3, box.shape[0] // 3), interpolation=cv2.INTER_CUBIC)
    kernel_size = 3
    box_temp = cv2.GaussianBlur(box_temp, (3, kernel_size), 0)
    edge_map = cv2.Canny(box_temp, 130, 150, apertureSize=3, L2gradient=True)
    cv2.imwrite('box_edge_map.jpg', edge_map)
    cv2.imshow('edge_map', edge_map)
    # cv2.waitKey(0)

    gradient_map = getGradient(edge_map)
    votes = np.zeros(edge_map.shape)

    for i in range(edge_map.shape[0]):
        for j in range(edge_map.shape[1]):
            if edge_map[i][j] != 0:
                p1 = (i, j)
                for row in range(edge_map.shape[0]):
                    for col in range(edge_map.shape[1]):
                        if edge_map[row][col] != 0:
                            p2 = (row, col)
                            xi_1 = gradient_map[i][j]
                            xi_2 = gradient_map[row][col]

                            votes = add_votes(p1, p2, xi_1, xi_2, votes_im=votes, elips_min_dis=7, elips_max_dis=10)

    votes_temp = np.copy(votes)

    for i in range(votes.shape[0]):
        for j in range(votes.shape[1]):
            if votes[i][j] > np.max(np.max(votes)) - 0.0002:
                temp = cv2.circle(box_temp, (i, j), 2, (0, 0, 255), -1)

    cv2.imwrite('box_res.jpg', box_temp)
    cv2.imshow('box', box_temp)
    cv2.waitKey(0)

    # # 5-------------------------------------------- round image -------------------------------------------- #
    round_im = cv2.imread('1271488188_2077d21f46_b.jpg')
    round_temp = cv2.resize(round_im, (round_im.shape[1] // 3, round_im.shape[0] // 3), interpolation=cv2.INTER_CUBIC)
    kernel_size = 3
    round_temp = cv2.GaussianBlur(round_temp, (3, kernel_size), 0)
    edge_map = cv2.Canny(round_temp, 200, 500, apertureSize=3, L2gradient=True)
    cv2.imwrite('round_edge_map.jpg', edge_map)
    cv2.imshow('edge_map', edge_map)
    # cv2.waitKey(0)
    gradient_map = getGradient(edge_map)
    votes = np.zeros(edge_map.shape)

    for i in range(edge_map.shape[0]):
        for j in range(edge_map.shape[1]):
            if edge_map[i][j] != 0:
                p1 = (i, j)
                for row in range(edge_map.shape[0]):
                    for col in range(edge_map.shape[1]):
                        if edge_map[row][col] != 0:
                            p2 = (row, col)
                            xi_1 = gradient_map[i][j]
                            xi_2 = gradient_map[row][col]

                            votes = add_votes(p1, p2, xi_1, xi_2, votes_im=votes, elips_min_dis=5, elips_max_dis=50)

    votes_temp = np.copy(votes)

    for i in range(votes.shape[0]):
        for j in range(votes.shape[1]):
            if votes[i][j] > np.max(np.max(votes)) - 0.003:
                temp = cv2.circle(round_temp, (i, j), 2, (0, 0, 255), -1)

    cv2.imwrite('round_res.jpg', round_temp)
    cv2.imshow('round', round_temp)
    cv2.waitKey(0)

    # # 6-------------------------------------------- london image -------------------------------------------- #
    london = cv2.imread('gettyimages-1212455495-612x612.jpg')
    london_temp = cv2.resize(london, (london.shape[1] // 3, london.shape[0] // 3), interpolation=cv2.INTER_CUBIC)
    kernel_size = 7
    london_temp = cv2.GaussianBlur(london_temp, (1, kernel_size), 0)
    edge_map = cv2.Canny(london_temp, 240, 400, L2gradient=True)
    cv2.imwrite('london_edge_map.jpg', edge_map)
    cv2.imshow('edge_map', edge_map)
    # cv2.waitKey(0)
    gradient_map = getGradient(edge_map)
    votes = np.zeros(edge_map.shape)

    for i in range(edge_map.shape[0]):
        for j in range(edge_map.shape[1]):
            if edge_map[i][j] != 0:
                p1 = (i, j)
                for row in range(edge_map.shape[0]):
                    for col in range(edge_map.shape[1]):
                        if edge_map[row][col] != 0:
                            p2 = (row, col)
                            xi_1 = gradient_map[i][j]
                            xi_2 = gradient_map[row][col]

                            votes = add_votes(p1, p2, xi_1, xi_2, votes_im=votes, elips_min_dis=5, elips_max_dis=50)

    votes_temp = np.copy(votes)

    for i in range(votes.shape[0]):
        for j in range(votes.shape[1]):
            if votes[i][j] > np.max(np.max(votes)) - 0.02:
                temp = cv2.circle(london_temp, (i, j), 2, (0, 0, 255), -1)

    cv2.imwrite('london_res.jpg', london_temp)
    cv2.imshow('london', london_temp)
    cv2.waitKey(0)

    last = cv2.imread('5da02f8f443e6-brondby-haveby-allotment-gardens-copenhagen-denmark-7.jpg.png')
    last_temp = cv2.resize(last, (last.shape[1] // 3, last.shape[0] // 3), interpolation=cv2.INTER_CUBIC)
    kernel_size = 7
    last_temp = cv2.GaussianBlur(last_temp, (1, kernel_size), 0)
    edge_map = cv2.Canny(last_temp, 240, 400, L2gradient=True)
    cv2.imwrite('temp_edge_map.jpg', edge_map)
    cv2.imshow('edge_map', edge_map)
    cv2.waitKey(0)
    gradient_map = getGradient(edge_map)
    votes = np.zeros(edge_map.shape)

    for i in range(edge_map.shape[0]):
        for j in range(edge_map.shape[1]):
            if edge_map[i][j] != 0:
                p1 = (i, j)
                for row in range(edge_map.shape[0]):
                    for col in range(edge_map.shape[1]):
                        if edge_map[row][col] != 0:
                            p2 = (row, col)
                            xi_1 = gradient_map[i][j]
                            xi_2 = gradient_map[row][col]
                            votes = add_votes(p1, p2, xi_1, xi_2, votes_im=votes, elips_min_dis=5, elips_max_dis=50)

    votes_temp = np.copy(votes)

    for i in range(votes.shape[0]):
        for j in range(votes.shape[1]):
            if votes[i][j] > np.max(np.max(votes)) - 0.02:
                temp = cv2.circle(last_temp, (i, j), 2, (0, 0, 255), -1)

    cv2.imwrite('last_res.jpg', last_temp)
    cv2.imshow('last', last_temp)
    cv2.waitKey(0)
