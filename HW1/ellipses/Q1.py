import cv2.cv2 as cv2
import glob
from copy import deepcopy
from scipy.signal import convolve
import numpy as np


def contrastEnhance(im, range):
    a = float(255 / (range[1] - range[0]))
    b = 0 - a * range[0]
    # print(a, b)
    nim = a * im.astype(np.float32) + b
    return nim

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


def add_votes(point_1, point_2, xi1, xi2, votes_im):
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
        mid_point = (int((point_1[0] + point_2[0]) // 2), int((point_1[1] + point_2[1]) // 2))  # M

        intersection_point = (
            int((point_1[1] - point_2[1] - (point_1[0] * xi1) + (point_2[0] * xi2)) // (xi2 - xi1))+1,
            int((xi1 * xi2 * (point_2[0] - point_1[0]) - (point_2[1] * xi1) + (point_1[1] * xi2)) // (xi2 - xi1)))  # T

        # print(mid_point, '.........', intersection_point)
        temp = np.zeros(votes_im.shape)
        if intersection_point[0] != mid_point[0]:
            tm_slope = (intersection_point[1] - mid_point[1]) / (intersection_point[0] - mid_point[0])
            tm_const = mid_point[1] - (tm_slope * mid_point[0])

            second_point_X= int( mid_point[0]+1)
            second_point_Y = int(second_point_X*tm_slope + tm_const)

            votes_im = cv2.line(temp, mid_point, (second_point_X,second_point_Y), (0.0001, 0, 0), 1) + votes_im
        else:
            votes_im = cv2.line(temp, mid_point,intersection_point, (0.0001, 0, 0), 1) + votes_im

    return votes_im

        # tm_slope = (intersection_point[1] - mid_point[1]) / (intersection_point[0] - mid_point[0])
        # tm_const = mid_point[1] - (tm_slope * mid_point[0])
        #
        # for r in range(votes_im.shape[0]):
        #     for c in range(votes_im.shape[1]):
        #         if np.abs(r * tm_slope + tm_const - c) < 1:
        #             votes_im[r][c] += 1


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
            edge_map = cv2.Canny(im, 100, 500, apertureSize=3, L2gradient=True)
            # cv2.imshow('edge map', edge_map)
            continue
        elif im_name == "72384675-very-long-truck-trailer-for-exceptional-transport-with-many-sturdy-tires.webp":
            edge_map = cv2.Canny(im, 50, 255)
            continue
        elif im_name == "1271488188_2077d21f46_b.jpg":
            edge_map = cv2.Canny(im, 70, 150)
            continue
        elif im_name == "alvtd333_alvin_template_small_ellipse.png":
            edge_map = cv2.Canny(im, 80, 200)
            continue
        elif im_name == "gettyimages-1212455495-612x612.jpg":
            edge_map = cv2.Canny(im, 160, 180)
            continue
        elif im_name == "hammer-tissot-big.jpg":
            edge_map = cv2.Canny(im, 100, 200)
            continue
        elif im_name == "Headline-Pic.jpg":
            edge_map = cv2.Canny(im, 100, 200)
        elif im_name == "images.jpg":
            edge_map = cv2.Canny(im, 100, 200)
            continue
        elif im_name == "nEKGD2wNiwqrTOc63kiWZT7b4.png":
            edge_map = cv2.Canny(im, 100, 500, apertureSize=3, L2gradient=True)
            continue
        elif im_name == "s-l400.jpg":
            edge_map = cv2.Canny(im, 100, 180)
            continue
        elif im_name == "sewage-treatment-plant-wastewater-treatment-water-use-filtration-effluent-and-waste-water" \
                        "-industrial-solutions-for-sewerage-water-treatment-and-rec-2H4RX20.jpg":
            edge_map = cv2.Canny(im, 100, 200)
            continue
        else:  # "Traitement-dimage-drone-.jpeg":
            edge_map = cv2.Canny(im, 100, 200)
            continue

        gradient_map = getGradient(edge_map)



        scale_percent = 20  # percent of original size
        width = int(edge_map.shape[1] * scale_percent / 100)
        height = int(edge_map.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(edge_map, dim, interpolation=cv2.INTER_CUBIC)
        im_content_resized = cv2.resize(im_content, dim, interpolation=cv2.INTER_CUBIC)
        ellipse_center_votes = np.zeros(resized.shape)
        gradient_map = getGradient(resized)
        black_board = np.zeros((resized.shape[0],resized.shape[1],3))
        # cv2.imshow("Resized image", resized)


        # TODO: choose P and Q

        for i in range(resized.shape[0]):
            for j in range(resized.shape[1]):
                if resized[i][j] >= 1:
                    p1 = (i, j)

                    for k in range(resized.shape[0]):
                        for l in range(resized.shape[1]):
                            if resized[k][l] != 0 and (i!=k or j!=l):
                                p2 = (k, l)
                                ellipse_center_votes = add_votes(p1, p2, gradient_map[p1[0], p1[1]],
                                                                 gradient_map[p2[0], p2[1]],
                                                                 ellipse_center_votes)

        cv2.imshow('ellipse_center_votes', ellipse_center_votes)
        ellipse_center_votes = contrastEnhance(ellipse_center_votes, [np.min(np.ravel(ellipse_center_votes)),
                                                                      np.max(np.ravel(ellipse_center_votes))])
        cv2.imshow('ellipse_center_votes -contrast', ellipse_center_votes/510)
        cv2.imshow('im_content_resized', im_content_resized)
        ellipse_center_votes /= 510


        for i in range (ellipse_center_votes.shape[0]):
            for j in range (ellipse_center_votes.shape[1]):
                if ellipse_center_votes[i][j]>np.max(np.ravel(ellipse_center_votes))-0.05:
                    # ellipse_center_votes = cv2.circle(ellipse_center_votes, (i, j), 3, (255, 0, 0), 3)
                    black_board = cv2.circle(black_board, (i, j), 2, (0, 0, 255), 2)

                #     # ellipse_center_votes[i][j] = 1
                # else:
                #     ellipse_center_votes[i][j] = 0


        cv2.imshow('res', im_content_resized)

        scale_percent = 500  # percent of original size
        width = int(black_board.shape[1] * scale_percent / 100)
        height = int(black_board.shape[0] * scale_percent / 100)
        dim = (width, height)
        black_board = cv2.resize(black_board, dim, interpolation=cv2.INTER_CUBIC)
        black_board = black_board.astype(np.uint8)
        cv2.imshow('black_board-res', black_board)
        cv2.imshow('im_content-pre-res', im_content)
        im_content += black_board

        cv2.imshow('im_content-res', im_content)




        # thresh = 10
        # centers = get_ellipse_centers(ellipse_center_votes, thresh)


        # im_lines = cv2.HoughLinesP(ellipse_center_votes, 1, np.pi / 180, 0.5, minLineLength=10, maxLineGap=250)
        #
        # for line in im_lines:
        #     x1, y1, x2, y2 = line[0]
        #     cv2.line(temp, (x1, y1), (x2, y2), (255, 0, 0), 3)
        #
        # cv2.imshow('temp', temp)
        # lines = cv2.HoughLines(edge_map, 1, np.pi / 180, 70)
        # for rho, theta in lines[0]:
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     x0 = a * rho
        #     y0 = b * rho
        #     pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        #     pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        #     temp = cv2.line(temp, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

        # temp[int(x0)][int(y0)] = 255

        #
        # lines = cv2.HoughLines(temp, 1, np.pi / 180, 200)
        # for rho, theta in lines[0]:
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     x0 = a * rho
        #     y0 = b * rho
        #
        #     im_content = cv2.circle(im_content, (x0,y0), 3,(255,0,0),2 )

        # bonus:
        # draw_ellipse(centers, im_content, (255, 255, 0))

        # cv2.imshow('image', im)
        # cv2.imshow('edge map', edge_map)
        # cv2.imshow('gradient', gradient_map)
        # cv2.imshow('Result', im_content)

        cv2.waitKey(0)

        break
