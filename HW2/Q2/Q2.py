import numpy as numpy
import cv2.cv2 as cv2
import matplotlib.pyplot as plt
from copy import deepcopy
import plotly.express as px

from scipy import linalg



def get_matches(pts1_file, pts2_file):
    """Returns a dictionary of matches that read from the received files as arguments."""
    matches_dict = {}

    with open(pts1_file, 'r') as jab1, open(pts2_file, 'r') as jab2:
        pts1_lines, pts2_line = jab1.readlines(), jab2.readlines()

        for line1, line2 in zip(pts1_lines, pts2_line):
            if len(line1) > 0 and len(line2) > 0:
                temp1, temp2 = line1.split(','), line2.split(',')
                point1, point2 = [], []
                for val in temp1:
                    point1.append(float(val))

                point1 = tuple(point1)
                for val in temp2:
                    point2.append(float(val))

                point2 = tuple(point2)

                matches_dict[point1] = point2

    return matches_dict


def read_matrix(file_name):
    """Reads the camera matrix in the file received as an argument and returns it as a numpy array."""
    res_mat = []

    with open(file_name, 'r') as jabber:
        for line in jabber.readlines():
            if len(line) > 0:
                curr_vec = line.split(',')
                temp = []
                for val in curr_vec:
                    temp.append(float(val))
                res_mat.append(temp)

    res_mat = numpy.array(res_mat)
    return res_mat

def DLT(P1, P2, point1, point2):
    A = [point1[1] * P1[2, :] - P1[1, :],
         P1[0, :] - point1[0] * P1[2, :],
         point2[1] * P2[2, :] - P2[1, :],
         P2[0, :] - point2[0] * P2[2, :]
         ]
    A = numpy.array(A).reshape((4, 4))
    # print('A: ')
    # print(A)

    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices=False)

    print('Triangulated point: ')
    print(Vh[3, 0:3] / Vh[3, 3])
    return Vh[3, 0:3] / Vh[3, 3]


if __name__ == "__main__":

    # task 1

    cam_mat1, cam_mat2 = read_matrix('cameraMatrix1.txt'), read_matrix('cameraMatrix2.txt')
    matches = get_matches('matchedPoints1.txt', 'matchedPoints2.txt')

    im1, im2 = cv2.imread('house_1.png'), cv2.imread('house_2.png')

    temp_im1, temp_im2 = deepcopy(im1), deepcopy(im2)

    points1 = [k for k in matches.keys()]
    points2 = [v for v in matches.values()]

    colors = [
        (255, 0, 0),
        (255, 255, 0),
        (0, 255, 0),
        (0, 255, 255),
        (0, 0, 255),
        (255, 0, 255),
        (0, 255, 255)
    ]

    for i in range(len(points1) - 1):
        cv2.line(temp_im1, (int(points1[i][0]), int(points1[i][1])), (int(points1[i + 1][0]), int(points1[i + 1][1])),
                 colors[i % len(colors)], thickness=2)

    for i in range(len(points2) - 1):
        cv2.line(temp_im2, (int(points2[i][0]), int(points2[i][1])), (int(points2[i + 1][0]), int(points2[i + 1][1])),
                 colors[i % len(colors)], thickness=2)

    plt.subplot(1, 2, 1)
    plt.imshow(temp_im1)

    plt.subplot(1, 2, 2)
    plt.imshow(temp_im2)

    plt.savefig('our_matches_connected.jpg')
    plt.show()

    # task 2




    points_3d = []
    sum = numpy.zeros(3)
    for pt1, pt2 in zip(points1, points2):
        pt_3d = DLT(cam_mat1, cam_mat2, pt1, pt2)
        sum += pt_3d
        points_3d.append(pt_3d)
    points_3d = numpy.array(points_3d)
    avrg_points_3d = numpy.divide(sum,22)
    points_3d = points_3d - avrg_points_3d

    matches_cy_projected = numpy.ones(shape=(8,8))



    for i in range(21):
        cv2.line(matches_cy_projected, (int(points_3d[i][0]), int(points_3d[i][1])), (int(points_3d[i + 1][0]), int(points_3d[i + 1][1])),
                 0, thickness=10)

    fig = px.imshow(matches_cy_projected)
    fig.show()
