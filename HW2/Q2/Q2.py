import numpy as np
import cv2.cv2 as cv2
import matplotlib.pyplot as plt
from copy import deepcopy


def DLT(p1, p2, point1, point2):
    a = [point1[1] * p1[2, :] - p1[1, :],
         p1[0, :] - point1[0] * p1[2, :],
         point2[1] * p2[2, :] - p2[1, :],
         p2[0, :] - point2[0] * p2[2, :]
         ]
    a = np.array(a).reshape((4, 4))

    b = a.transpose() @ a
    from scipy import linalg
    u, s, vh = linalg.svd(b, full_matrices=False)

    print('Triangulated point: ')
    print(vh[3, 0:3] / vh[3, 3])
    return vh[3, 0:3] / vh[3, 3]


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

    res_mat = np.array(res_mat)
    return res_mat


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

    
