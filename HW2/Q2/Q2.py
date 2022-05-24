import numpy as np
import cv2.cv2 as cv2
import matplotlib.pyplot as plt
import os
import random
import imageio
from copy import deepcopy


def create_directory(directory_name):
    """Creates an empty directory in the current path."""

    # parent directory path
    parent_dir = os.getcwd()

    # path
    dir_path = os.path.join(parent_dir, directory_name)

    try:
        os.mkdir(dir_path)  # created the directory for images to use in creating the gif

    except FileExistsError:
        # directory already exists ... nothing is needed to be done
        pass


def create_gif(image_base_name, start_num, end_num, images_directory):
    """
    Creates a GIF file from the plots saved as JPG files.
    :param image_base_name: The prefix of each image name in the directory.
    :param start_num: The first number to concatenate to `image_base_name`.
    :param end_num: The last number to concatenate to `image_base_name`.
    :param images_directory: The path of the directory containing the JPG images.
    :return: None
    """

    images = []

    for file_num in range(start_num, end_num + 1):
        curr_file_name = images_directory + image_base_name + str(file_num) + '.jpg'
        images.append(imageio.imread(curr_file_name))

    imageio.mimsave('our_reconstruction.gif', images)


def get_random_rotation_matrix():
    """Returns a random 3D rotation matrix"""

    alpha, beta, gamma = random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360)

    yaw = np.array([
        [np.cos(alpha), -1 * np.sin(alpha), 0],
        [np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]
    ])

    pitch = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-1 * np.sin(beta), 0, np.cos(beta)]
    ])

    roll = np.array([
        [1, 0, 0],
        [0, np.cos(gamma), -1 * np.sin(gamma)],
        [0, np.sin(gamma), np.cos(gamma)]
    ])

    temp = np.matmul(yaw, pitch)
    return np.matmul(temp, roll)


def create_images(points, directory_path):
    """Creates the images needed for creating the GIF as described in the PDF file."""
    image_num = 1

    for degree in range(0, 361, 10):
        x_rotate = np.array([
            [1, 0, 0],
            [0, np.cos(2 * np.pi * degree / 360), -1 * np.sin(2 * np.pi * degree / 360)],
            [0, np.sin(2 * np.pi * degree / 360), np.cos(2 * np.pi * degree / 360)]
        ])

        res = []

        for point in points:
            res.append(np.matmul(x_rotate, point))

        x_coordinates = []
        y_coordinates = []

        for point in res:
            x_coordinates.append(point[0])
            y_coordinates.append(point[1])

        plt.figure()
        plt.ylim(-4, 4)
        plt.xlim(-4, 4)
        plt.scatter(x_coordinates, y_coordinates)
        plt.ylabel('y')
        plt.xlabel('x')

        plt.plot(x_coordinates, y_coordinates)
        plt.savefig(directory_path + 'plot' + str(image_num) + '.jpg')
        image_num += 1

    for degree in range(0, 361, 10):
        y_rotate = np.array([
            [np.cos(2 * np.pi * degree / 360), 0, np.sin(2 * np.pi * degree / 360)],
            [0, 1, 0],
            [-1 * np.sin(2 * np.pi * degree / 360), 0, np.cos(2 * np.pi * degree / 360)]
        ])

        res = []

        for point in points:
            res.append(np.matmul(y_rotate, point))

        x_coordinates = []
        y_coordinates = []

        for point in res:
            x_coordinates.append(point[0])
            y_coordinates.append(point[1])

        plt.figure()
        plt.ylim(-4, 4)
        plt.xlim(-4, 4)

        plt.scatter(x_coordinates, y_coordinates)
        plt.ylabel('y')
        plt.xlabel('x')

        plt.plot(x_coordinates, y_coordinates)
        plt.savefig(directory_path + 'plot' + str(image_num) + '.jpg')
        image_num += 1


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
    return vh[3, 0:3] / vh[3, 3]


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

    # creating directory for images to use in creating the gif
    dir_name = 'ImagesForGIF'
    create_directory(dir_name)

    points_3d = []
    _sum = np.zeros(3)
    for pt1, pt2 in zip(points1, points2):
        pt_3d = DLT(cam_mat1, cam_mat2, pt1, pt2)
        _sum += pt_3d
        points_3d.append(pt_3d)

    points_3d = np.array(points_3d)
    avg_points_3d = _sum / 22
    points_3d -= avg_points_3d

    for i in range(len(colors)):
        colors[i] = (colors[i][0] / 255, colors[i][1] / 255, colors[i][2] / 255)

    plt.figure()

    xis = []
    yis = []

    for point_3d in points_3d:
        xis.append(point_3d[0])
        yis.append(point_3d[1])

    plt.scatter(xis, yis)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.ylim(-4, 4)
    plt.xlim(-4, 4)

    plt.plot(xis, yis)

    plt.savefig('our_matches_xy_projected.jpg')
    plt.show()

    # last task: create images in for loops as described in the pdf file, then create the gif

    random_rotation = get_random_rotation_matrix()
    for i in range(len(points_3d)):
        points_3d[i] = np.matmul(random_rotation, points_3d[i])

    xis = []
    yis = []

    for point_3d in points_3d:
        xis.append(point_3d[0])
        yis.append(point_3d[1])

    plt.scatter(xis, yis)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.ylim(-4, 4)
    plt.xlim(-4, 4)

    plt.plot(xis, yis)
    plt.show()

    path = os.path.join(os.getcwd(), dir_name)

    create_images(points_3d, path + '\\')
    create_gif('plot', 1, 74, path + '\\')
