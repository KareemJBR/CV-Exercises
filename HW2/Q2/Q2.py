import numpy as np


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
    cam_mat1, cam_mat2 = read_matrix('cameraMatrix1.txt'), read_matrix('cameraMatrix2.txt')
    matches = get_matches('matchedPoints1.txt', 'matchedPoints2.txt')
