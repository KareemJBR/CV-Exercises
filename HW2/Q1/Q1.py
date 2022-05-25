import numpy as np
import matplotlib.pyplot as plt
import cv2.cv2 as cv2
from copy import deepcopy


def tellme(msg, img):
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.setp(plt.gca())
    plt.title(msg, fontsize=12)
    plt.draw()


def getImagePts(image1, image2, var_name1, var_name2):
    # first image
    tellme('please select 10 points in for this image', image1)
    pts = np.asarray(plt.ginput(10, timeout=-1, show_clicks=True))
    plt.close()
    np.save(var_name1 + '.npy', np.concatenate((np.round(pts), np.ones((np.shape(pts)[0]))[:, np.newaxis]), axis=1))

    # second image
    tellme('please select 10 points in for this image', image2)
    pts = np.asarray(plt.ginput(10, timeout=-1, show_clicks=True))
    plt.close()
    np.save(var_name2 + '.npy', np.concatenate((np.round(pts), np.ones((np.shape(pts)[0]))[:, np.newaxis]), axis=1))


def draw_lines(img1, img2, lines, pts1, pts2, used_colors):
    """ img1 - image on which we draw the epilines for the points in img2 lines - corresponding epilines. """
    r_, c_ = img1.shape[:2]

    i = 0
    for r1_, pt1, pt2 in zip(lines, pts1, pts2):
        color = used_colors[i]
        x_0, y_0 = map(int, [0, -r1_[2] / r1_[1]])
        x_1, y_1 = map(int, [c_, -(r1_[2] + r1_[0] * c_) / r1_[1]])
        img1 = cv2.line(img1, (x_0, y_0), (x_1, y_1), color, 2)
        img1 = cv2.circle(img1, (pt1[0], pt1[1]), 5, color, -1)
        img2 = cv2.circle(img2, (pt2[0], pt2[1]), 5, color, -1)
        i += 1

    return img1, img2


def calc_SED(s_1, s_2, l_, l_tag):
    """
    Calculates the value of SED.
    :param s_1: The 10 points in the first image.
    :param s_2: The corresponded 10 points in the second image.
    :param l_: The lines created from matching the first image to the second one.
    :param l_tag: The lines created from matching the second image to the first one.
    :return: The value of SED as float.
    """
    p_, p_tag = s_1[:10], s_2[:10]
    res = 0

    for ind in range(p_.shape[0]):
        x, y, a, b, c_ = p_[ind][0], p_[ind][1], l_[ind][0], l_[ind][1], l_[ind][2]
        dist1 = 1 / (abs(a * x + b * y + c_) / (a ** 2 + b ** 2))
        x, y, a, b, c_ = p_tag[ind][0], p_tag[ind][1], l_tag[ind][0], l_tag[ind][1], l_tag[ind][2]

        dist2 = abs(a * x + b * y + c_) / np.sqrt(a ** 2 + b ** 2)
        res += np.sqrt(dist1 ** 2 + dist2 ** 2)

    res /= 10

    return res


if __name__ == "__main__":

    loc2_frame1 = cv2.imread('location_2_frame_001.jpg')
    loc2_frame2 = cv2.imread('location_2_frame_002.jpg')

    loc1_frame1 = cv2.imread('location_1_frame_001.jpg')
    loc1_frame2 = cv2.imread('location_1_frame_002.jpg')

    colors = [
        (255, 0, 0),
        (255, 255, 0),
        (255, 255, 255),
        (0, 255, 0),
        (0, 255, 255),
        (0, 0, 255),
        (255, 0, 255),
        (0, 0, 0),
        (100, 200, 200),
        (50, 150, 255),
    ]

    sets = [(loc2_frame1, loc2_frame2, 'loc2'), (loc1_frame1, loc1_frame2, 'loc1')]

    for im1, im2, file_prefix in sets:
        # getImagePts(im1, im2, file_prefix + '_s1', file_prefix + '_s2')
        # getImagePts(im1, im2, file_prefix + '_t1', file_prefix + '_t2')

        s1 = np.load(file_prefix + '_' + 's1.npy').astype(int)  # first set - left image points
        s2 = np.load(file_prefix + '_' + 's2.npy').astype(int)  # first set - right image points

        t1 = np.load(file_prefix + '_' + 't1.npy').astype(int)  # second set - left image points
        t2 = np.load(file_prefix + '_' + 't2.npy').astype(int)  # second set - right image points

        F1, mask1 = cv2.findFundamentalMat(s1, s2, cv2.FM_8POINT)

        for x1, x2, file_name_suffix in [(s1, s2, 'S1'), (t1, t2, 'S2')]:
            im1_copy = deepcopy(im1)
            im2_copy = deepcopy(im2)

            lines1 = cv2.computeCorrespondEpilines(x2.reshape(-1, 1, 2), 2, F1).reshape(-1, 3)
            im1_res, _ = draw_lines(im1_copy, im2_copy, lines1, x1, x2, colors)

            lines2 = cv2.computeCorrespondEpilines(x1.reshape(-1, 1, 2), 1, F1).reshape(-1, 3)
            im2_res, _ = draw_lines(im2_copy, im1_copy, lines2, x2, x1, colors)

            curr_sed = calc_SED(x1, x2, lines1, lines2)

            plt.figure()
            plt.suptitle('SED = ' + str(curr_sed))

            plt.subplot(1, 2, 1)
            plt.imshow(im1_res)
            plt.subplot(1, 2, 2)
            plt.imshow(im2_res)

            plt.savefig(file_prefix + '_' + file_name_suffix + '.jpg')
            plt.show()
