import numpy as np
import matplotlib.pyplot as plt
import cv2
from copy import deepcopy
from numpy.linalg import norm


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


def draw_lines(img1, img2, lines, pts1, pts2):
    """ img1 - image on which we draw the epilines for the points in img2 lines - corresponding epilines. """
    r_, c_ = img1.shape[:2]

    i = 0
    for r1_, pt1, pt2 in zip(lines, pts1, pts2):
        color = colors[i]
        x_0, y_0 = map(int, [0, -r1_[2] / r1_[1]])
        x_1, y_1 = map(int, [c_, -(r1_[2] + r1_[0] * c_) / r1_[1]])
        img1 = cv2.line(img1, (x_0, y_0), (x_1, y_1), color, 1)
        img1 = cv2.circle(img1, (pt1[0], pt1[1]), 5, color, -1)
        img2 = cv2.circle(img2, (pt2[0], pt2[1]), 5, color, -1)
        i += 1

    return img1, img2


if __name__ == "__main__":
    im1 = cv2.imread('location_2_frame_001.jpg')
    im2 = cv2.imread('location_2_frame_002.jpg')

    # getImagePts(im1, im2, 's1', 's2')     # TODO: uncomment after tests
    # getImagePts(im1, im2, 't1', 't2')

    s1 = np.load('s1.npy').astype(int)  # first set - left image points
    s2 = np.load('s2.npy').astype(int)  # first set - right image points

    t1 = np.load('t1.npy').astype(int)  # second set - left image points
    t2 = np.load('t2.npy').astype(int)  # second set - right image points

    F1, mask1 = cv2.findFundamentalMat(t1, t2, cv2.FM_8POINT)

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

    original_im1 = deepcopy(im1)
    original_im2 = deepcopy(im2)
    # Find epilines corresponding to the points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(t2.reshape(-1, 1, 2), 2, F1)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = draw_lines(im1, im2, lines1, t1, t2)

    # Find epilines corresponding to the points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(t1.reshape(-1, 1, 2), 1, F1)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = draw_lines(im2, im1, lines2, t2, t1)

    r, c = im1.shape[:2]
    sed = 0
    for r1, s in zip(lines1, t2):
        x0, y0 = map(int, [0, -r1[2] / r1[1]])
        x1, y1 = map(int, [c, -(r1[2] + r1[0] * c) / r1[1]])
        p1 = np.asarray((x0, y0))
        p2 = np.asarray((x1, y1))
        p3 = np.asarray((s[0], s[1]))
        d = norm(np.cross(p2 - p1, p1 - p3)) / norm(p2 - p1)
        sed += d**2

    plt.figure()
    plt.suptitle("SED = " + str(sed))

    plt.subplot(1, 2, 1)
    plt.imshow(img3)

    plt.subplot(1, 2, 2)
    plt.imshow(img4)
    plt.savefig('result.jpg')
    plt.show()

    # the second set

    im1 = original_im1
    im2 = original_im2
    # Find epilines corresponding to the points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(s2.reshape(-1, 1, 2), 2, F1)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = draw_lines(im1, im2, lines1, s1, s2)

    # Find epilines corresponding to the points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(s1.reshape(-1, 1, 2), 1, F1)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = draw_lines(im2, im1, lines2, s2, s1)

    r, c = im1.shape[:2]
    sed = 0
    for r1, s in zip(lines1, s2):
        x0, y0 = map(int, [0, -r1[2] / r1[1]])
        x1, y1 = map(int, [c, -(r1[2] + r1[0] * c) / r1[1]])
        p1 = np.asarray((x0, y0))
        p2 = np.asarray((x1, y1))
        p3 = np.asarray((s[0], s[1]))
        d = norm(np.cross(p2 - p1, p1 - p3)) / norm(p2 - p1)
        sed += d**2

    plt.figure()
    plt.suptitle("SED = " + str(sed))

    plt.subplot(1, 2, 1)
    plt.imshow(img3)

    plt.subplot(1, 2, 2)
    plt.imshow(img4)
    plt.savefig('result.jpg')
    plt.show()
