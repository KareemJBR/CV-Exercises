import cv2
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img1 = cv.imread('cameleon__N_8__sig_noise_5__sig_motion_103/source_01.jpg')  # queryImage
    img2 = cv.imread('cameleon__N_8__sig_noise_5__sig_motion_103/target.jpg')  # trainImage

    sift = cv.SIFT_create()

    ratio_test = 0.8

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < ratio_test * n.distance:
            good.append([m])

    points1 = np.zeros((len(good), 2), dtype=np.float32)
    points2 = np.zeros((len(good), 2), dtype=np.float32)

    for i, match in enumerate(good):
        points1[i, :] = kp1[good[i][0].queryIdx].pt
        points2[i, :] = kp2[good[i][0].trainIdx].pt

    h, mask = cv.findHomography(points1, points2, cv.RANSAC)
    height, width, channels = img2.shape
    regIm = cv.warpPerspective(img1, h, (width, height), flags=2)  # 2 stands for cubic interpolation

    cv.imshow("Result", regIm)
    cv.waitKey(0)
