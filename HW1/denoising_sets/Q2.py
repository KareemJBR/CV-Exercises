import cv2.cv2 as cv2
import numpy as np
import glob     # will use it to get files' names

if __name__ == "__main__":
    cameleon_dir = 'cameleon__N_8__sig_noise_5__sig_motion_103'
    eagle_dir = 'eagle__N_16__sig_noise_13__sig_motion_76'
    einstein_dir = 'einstein__N_5__sig_noise_5__sig_motion_274'
    palm_dir = 'palm__N_4__sig_noise_5__sig_motion_ROT'

    ratio_test = 0.8
    RANSAC_iterations = 10

    for im_dir in (cameleon_dir, eagle_dir, einstein_dir, palm_dir):
        curr_target = cv2.imread(im_dir + '/target.jpg')

        source_names = glob.glob(im_dir + "/*.jpg")     # a list of all .jpg files in the current directory
        source_names.remove(im_dir + '\\target.jpg')    # keeping the source images only

        source_images = [cv2.imread(im_name) for im_name in source_names]

        counters = np.zeros(shape=curr_target.shape)
        results = []    # for each directory, this list will have the results of warping the source images

        for source_image in source_images:
            sift = cv2.SIFT_create()
            bf = cv2.BFMatcher()

            kp1, des1 = sift.detectAndCompute(source_image, None)
            kp2, des2 = sift.detectAndCompute(curr_target, None)

            best_matches = []

            for i in range(RANSAC_iterations):
                matches = bf.knnMatch(des1, des2, k=2)

                good_matches = []   # keeping only the matches that passed the ratio test
                for m, n in matches:
                    if m.distance < ratio_test * n.distance:
                        good_matches.append([m])

                if len(good_matches) > len(best_matches):
                    best_matches = good_matches

            points1 = np.zeros((len(best_matches), 2), dtype=np.float32)
            points2 = np.zeros((len(best_matches), 2), dtype=np.float32)

            for i, match in enumerate(best_matches):
                points1[i, :] = kp1[best_matches[i][0].queryIdx].pt
                points2[i, :] = kp2[best_matches[i][0].trainIdx].pt

            h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
            height, width, channels = curr_target.shape
            regIm = cv2.warpPerspective(source_image, h, (width, height), flags=2)  # 2 stands for bi-cubic
            # interpolation

            results.append(regIm)

            ones = np.ones(shape=curr_target.shape)     # will use it for counting for each pixel how many pixels were
            # mapped to it
            ones = cv2.warpPerspective(ones, h, (width, height), flags=2)

            ones[ones != 0] = 1
            counters += ones

        denoised_im = np.zeros(shape=curr_target.shape, dtype=np.float32)
        for res in results:
            denoised_im = denoised_im + res

        denoised_im /= 255      # colors in cv2 are saved as values in [0, 1] instead of [0, 255]

        for i in range(denoised_im.shape[0]):
            for j in range(denoised_im.shape[1]):
                for k in range(denoised_im.shape[2]):
                    if counters[i][j][k] != 0:      # avoiding dividing by 0
                        denoised_im[i][j][k] /= counters[i][j][k]

        cv2.imshow('Denoising Result', denoised_im)
        cv2.imshow('Target Image', curr_target)
        cv2.waitKey(0)
