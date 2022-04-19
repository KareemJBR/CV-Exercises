import cv2
import numpy as np
import glob     # will use it to get files' names
import random


def denoise(im_dir, ratio_test, ransac_iterations, k):
    curr_target = cv2.imread(im_dir + '/target.jpg')

    source_names = glob.glob(im_dir + "/*.jpg")  # a list of all .jpg files in the current directory
    source_names.remove(im_dir + '\\target.jpg')  # keeping the source images only

    source_images = [cv2.imread(im_name) for im_name in source_names]

    counters = np.zeros(shape=(curr_target.shape[0], curr_target.shape[1]))
    results = []  # for each directory, this list will have the results of warping the source images

    sift = cv2.SIFT_create()
    target_kp, target_des = sift.detectAndCompute(curr_target, None)

    for source_image in source_images:
        bf = cv2.BFMatcher()

        source_kp, source_des = sift.detectAndCompute(source_image, None)
        matches = bf.knnMatch(target_des, source_des, k=2)

        good_matches = []  # keeping only the matches that passed the ratio test
        for m, n in matches:
            if m.distance < ratio_test * n.distance:
                good_matches.append(m)

        best_inlier, best_m = -1, None

        for i in range(ransac_iterations):  # implementing RANSAC loop
            counter = 0
            random_matches = random.sample(good_matches, k)

            src_pts = np.float32([source_kp[random_matches[i].trainIdx].pt for i in range(k)])
            dst_pts = np.float32([target_kp[random_matches[i].queryIdx].pt for i in range(k)])

            m_, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            for m in good_matches:
                m_source = source_kp[m.trainIdx].pt
                m_target_old = target_kp[m.queryIdx].pt
                m_target_new = np.matmul(m_, [m_source[0], m_source[1], 1])

                if m_target_new[2] != 0:
                    m_target_new[0] = m_target_new[0] / m_target_new[2]
                    m_target_new[1] = m_target_new[1] / m_target_new[2]

                euclidean_distance = np.sqrt(
                    ((m_target_old[0] - m_target_new[0]) ** 2) + ((m_target_old[1] - m_target_new[1]) ** 2))
                if euclidean_distance < 0.5:
                    counter += 1

            if counter > best_inlier:
                best_inlier = counter
                best_m = m_

        h, w, channels = curr_target.shape

        warped = cv2.warpPerspective(source_image, best_m, (w, h))
        results.append(warped)  # appending the final result of the current source image

        new_img = np.copy(warped)
        for i in range(len(new_img)):   # updating counters matrix used for calculating the average of each pixel
            for j in range(len(new_img[i])):
                color = new_img[i][j]
                if color.any():
                    counters[i][j] += 1

    denoised_im = np.zeros(shape=curr_target.shape, dtype=np.float32)
    for res in results:
        denoised_im = denoised_im + res

    denoised_im /= 255  # colors in cv2 are saved as float values between 0 and 1 instead of [0, 255]

    for i in range(denoised_im.shape[0]):   # we shall divide each pixel value by the number of pixels mapped to it
        for j in range(denoised_im.shape[1]):
            if counters[i][j] != 0:     # avoiding dividing by 0
                denoised_im[i][j] /= counters[i][j]

    # the following commented code is used to write the result to a jpg file

    # if im_dir == 'cameleon__N_8__sig_noise_5__sig_motion_103':
    #     denoised_file = 'Cameleon Denoised.jpg'
    # elif im_dir == 'eagle__N_16__sig_noise_13__sig_motion_76':
    #     denoised_file = 'Eagle Denoised.jpg'
    # elif im_dir == 'einstein__N_5__sig_noise_5__sig_motion_274':
    #     denoised_file = 'Einstein Denoised.jpg'
    # else:
    #     denoised_file = 'Palm Denoised.jpg'
    #
    # denoised_im *= 255
    # cv2.imwrite(denoised_file, denoised_im)
    # denoised_im /= 255

    # showing the final results
    cv2.imshow('Denoising Result', denoised_im)
    cv2.imshow('Target Image', curr_target)
    cv2.waitKey(0)


if __name__ == "__main__":
    cameleon_dir = 'cameleon__N_8__sig_noise_5__sig_motion_103'
    eagle_dir = 'eagle__N_16__sig_noise_13__sig_motion_76'
    einstein_dir = 'einstein__N_5__sig_noise_5__sig_motion_274'
    palm_dir = 'palm__N_4__sig_noise_5__sig_motion_ROT'

    for im_d in (cameleon_dir, eagle_dir, einstein_dir, palm_dir):
        denoise(im_d, 0.8, 260, 4)
