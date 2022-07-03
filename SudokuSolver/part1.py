import cv2
import matplotlib.pyplot as plt
import operator
import numpy as np


if __name__ == '__main__':
    img = cv2.imread('./img/sudoku1.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = img_gray.shape

    # 2.1 Reduce the noise
    img_sm = cv2.GaussianBlur(img_gray, ksize=None, sigmaX=1, sigmaY=1)

    # 2.2 Extract the regions corresponding to contours
    img_binary = cv2.adaptiveThreshold(src=img_sm, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       thresholdType=cv2.THRESH_BINARY, blockSize=21, C=10)

    # 2.3 Find the contours
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 2.4 Identify the puzzle contour among the found contours
    counters_list = []
    max_counter = contours[0]
    max_area = cv2.contourArea(max_counter)

    for contour in contours:
        if 50 < cv2.contourArea(contour) < H*W * 0.9:  # drop outer frame
            counters_list.append(contour)

            if cv2.contourArea(contour) > max_area:  # max area with in the image
                max_area = cv2.contourArea(contour)
                max_counter = contour

    cv2.drawContours(img, counters_list, -1, (0, 0, 255), 2)

    # 3. Deskew the Sudoku puzzle
    for epsilon in range(512, 0, -50):
        approx = cv2.approxPolyDP(max_counter, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] == 4:
            cv2.drawContours(img, approx, -1, (255, 0, 0), 10)
            plt.imshow(img, cmap='gray')
            plt.show()
            break

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in approx]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in approx]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in approx]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in approx]), key=operator.itemgetter(1))

    src_points = np.float32([approx[top_left], approx[top_right], approx[bottom_right], approx[bottom_left]])
    dist_points = np.float32([[0, 0], [512, 0], [512, 512], [0, 512]])
    ### Method 1. opencv warpPerspective
    # matrix = cv2.getPerspectiveTransform(src_points, dist_points)  # <= x, y direction
    # img_trans = cv2.warpPerspective(img_gray, matrix, (512, 512))

    ### Method 2. manually done warpPerspective
    matrix = cv2.getPerspectiveTransform(dist_points, src_points)  # <= x, y direction
    img_trans = np.zeros((512, 512))
    for i in range(img_trans.shape[0]):
        for j in range(img_trans.shape[1]):
            p_ = matrix @ np.array([i, j, 1])
            x, y = p_[0] / p_[2], p_[1] / p_[2]
            kx, ky = int(x), int(y)

            if 0 < kx < img_gray.shape[1] and 0 < ky < img_gray.shape[0]:
                img_trans[j, i] = img_gray[ky, kx]

    plt.imshow(img_trans, cmap='gray')
    plt.show()
