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
            break

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in approx]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in approx]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in approx]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in approx]), key=operator.itemgetter(1))

    src_points = np.float32([approx[top_left], approx[top_right], approx[bottom_right], approx[bottom_left]])
    dist_points = np.float32([[0, 0], [512, 0], [512, 512], [0, 512]])
    matrix = cv2.getPerspectiveTransform(src_points, dist_points)
    img_trans = cv2.warpPerspective(img_gray, matrix, (512, 512))

    # 4. Localize the cells in the puzzle
    cells_list = []
    for i in range(9):
        for j in range(9):
            cells_list.append(img_trans[8*(i+1)+48*i: 8*(i+1)+48*(i+1), 8*(j+1)+48*j: 8*(j+1)+48*(j+1)])
            plt.subplot(9, 9, 9*i+j+1)
            plt.imshow(img_trans[8*(i+1)+48*i: 8*(i+1)+48*(i+1), 8*(j+1)+48*j: 8*(j+1)+48*(j+1)], cmap='gray')
            plt.xticks([])
            plt.yticks([])
    plt.show()

    # 5. Identify the empty/filled cells
    def isCellEmpty(cell: np.array, th=90) -> bool:
        img_binary = cv2.adaptiveThreshold(src=cell, maxValue=1, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           thresholdType=cv2.THRESH_BINARY, blockSize=17, C=25)
        num = np.ones_like(cell).sum() - cv2.countNonZero(img_binary)
        return True if num < th else False

    for i, cell in enumerate(cells_list):
        if isCellEmpty(cell):
            cells_list[i] = np.zeros_like(cells_list[i])

        plt.subplot(9, 9, i+1)
        plt.imshow(cells_list[i], cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()
