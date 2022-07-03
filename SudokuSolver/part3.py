import cv2
import matplotlib.pyplot as plt
import operator
import numpy as np
import os
from sudoku import Sudoku


if __name__ == '__main__':
    img = cv2.imread('./img/sudoku2.png')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = img_gray.shape

    # 2.1 Reduce the noise
    img_sm = cv2.GaussianBlur(img_gray, ksize=None, sigmaX=1, sigmaY=1)

    # 2.2 Extract the regions corresponding to contours
    img_binary = cv2.adaptiveThreshold(src=img_sm, maxValue=1, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
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
    img_puzzle = img_trans.copy()  # for display solve

    # 4. Localize the cells in the puzzle
    cells_list = []
    for i in range(9):
        for j in range(9):
            cells_list.append(img_trans[8*(i+1)+48*i: 8*(i+1)+48*(i+1), 8*(j+1)+48*j: 8*(j+1)+48*(j+1)])

    # 5. Identify the empty/filled cells
    def isCellEmpty(cell: np.array, th=90) -> bool:
        img_binary = cv2.adaptiveThreshold(src=cell, maxValue=1, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           thresholdType=cv2.THRESH_BINARY, blockSize=17, C=25)
        num = np.ones_like(cell).sum() - cv2.countNonZero(img_binary)
        return True if num < th else False

    for i, cell in enumerate(cells_list):
        if isCellEmpty(cell):
            cells_list[i] = np.zeros_like(cells_list[i])

    # 7.1 Load sample digits’ images
    digits_list = []
    labels_list = []
    i = 0
    for root, dirs, files in os.walk('./digits_arial_calibri/', topdown=False):
        for name in files:
            label = name.split('.png')[0]
            path = os.path.join(root, name)

            digit = cv2.imread(path)
            digit_gray = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
            digit_gray = cv2.resize(digit_gray, (20, 20))  # resize to 20*20 pixels
            digits_list.append(digit_gray)
            labels_list.append(label)

            plt.subplot(4, 5, i+1)
            plt.imshow(digit_gray, cmap='gray')
            plt.title(label)
            plt.xticks([])
            plt.yticks([])
            i += 1

    plt.show()

    # 7.2 Compute HOG descriptors of the sample images
    winSize = (20, 20)
    blockSize = (8, 8)
    blockStride = (4, 4)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = -1
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 0
    nlevels = 64
    signedGradient = 1
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)
    winStride = (3, 3)
    padding = (3, 3)
    referenceHOG_list = []  # 20 HOG vectors
    for digit in digits_list:
        digit_hog = hog.compute(digit, winStride, padding).reshape((-1,))
        referenceHOG_list.append(digit_hog)

    # 7.3 Implement a nearest neighbor classifier
    def compute_NN(testHOG, referenceHOG):
        id_min = 0
        dist_min = np.linalg.norm(testHOG - referenceHOG[id_min])
        for i, hog in enumerate(referenceHOG):
            if np.linalg.norm(testHOG - hog) < dist_min:
                dist_min = np.linalg.norm(testHOG - hog)
                id_min = i
        return id_min, dist_min

    # 7.4 Sliding window
    def sliding_window(image, window, step):
        for x in range(0, image.shape[0] - window[0], step):
            for y in range(0, image.shape[1] - window[1], step):
                yield image[x:x + window[0], y:y + window[1]]

    windowSize = 15
    i = 0  # filled cells index
    board = [[0]*9 for _ in range(9)]  # empty board
    for index, cell in enumerate(cells_list):
        if np.sum(cell) != 0:  # filled cells
            id_min = 0
            dist_min = 10e9
            for win_size in range(windowSize, 48, 3):  # enlarge window size

                for subimage in sliding_window(cell, (win_size, win_size), 1):  # sliding window
                    test_image = cv2.resize(subimage, (20, 20))
                    testHOG = hog.compute(test_image, winStride, padding).reshape((-1,))
                    id, dist = compute_NN(testHOG, referenceHOG_list)
                    if dist < dist_min:
                        dist_min = dist
                        id_min = id

            plt.subplot(6, 5, i + 1)
            plt.imshow(cv2.putText(cell, labels_list[id_min], (0, 48), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0)), cmap='gray')
            plt.xticks([])
            plt.yticks([])
            i += 1

            board[index//9][index % 9] = int(labels_list[id_min])

    plt.show()

    # 8. Solve the Sudoku puzzle
    puzzle = Sudoku(3, 3, board=board)
    solution = puzzle.solve()
    solution.show_full()

    # 8.0 show digit in raw image
    for i in range(9):
        for j in range(9):
            if puzzle.board[i][j] is None:
                answer = solution.board[i][j]
                img_puzzle = cv2.putText(img_puzzle, str(answer), (8*(j+1)+48*j+12, 8*(i+1)+48*(i+1)-12),
                                         cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0))

    plt.imshow(img_puzzle, cmap='gray')
    plt.show()

    img_trans = cv2.warpPerspective(img_puzzle, np.linalg.pinv(matrix), (W, H))
    img_sm[img_trans > 0] = img_trans[img_trans > 0]
    plt.imshow(img_sm, cmap='gray')
    plt.show()


