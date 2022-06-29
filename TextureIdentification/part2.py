import numpy as np
from tqdm import tqdm
from utils import *
import os


if __name__ == '__main__':
    lbp_hist_list = []
    labels_list = []

    # train
    for root, dirs, files in tqdm(os.walk("./data/train", topdown=False)):
        for name in files:
            label = root.split('\\')[1]
            path = os.path.join(root, name)

            img = cv2.imread(path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            H, W = img_gray.shape

            I = lbp_pixel_GPU(img_gray)
            hist = calc_lbp_hist(I)

            lbp_hist_list.append(hist)
            labels_list.append(label)

    # test
    for root, dirs, files in os.walk("./data/test", topdown=False):
        for name in files:
            label = root.split('\\')[1]
            path = os.path.join(root, name)

            img = cv2.imread(path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            H, W = img_gray.shape

            I = lbp_pixel_GPU(img_gray)
            hist = calc_lbp_hist(I)

            idx = find_nearest(lbp_hist_list, hist)
            pred = labels_list[idx]

            print('-'*20)
            print('GT texture:', label)
            print('Pred texture:', pred)
