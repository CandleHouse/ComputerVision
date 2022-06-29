"""
    Local Binary Patterns (LBP) identify pictures based on their texture.
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from tqdm import trange
from utils import *


if __name__ == '__main__':
    img = cv2.imread('./data/train/wood/wood1.png')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = img_gray.shape

    # first operation
    I = lbp_pixel_GPU(img_gray)
    hist0 = calc_lbp_hist(I)

    # compute hist dist after rotation with increasing angles
    dist_list = []
    for angle in trange(360):
        img_trans = transformImg(img_gray, center=(W/2, H/2), angle=angle, scale=2)
        I = lbp_pixel_GPU(img_trans)
        hist = calc_lbp_hist(I)
        dist = lbp_histograms_distance(hist0, hist)
        dist_list.append(dist)

    print(dist_list)
    plt.plot(range(360), dist_list)
    plt.title('Rotation histogram distance compare')
    plt.xlabel('rotation degree')
    plt.ylabel('normalized histogram distance')
    plt.show()

    # compute hist dist after scaling with increasing factors
    dist_list = []
    for scale in trange(5, 30):  # scale * 10 for Integer loop in range
        img_trans = transformImg(img_gray, center=(W/2, H/2), angle=0, scale=scale/10)
        I = lbp_pixel_GPU(img_trans)
        hist = calc_lbp_hist(I)
        dist = lbp_histograms_distance(hist0, hist)
        dist_list.append(dist)

    print(dist_list)
    plt.plot([i/10 for i in range(5, 30)], dist_list)
    plt.title('Scaling histogram distance compare')
    plt.xlabel('scaling')
    plt.ylabel('normalized histogram distance')
    plt.show()
