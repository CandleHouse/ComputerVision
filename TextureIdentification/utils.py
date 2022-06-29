import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import cv2


def lbp_pixel_GPU(img):
    """
        CUDA accelerate Local Binary Patterns (LBP)
    """
    @cuda.jit
    def sub_func(temp_device, img_device):
        M, N = temp_device.shape
        x = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x  # H
        y = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y  # W
        if x >= M or y >= N:  # boundary, carefully determined
            return

        b1 = 1 if x - 1 >= 0 and y - 1 >= 0 and temp_device[x - 1, y - 1] > temp_device[x, y] else 0
        b2 = 1 if x - 1 >= 0 and temp_device[x - 1, y] > temp_device[x, y] else 0
        b3 = 1 if x - 1 >= 0 and y + 1 < N and temp_device[x - 1, y + 1] > temp_device[x, y] else 0
        b4 = 1 if y + 1 < N and temp_device[x, y + 1] > temp_device[x, y] else 0
        b5 = 1 if x + 1 < M and y + 1 < N and temp_device[x + 1, y + 1] > temp_device[x, y] else 0
        b6 = 1 if x + 1 < M and temp_device[x + 1, y] > temp_device[x, y] else 0
        b7 = 1 if x + 1 < M and y - 1 >= 0 and temp_device[x + 1, y - 1] > temp_device[x, y] else 0
        b8 = 1 if y - 1 >= 0 and temp_device[x, y - 1] > temp_device[x, y] else 0

        lbp = 0
        for b in (b1, b2, b3, b4, b5, b6, b7, b8):
            lbp = lbp * 2 + b

        img_device[x, y] = lbp

    img_temp = img.copy()
    H, W = img.shape
    BlockThread = (16, 16)  # use 16x16 threads concurrent
    GridBlock = (H // BlockThread[0] + 1, W // BlockThread[1] + 1)  # do H*W times

    temp_device = cuda.to_device(img_temp)
    img_device = cuda.to_device(img)
    sub_func[GridBlock, BlockThread](temp_device, img_device)
    cuda.synchronize()
    I = img_device.copy_to_host()

    return I


def calc_lbp_hist(img, show=False):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist /= np.sum(hist)
    if show:
        plt.plot(hist)
        plt.title('256 bins normalized histogram')
        plt.xlabel('pixel value')
        plt.ylabel('proportion')
        plt.show()
    return hist


def lbp_histograms_distance(h1, h2):
    dist = np.sum(np.abs(h1 - h2))
    return dist


def transformImg(img, center, angle, scale):
    H, W = img.shape
    transform_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
    transform_image = cv2.warpAffine(src=img, M=transform_matrix, dsize=(W, H))
    return transform_image


def find_nearest(array, value) -> int:
    array = np.squeeze(np.asarray(array))
    value = np.squeeze(np.asarray(value))

    dist_min = 10e5
    for i in range(array.shape[0]):
        dist = lbp_histograms_distance(array[i], value)
        if dist < dist_min:
            dist_min = dist
            idx = i
    return idx
