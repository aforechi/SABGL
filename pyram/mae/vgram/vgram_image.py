import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2



def emptyLike(image):
    return np.empty_like(image)


def flattenImage(image):
    return image.flatten() #http://stackoverflow.com/questions/7755684/flatten-opencv-numpy-array


def translateImage(image, x, y):
    rows, cols, _ = image.shape
    M = np.float32([[1,0,x],[0,1,y]])
    return cv2.warpAffine(image, M, (cols,rows))


def cropImage(image, x1, y1, x2, y2):
    return image[y1:y2, x1:x2]


def emptyDataset(rows, width, height):
    return np.zeros((rows, width * height), dtype=np.int32)


def concatImages(left, right):
    return np.hstack((left,right))


def readImageColor(filename):
    return cv2.imread(filename, cv2.IMREAD_COLOR)


def createCLAHE(clipLimit, tileGridSize):
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    return clahe


def applyCLAHE(clahe, image):
    return clahe.apply(image)


def resizeInterLinear(image, width, height):
    return cv2.resize(image, (height, width), interpolation = cv2.INTER_LINEAR)


def applyGaussian(image, radius, sigma):
    return cv2.GaussianBlur(image, (radius,)*2, sigma)


def convertBGR2RGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convertINT2RGB(image):
    rgb = image.view(np.int8).reshape(image.shape+(4,))[..., :3]
    return rgb.astype(np.uint8)


def convertRGB2INT(image):
    rbg_int = image.astype(np.int32)
    rbg_int = np.bitwise_or(np.bitwise_or(np.bitwise_and(rbg_int[:,:,0], 0x000000ff),
                            np.left_shift(np.bitwise_and(rbg_int[:,:,1], 0x000000ff), 8)),
                            np.left_shift(np.bitwise_and(rbg_int[:,:,2], 0x000000ff), 16))
    return rbg_int


def convertBGR2INT(image):
    rbg_int = image.astype(np.int32)
    rbg_int = np.bitwise_or(np.bitwise_or(np.bitwise_and(rbg_int[:,:,2], 0x000000ff),
                            np.left_shift(np.bitwise_and(rbg_int[:,:,1], 0x000000ff), 8)),
                            np.left_shift(np.bitwise_and(rbg_int[:,:,0], 0x000000ff), 16))
    return rbg_int


def showImageRGB(image):
    plt.axis("off")
    plt.imshow(image)
    plt.show()


def showImageBGR(image):
    showImageRGB(convertBGR2RGB(image))


def showImageGreyscale(image):
    plt.axis("off")
    plt.imshow(image, cmap = cm.Greys_r)
    plt.show()
        
