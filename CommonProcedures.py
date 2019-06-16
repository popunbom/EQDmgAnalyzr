import os
from datetime import datetime
import cv2
import numpy as np
from mamba import imageMb
from mamba.miscellaneous import Mamba2PIL, PIL2Mamba
from PIL import Image

import MaskLabeling as ml


def MAX(a, b):
    return a if (a > b) else b


def MIN(a, b):
    return a if (a < b) else b


def getRect(img_shape, points_of_label):
    yMin = img_shape[0]
    xMin = img_shape[1]
    yMax = -1
    xMax = -1

    for p in points_of_label:
        if (p == (-1, -1)):
            continue
        yMin = MIN(yMin, p[0])
        yMax = MAX(yMax, p[0])
        xMin = MIN(xMin, p[1])
        xMax = MAX(xMax, p[1])

    return [yMin, yMax, xMin, xMax]


def imageDivideByMask(src_img, npy_label, dir_name):
    assert (src_img.shape[2] == 3), \
        " 'src_img' must be 3-ch RGB image."

    if (not os.path.exists("img/divided/" + dir_name)):
        os.mkdir("img/divided/" + dir_name)

    for i in range(len(npy_label)):
        P = getRect(src_img.shape, npy_label[i])

        cv2.imwrite("img/divided/" + dir_name + "/%05d.png" %
                    i, src_img[P[0]:P[1], P[2]:P[3]])
        cv2.imwrite("img/divided/" + dir_name + "/canny_%05d.png" % i,
                    cv2.Canny(src_img[P[0]:P[1], P[2]:P[3]], 126, 174))

    print("Total %d image were created." % len(npy_label))

    return len(npy_label)


def imgPreProc(src_img):

    print("Image Pre-Processing ... ", flush=True, end="")
    hsv = cv2.split(cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV))

    # Hue, Saturation: Median( r = 5.0 )
    hsv[0] = cv2.medianBlur(hsv[0], 5)
    hsv[1] = cv2.medianBlur(hsv[1], 5)
    # Value: MeanShift( spacial=8.0 chroma=9.0 )
    hsv[2] = cv2.cvtColor(hsv[2], cv2.COLOR_GRAY2BGR)
    hsv[2] = cv2.pyrMeanShiftFiltering(hsv[2], 8, 18)
    hsv[2] = cv2.cvtColor(hsv[2], cv2.COLOR_BGR2GRAY)

    print("done! ", flush=True)

    return cv2.cvtColor(cv2.merge(hsv), cv2.COLOR_HSV2BGR)


def getROI(img, center_x, center_y, kSize):
    x = center_x
    y = center_y
    assert(kSize % 2 != 0), "kSize is an odd number !"

    if len(img.shape) == 2:
        ret = np.zeros((kSize, kSize), dtype=img.dtype)
    else:
        ret = np.zeros((kSize, kSize, img.shape[2]), dtype=img.dtype)

    for (m, i) in zip(range(0, kSize), range(-(kSize // 2), (kSize // 2) + 1)):
        for (n, j) in zip(range(0, kSize), range(-(kSize // 2), (kSize // 2) + 1)):
            if (0 <= y+i < img.shape[0] and 0 <= x+j < img.shape[1]):
                ret[m, n] = img[y+i, x+j]

    return ret


def getAnswer(img_mask, img_answer):
    assert img_mask.shape == img_answer.shape, "must be same shape!"
    W = np.array([255, 255, 255], dtype=np.uint8)
    R = np.array([0, 0, 255], dtype=np.uint8)
    K = np.array([0, 0, 0], dtype=np.uint8)
    dst = np.zeros(img_answer.shape, dtype=np.uint8)
    # for i in range(img_answer.shape[0]):
    #   for j in range(img_answer.shape[1]):
    #     a = img_answer[i][j]
    #     m = img_mask[i][j]
    #     if ((a == R).all() and (m == W).all()):
    #       dst[i][j] = R
    #     elif ((a == K).all() and (m == W).all()):
    #       dst[i][j] = W

    for i, (aa, mm) in enumerate(zip(img_answer, img_mask)):
        for j, (a, m) in enumerate(zip(aa, mm)):
            if i % 100 == 0 and j == 0:
                print(i, j)
            if (m == W).all():
                if (a == R).all():
                    dst[i][j] = R
                elif (a == K).all():
                    dst[i][j] = W

    return dst


def getTempDirName(prefix="", suffix=""):
    t_now = datetime.now()

    return "{prefix}{date_str}{suffix}".format(
        prefix=prefix + ("_" if prefix else ""),
        date_str=t_now.strftime("%Y%m%d_%H%M%S"),
        suffix=("_" if suffix else "") + suffix
    )

# numpy.ndarray --> mamba.base.imageMb


def cv2mamba(npy_img):
    """
    Convert OpenCV's image (numpy.ndarray) to
    mamba's image (mamba.base.imageMb)

    Parameters
    ----------
    npy_img : numpy.ndarray
        OpenCV's image

    Returns
    -------
    mb_img : mamba.base.iamgeMb
        mamba's image
    """

    assert type(npy_img) == np.ndarray, \
        f"argument 'npy_img' must be numpy.ndarray, not {type(npy_img)}"

    mb_img = imageMb(npy_img.shape[1], npy_img.shape[0])

    if npy_img.ndim == 3:
        npy_img = cv2.cvtColor(npy_img, cv2.COLOR_BGR2RGB)

    PIL2Mamba(Image.fromarray(npy_img), mb_img)

    return mb_img


# mamba.base.imageMb --> numpy.ndarray
def mamba2cv(mb_img):
    """
    Convert mamba's image (mamba.base.imageMb) to 
    OpenCV's image (numpy.ndarray)_


    Parameters
    ----------
    mb_img : mamba.base.iamgeMb
        mamba's image

    Returns
    -------
    npy_img : numpy.ndarray
        OpenCV's image
    """

    assert type(mb_img) == imageMb, \
        f"argument 'mb_img' must be mamba.base.imageMb, not {type(mb_img)}"

    npy_img = np.array(Mamba2PIL(mb_img))

    if npy_img.ndim == 3:
        npy_img = cv2.cvtColor(npy_img, cv2.COLOR_RGB2BGR)

    return npy_img


if __name__ == '__main__':
    # path_img = "img/resource/expr_2_aerial.png"
    # dst = imgPreProc(cv2.imread(path_img, cv2.IMREAD_COLOR))
    # cv2.imwrite("img/resource/"+os.path.splitext(os.path.basename(path_img))[0]+"_customblur.png", dst)
    img_mask = cv2.imread("img/resource/expr_2_mask.png", cv2.IMREAD_COLOR)
    img_answer = cv2.imread(
        "/Users/popunbom/Desktop/expr_2_answer.png", cv2.IMREAD_COLOR)

    dst = getAnswer(img_mask, img_answer)
    cv2.imshow("Test", dst)
    cv2.waitKey(0)
    cv2.imwrite("img/resource/expr_2_answer.png", dst)
