# Importing PIL Image module and mamba
import os
import cv2
import numpy as np
from PIL import Image
from mamba import *

import CommonProcedures as cp

D_SAVE_TEMP_IMG = True
D_PATH_TEMP_IMG = "./img/tmp/binary_segmentation"


def binarySegmentation(img):
    global D_PATH_TEMP_IMG

    assert isinstance(img, np.ndarray), \
        f"argument 'img' must be numpy.ndarray, not {type(img)}"

    # Create temp dir
    if D_SAVE_TEMP_IMG:
        D_PATH_TEMP_IMG = os.path.join(D_PATH_TEMP_IMG, cp.getTempDirName())
        if not os.path.exists(D_PATH_TEMP_IMG):
            os.makedirs(D_PATH_TEMP_IMG)

    if img.ndim == 3:
        imBlue, imGreen, imRed = [cp.cv2mamba(img[:, :, i]) for i in range(3)]
    elif img.ndim == 2:
        imBlue, imGreen, imRed = [cp.cv2mamba(img)] * 3

    # We will perform a thick gradient on each color channel (contours in original
    # picture are more or less fuzzy) and we add all these gradients
    gradIm = imageMb(imRed)
    imWrk1 = imageMb(imRed)
    gradIm.reset()
    gradient(imRed, imWrk1, 2)
    add(imWrk1, gradIm, gradIm)
    gradient(imGreen, imWrk1, 2)
    add(imWrk1, gradIm, gradIm)
    gradient(imBlue, imWrk1, 2)
    add(imWrk1, gradIm, gradIm)

    # Then we invert the gradient image and we compute its quasi-distance
    qDist = imageMb(gradIm, 32)
    negate(gradIm, gradIm)
    quasiDistance(gradIm, imWrk1, qDist)

    if D_SAVE_TEMP_IMG:
        imWrk1.save(os.path.join(D_PATH_TEMP_IMG, "quasi_dist_gradient.png"))
    if D_SAVE_TEMP_IMG:
        qDist.save(os.path.join(D_PATH_TEMP_IMG, "quasi_dist.png"))

    # The maxima of the quasi-distance are extracted and filtered (too close maxima,
    # less than 6 pixels apart, are merged)
    imWrk2 = imageMb(imRed)
    imMark = imageMb(gradIm, 1)
    copyBytePlane(qDist, 0, imWrk1)
    subConst(imWrk1, 3, imWrk2)
    build(imWrk1, imWrk2)
    maxima(imWrk2, imMark)

    # The marker-controlled watershed of the gradient is performed
    imWts = imageMb(gradIm)
    label(imMark, qDist)
    negate(gradIm, gradIm)
    watershedSegment(gradIm, qDist)
    copyBytePlane(qDist, 3, imWts)

    # The segmented binary and color image are stored
    logic(imRed, imWts, imRed, "sup")
    logic(imGreen, imWts, imGreen, "sup")
    logic(imBlue, imWts, imBlue, "sup")
    pilim = mix(imRed, imGreen, imBlue)
    if D_SAVE_TEMP_IMG:
        pilim.save(os.path.join(D_PATH_TEMP_IMG, 'segmented_gallery.png'))
    negate(imWts, imWts)
    if D_SAVE_TEMP_IMG:
        imWts.save(os.path.join(D_PATH_TEMP_IMG, 'binary_segmentation.png'))


def extractByLabel(src_img, labels, label_nums):
    dst_img = np.zeros(src_img.shape, dtype=src_img.dtype)

    for label_num in label_nums:
        if src_img.ndim == 3:
            for i in range(3):
                dst_img[:, :, i] += src_img[:, :, i] * (labels == label_num)

    return dst_img


def getSegmentedImageWithLabel(segmented_img, color=(255, 0, 0)):
    assert type(segmented_img) == np.ndarray and segmented_img.ndim == 2, \
        "argument 'segmented_img' must be numpy.ndarray and ndim = 2"

    segmented_img = cv2.resize(
        segmented_img,
        dsize=None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_NEAREST
    )
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        segmented_img, connectivity=4)

    dst = cv2.cvtColor(segmented_img, cv2.COLOR_GRAY2BGR)
    for label_num, centroid in enumerate(centroids):
        cv2.putText(
            img=dst,
            text=str(label_num),
            org=tuple(centroid.astype(np.int16)),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=0.5,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA
        )


if __name__ == '__main__':
    IMG_PATH = "./img/resource/aerial_roi1_raw_denoised_cripped.png"

    img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)

    binarySegmentation(img)
