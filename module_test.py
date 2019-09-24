#/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/09/23

import cv2
import numpy as np
from skimage.morphology import disk

from scipy import ndimage as ndi

from imgproc.edge import EdgeProcedures
from imgproc.utils import compute_by_window
from utils.mpl import show_images, imshow
from utils.logger import ImageLogger
from os import path

PATH_ROOT = "./img/resource"

img = cv2.imread(
    path.join(PATH_ROOT, "aerial_roi4_without_line_noise.png")
)
X, Y, W, H = 0, 441, 348, 287
img = img[Y:Y + H, X:X + W]
img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

logger = ImageLogger("./img/tmp/module_test")


def fourier_hpf( in_img, threshold ):
    def disk_mask( r, h, w ):
        mask = disk( r )
        p_h, p_w = (h - mask.shape[0], w - mask.shape[1])
        mask = np.pad(
            mask,
            [(
                (p_h) // 2,
                (p_h) // 2 + (p_h % 2)
            ), (
                (p_w) // 2,
                (p_w) // 2 + (p_w % 2)
            )],
            'constant'
        ).astype( bool )
        
        return mask
    
    
    if in_img.ndim == 3:
        img_gs = cv2.cvtColor( in_img, cv2.COLOR_BGR2GRAY )
    else:
        img_gs = in_img
    
    fft = np.fft.fftshift(
        np.fft.fft2( img_gs )
    )
    
    mask = disk_mask( threshold, *img_gs.shape[:2] )
    #     mask = np.bitwise_not(disk_mask(threshold, *img.shape[:2]))
    
    fft_masked = fft.copy()
    fft_masked[mask] = 0 + 0j
    
    ifft = np.fft.ifft2( fft_masked )
    
    # Calcurate Feature
    fd_img = compute_by_window(
        np.abs( ifft ),
        lambda img: np.mean( img ),
        window_size=8,
        step=2,
        dst_dtype=np.float64
    )
    
    fd_img = ndi.zoom(
        fd_img / fd_img.max(),
        (img_gs.shape[0] / fd_img.shape[0], img_gs.shape[1] / fd_img.shape[1]),
        order=0,
        mode='nearest'
    )
    
    show_images(
        list_img=[
            np.log10( np.abs( fft ) ),
            mask,
            np.log10( np.abs( fft_masked ) ),
            np.abs( ifft ),
            fd_img,
        ],
        plt_title="HPF",
        list_title=[
            "Power Spector (log10)",
            "Mask",
            "Power Spector (masked, log10)",
            "Image with HPF",
            "Mean(window_size=8, step=2)",
        ],
        list_cmap=[
            "jet",
            "gray_r",
            "jet",
            "gray",
            "jet",
        ],
        tuple_shape=(2, 3),
        logger=logger
    )
    
    return fd_img


fd_hpf = fourier_hpf(img_gs, int(min(img_gs.shape[:2]) * 0.05))
