# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/06/22

import cv2
import numpy as np
from imgproc.segmentation import RegionSegmentation
from imgproc.edge import EdgeProcedures

check_regions = [
    ((20, 22), "must"),
    ((55, 63), "must"),
    ((31, 33, 37), "must"),
    ((44, 54), "must"),
    ((84, 86, 99, 91, 105), "must"),
    ((149, 150), "must"),
    ((184, 187), "must"),
    ((168, 171), "must"),
    ((211, 212), "must"),
    ((245, 257), "must"),
    ((251, 261), "must"),
    ((1, 21), "should"),
    ((20, 40), "should"),
    ((62, 57), "should"),
    ((50, 74), "should"),
    ((88, 89, 92), "should"),
    ((126, 142), "should"),
    ((138, 144), "should"),
    ((189, 205), "should"),
    ((185, 192), "should"),
    ((174, 176), "should"),
    ((28, 43), "not"),
    ((111, 107), "not"),
    ((107, 108), "not"),
    ((108, 109), "not"),
    ((109, 106), "not"),
    ((106, 100), "not"),
]

def check_scores( img_path, check_list ):
    import itertools
    
    seg = RegionSegmentation( img_path, logging=True )
    scores = seg.scores
    
    result = dict()
    
    for (labels, classify) in check_list:
        
        for label_1, label_2 in itertools.combinations( labels, 2 ):
            if label_1 in scores and label_2 in scores[label_1]:
                
                if classify not in result:
                    result[classify] = list()
                
                result[classify].append(
                    # (label_1, label_2, scores[label_1][label_2])
                    (label_1, label_2, seg.calc_entropy( label_1, label_2, mode='hsv' ))
                )
    
    for k, v in result.items():
        print( k )
        for vv in v:
            print( ', '.join( [str( vvv ) for vvv in vv] ) )


if __name__ == '__main__':
    IMG_PATH = "./img/resource/aerial_roi1_raw_denoised_clipped_ver2.png"
    seg = RegionSegmentation( IMG_PATH, logging=True )
    seg.merge_regions_by_score()

    # seg.merge_region(111, 107)
    # seg.merge_region(107, 108)
    # seg.merge_region(108, 109)
    # seg.merge_region(109, 106)
    # seg.merge_region(106, 100)
    # seg.merged_labels = seg.merged_labels
    
    
    # seg.labels
    
    
    
    # seg.get_segmented_image_with_label()
    # scores = print( seg.scores )
    # check_scores( IMG_PATH, check_regions )
