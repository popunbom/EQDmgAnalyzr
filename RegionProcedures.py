#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import json


def getEdgeMagnitudeAndAngle(src_img):
    """
    Sobel フィルタによるエッジ強度・エッジ角度の抽出

    Parameters
    ----------
    src_img : numpy.ndarray\n
    \t入力画像\n

    Returns
    -------
    M : numpy.ndarray(dtype=np.float32)\n
    \tエッジ強度 : [0, 1.0]\n
    A : numpy.ndarray(dtype=np.float32)\n
    \tエッジ角度 : [0°, 360°]\n

    """
    assert type(src_img) == np.ndarray, \
        f"argument 'src_img' must be numpy.ndarray, not '{type(src_img)}'"

    # Convert to Grayscale
    if src_img.ndim == 3:
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    # Sobel Edge Detector
    dx = cv2.Sobel(src_img, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(src_img, cv2.CV_32F, 0, 1, ksize=3)

    # Edge Magnitude
    M = np.sqrt(dx*dx + dy*dy)
    M /= M.max()

    # Edge Angle (0° <= θ <= 360°)
    A = np.round(np.degrees(np.arctan2(dy, dx))) + 180

    return M, A


def colorizedEdgeAngle(edge_angle, edge_magnitude, max_intensity=False, mask_img=None):
    """
    エッジ角度の疑似カラー画像を生成\n
    \n
    \tHue (色相) に角度値を割り当て、HSV→RGB への\n
    \t色空間変換を利用して疑似カラー画像生成を行う。\n


    Parameters
    ----------
    edge_angle : numpy.ndarray\n
    \tエッジ角度 : [0°, 360°]\n
    edge_magnitude : numpy.ndarray\n
    \tエッジ強度 : [0, 1.0]\n
    max_intensity : bool\n
    \tTrue のとき、HSV の V 値は全画素において最大値となる\n
    \tFalse のとき、HSV の V 値には正規化されたエッジ強度値\n
    \tが割り当てられる\n
    mask_img : [None, numpy.ndarray]\n
    \tマスク画像(2値化済み)\n
    \t白色(非ゼロ値)を透過対象とする\n
    \tmask_img が与えられた場合、疑似カラー画像に対して\n
    \tマスクを適用した結果が返却される\n


    Returns
    -------
    angle_img : numpy.ndarray\n
    \tmask_img が None の場合に返却される\n
    \t疑似カラー画像(BGR)\n
    masked_img : numpy.ndarray\n
    \tmask_img が None でない場合に返却される\n
    \tマスク済み疑似カラー画像(BGR)\n
    """

    assert type(edge_angle) == np.ndarray, \
        f"argument 'edge_angle' must be 'numpy.ndarray', not '{type(edge_angle)}'"
    assert type(edge_magnitude) == np.ndarray, \
        f"argument 'edge_magnitude' must be 'numpy.ndarray', not '{type(edge_magnitude)}'"
    assert mask_img is None or type(mask_img) == np.ndarray, \
        f"argument 'masgk_img' must be None or 'numpy.ndarray', not '{type(mask_img)}'"

    assert edge_angle.shape == edge_magnitude.shape, \
        "argument 'edge_angle' and 'edge_magnitude' must be same shape"

    M, A = edge_magnitude, edge_angle

    h = (A / 2).astype(np.uint8)
    s = np.ones(h.shape, dtype=h.dtype) * 255
    if max_intensity:
        v = np.ones(h.shape, dtype=h.dtype) * 255
    else:
        v = (M * (255.0 / M.max())).astype(np.uint8)

    angle_img = cv2.cvtColor(np.stack([h, s, v], axis=2), cv2.COLOR_HSV2BGR)

    if mask_img is None:
        return angle_img
    else:
        if mask_img.max() != 1:
            mask_img[mask_img > 0] = 1
        masked_img = angle_img * np.stack([mask_img] * 3, axis=2)

        return masked_img


def drawAngleLines(src_img, edge_angle, line_color=(255, 255, 255), line_length=10, mask_img=None):
    """
    エッジ角度に対応する線分を描画する

    Parameters
    ----------
    src_img : numpy.ndarray\n
    \t角度線が描画されるベース画像\n
    edge_angle : numpy.ndarray\n
    \tエッジ角度\n
    line_color : tuple\n
    \t線の色(R, G, B の順)\n
    line_length : int\n
    \t線の長さ\n
    mask_img : [None, numpy.ndarray]\n
    \tマスク画像(2値化済み)\n
    \tmask_img が与えられた場合、白色(非ゼロ値)の\n
    \t箇所のみ角度線が描画される\n

    Returns
    -------
    angle_line_img : numpy.ndarray\n
    \t角度線が描画された画像(BGR)\n
    \t線描画の都合上、画像の大きさが縦、横\n
    \tそれぞれ3倍されて返却される\n
    """

    assert type(src_img) == np.ndarray and src_img.ndim == 3, \
        f"argument 'src_img' must be numpy.ndarray with BGR Color, not '{type(src_img)}'"
    assert mask_img is None or (type(mask_img) == np.ndarray and mask_img.ndim == 2 and src_img.shape[:2] == mask_img.shape), \
        f"argument 'mask_img' must be None or numpy.ndarray(ndim=2, same shape with 'src_img')"

    angle_line_img = cv2.resize(
        src_img, dsize=None, fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST)

    if mask_img is None:
        for i in range(src_img.shape[0]):
            for j in range(src_img.shape[1]):
                a_rad = np.radians(edge_angle[i, j])

                p = np.array([3*(j+1)-2, 3*(i+1)-2], dtype=np.float32)
                a = np.array([np.cos(a_rad), np.sin(a_rad)], dtype=np.float32)

                d = (line_length * 0.5 * a)
                pt_1 = tuple((p + d).astype(np.int16))
                pt_2 = tuple((p - d).astype(np.int16))

                cv2.line(angle_line_img, pt_1, pt_2, line_color, thickness=1)
    else:
        for i in range(src_img.shape[0]):
            for j in range(src_img.shape[1]):
                if mask_img[i, j] != 0:
                    a_rad = np.radians(edge_angle[i, j])

                    p = np.array([3*(j+1)-2, 3*(i+1)-2], dtype=np.float32)
                    a = np.array([np.cos(a_rad), np.sin(a_rad)],
                                 dtype=np.float32)

                    d = (line_length * 0.5 * a)
                    pt_1 = tuple((p + d).astype(np.int16))
                    pt_2 = tuple((p - d).astype(np.int16))

                    cv2.line(angle_line_img, pt_1, pt_2,
                             line_color, thickness=1)

    return angle_line_img


def calcScoreBetweenRegions(src_img, labels, label_a, label_b):
    pass


def calcRegionRelation(segmented_img, labels=None, relations=None, iteration=1):
    """
    ラベリング結果から隣接領域情報を生成する\n

    Parameters
    ----------
    segmented_img : numpy.ndarray\n
    \t背景(黒)と領域分割線(白)のみの2値化画像
    labels : [None, numpy.ndarray]\n
    \tsegmented_img に対するラベリング結果\n
    \tlabels が与えられていない場合、\n
    \tcv2.connectedComponents()により算出する\n
    relations : dict\n
    \t隣接領域の情報\n
    iteration : int\n
    \t繰り返し処理を行う回数\n
    \t分割線を追跡する処理の都合上、孤立してしまう\n
    \t線分が存在する場合は、その分処理を繰り返す\n
    \t必要がある\n


    Returns
    -------
    relations : dict\n
    \t隣接領域の情報
    """

    import queue
    import itertools

    assert type(segmented_img) == np.ndarray and segmented_img.ndim == 2, \
        f"argument 'segmented_img' must be numpy.ndarray(ndim=2), not {type(segmented_img)}"
    assert labels is None or (type(labels) == np.ndarray and labels.shape == segmented_img.shape), \
        f"argument 'labels' must be None or numpy.ndarray(same shape with 'segmented_img'), not {type(labels)}"

    D = [
        (0,  1), (-1,  1), (-1,  0), (-1, -1),
        (0, -1), (1, -1), (1,  0), (1,  1)
    ]

    if labels is None:
        # Labeling (4-connectivity)
        _, labels = cv2.connectedComponents(segmented_img, connectivity=4)

    print(f"--- Tracking (Rest {iteration} times)")
    iteration -= 1

    # Init graph
    if relations is None:
        relations = dict()

    # Create 'checked' matrix
    checked = np.zeros(labels.shape, dtype=np.bool)

    # Find line pixel
    y, x = np.argwhere(segmented_img != 0)[0]

    # Init queue
    q = queue.Queue()
    q.put((y, x))

    # checked: (x, y)
    checked[y, x] = True

    while not q.empty():

        y, x = q.get()
        print(f"\rPixel: ({x}, {y}) ", end="", flush=True)

        label_set = set()
        for d in D:
            p = (y+d[0], x+d[1])
            if 0 <= p[0] < segmented_img.shape[0] and 0 <= p[1] < segmented_img.shape[1]:
                if segmented_img[p] != 0:
                    if not checked[p]:
                        # Mark 'p' as 'checked'
                        checked[p] = True
                        # Add Tracking Queue
                        q.put(p)
                else:
                    label_set.add(int(labels[p]))

        for key, l_connected in [z for z in itertools.product(label_set, repeat=2) if z[0] != z[1]]:
            if key not in relations.keys():
                relations[key] = set()

            relations[key].add(l_connected)

    print("\n")

    if iteration > 0:
        return calcRegionRelation(segmented_img, labels, relations, iteration)
    else:
        return relations


def test():
    src_img = cv2.imread(
        "./img/resource/aerial_roi1_raw_denoised_cripped.png",
        cv2.IMREAD_COLOR
    )
    segmented_img = cv2.imread(
        "./img/tmp/binary_segmentation/20190617_115343/binary_segmentation.png",
        cv2.IMREAD_GRAYSCALE
    )
    segmented_img = np.bitwise_not(segmented_img)

    # getEdgeMagnitudeAndAngle
    M, A = getEdgeMagnitudeAndAngle(src_img)

    # colorizedEdgeAngle
    colorized_img = colorizedEdgeAngle(
        A, M, max_intensity=True, mask_img=segmented_img)

    # drawAngleLines
    angle_line_img = drawAngleLines(colorized_img, A, mask_img=segmented_img)

    # calcRegionRelation
    relations = calcRegionRelation(segmented_img, iteration=2)

    for k, v in relations.items():
        relations[k] = sorted(list(v))

    print("Writing relations ... ", end="", flush=True)
    with open("./data/region_relation.json", "wt") as f:
        json.dump(relations, f, sort_keys=True)
    print("done!")


if __name__ == '__main__':
    test()
