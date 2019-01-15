import numpy as np
import cv2
import SaveResultAsImage as sr
import MaskLabeling as ml
import CommonProcedures as cp

FLT_EPS = np.finfo(np.float32).eps
QUANTIZE = 10
# DEBUG FLAG FOR VERBOSE
D_DEBUG_1 = False
# DEBUG FLAG FOR SAVING FILE
D_DEBUG_2 = True

# img 全体の角度分散を計算
def getEdgeAngleVariance(img):
    angle_hsv = getEdgeHSVArray(img)

    deg_arr = np.zeros((360), dtype=np.float64)
    total_votes = 0
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            angle = angle_hsv[i, j, 0]
            strength = angle_hsv[i, j, 2]
            if ( not(0 <= angle <= 360) ):
                print("data error! ( angle = %f )" % angle)
            if ( not(strength < FLT_EPS)):
                # Voting
                deg_arr[ int(angle) % 360 ] += strength
                total_votes += strength
    
    return np.sort(deg_arr)[::-1][0] / total_votes


# Algorithm: Mode of Edge Angle
# 返り値: 配列(label_dataと同サイズ, float64)
def getEdgeAngleVariance(edge_hsv, label_data):
    ret = np.zeros((label_data.shape[0]), dtype=np.float64)

    print("Calculating edge variance ... ", end="", flush=True)

    for i in range(1, label_data.shape[0]):  # FIX(2017/08/06): FIX THE RANGE OF 'i'
        # 角度について投票する配列 (浮動小数点型)
        deg_arr = np.zeros((360), dtype=np.float64)
        total_votes = 0
        # 各ラベルごとに投票処理
        for j in range(len(label_data[i])):
            # 先頭要素は処理しない
            if label_data[i][j] == (-1, -1): continue
            # label_data が示す座標位置の「角度」「強度」をedge_dataから取得
            angle = edge_hsv[label_data[i][j][0], label_data[i][j][1], 0]
            strength = edge_hsv[label_data[i][j][0], label_data[i][j][1], 2]
            # error check
            if (not (0 <= angle and angle <= 360)):
                print("data error! ( angle = %f )" % angle)
            # 「強度」が0でなかったら、「強度」値を投票する
            if (not (strength < FLT_EPS)):
                deg_arr[(int(angle) % 360)] += 1
                total_votes += 1  # FIXED(2017/08/21) from 'total_votes += 1'
        # 投票配列を降順ソートし、最頻値を取得、投票数で割り、戻り値としてセット
        ret[i] = np.sort(deg_arr)[::-1][0] / total_votes

    ret /= ret.max()

    print("done!", flush=True)

    return ret


def getEdgeHSVArray(src_img):
    assert (src_img.shape[2] == 1), "\"src_img\" must be 1-ch grayscale image."
    dst = np.zeros((src_img.shape[0], src_img.shape[1], 3), dtype=np.float32)
    
    
    print("Calculating Edge of whole image ... ", end="", flush=True)

    print("sobel... ", end="", flush=True)
    dx = cv2.Sobel(src_img, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(src_img, cv2.CV_32F, 0, 1, ksize=3)
    LV_MAX = np.sqrt(dx * dx + dy * dy).max()

    print("hsv calc... ", end="", flush=True)
    # 'Hue'        ::= Edge Angle ( normalized [0, 180] )
    dst[:, :, 0] = (np.arctan2(dy, dx) + np.pi) * 180 / np.pi
    # 'Satulation' ::= cannot use
    dst[:, :, 1] = 1.0
    # 'Value'      ::= Edge Strength = sqrt( dx^2 + dy^2 ) normalized [0, 1]
    dst[:, :, 2] = np.sqrt(dx * dx + dy * dy) / LV_MAX

    print("done!", flush=True)

    return dst


# 1. 航空画像全体にキャニーエッジ検出 -> cv2.Canny
# 2. ラベルデータをもとに道路領域の削除 -> cv2.add(src, src, mask=mask_img)
# 3. ラベルデータをもとに各領域を抽出
# 4. 各領域に対して平均エッジ長の計算 -> calcEdgeLength( img )



def getAverageEdgeLength(src_img, mask_img, label_data, sw):
    assert (len(src_img.shape) == 2), "input image must be 1-ch"
    assert (sw == 1 or sw == 2), "arguments \"sw\" must be '1' or '2'"

    edge = cv2.Canny(src_img, 126, 174)
    img = cv2.add(edge, edge, mask=mask_img)
    if (D_DEBUG_2):
        cv2.imwrite("img/edgelength_canny.png", img)
        print("saved canny img ... ", end="", flush=True)

    assert (len(img.shape) == 2), "edge image must be 1-ch"

    edge_length = np.zeros(label_data.shape, dtype=np.float64)

    print("Calculating EDGE Length ... ", end="", flush=True)
    for i in range(1, len(label_data)):
        r = cp.getRect(edge.shape, label_data[i])
        roi = img[r[0]:r[1], r[2]:r[3]]
        if (sw == 1):
            edge_length[i] = calcEdgeLength1(roi)
        elif (sw == 2):
            edge_length[i] = calcEdgeLength2(roi)
        if (i % 100 == 0): print("%d ... " % i, end="", flush=True)


    print("done!", flush=True)
    return edge_length


def calcEdgeLength1(img):
    n_of_edge = 0
    total_length = 0

    # 入力画像のチェック
    assert (len(img.shape) == 2), "input image must be 1ch binary image."

    checked = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            length = 0
            # 注目画素が未探索かつ白画素なら、探索開始
            if (img[i, j] == 255 and [i, j] not in checked):
                # チェック済み配列に注目画素を追加
                checked.append([i, j])
                length += 1
                checked, length = recursiveSearch1(img, [i, j], checked, length)
                n_of_edge += 1
                total_length += length
                if (D_DEBUG_1):
                    print("[%2d, %2d]  length = %2d, n_of_connect = %2d" % (i, j, length, n_of_edge), flush=True)

    if (D_DEBUG_1):
        print("n_of_connect = %d, total_length = %f" % (n_of_edge, total_length), flush=True)

    return (0) if (n_of_edge == 0) else (total_length / n_of_edge)


def recursiveSearch1(img, p, checked, length):
    # 近傍8画素(真横から時計回り)
    NEIGHBOR = np.array([(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)])

    if (D_DEBUG_1):
        print("start! (%d, %d)" % (p[1], p[0]), flush=True)

    for n in NEIGHBOR:
        P = p + n
        if (0 <= P[0] < img.shape[0] and 0 <= P[1] < img.shape[1]):
            if (D_DEBUG_1):
                print("   search... (%d, %d)" % (P[0], P[1]), flush=True)
            if (img[P[0], P[1]] == 255 and P.tolist() not in checked):
                checked.append(P.tolist())
                length += 1
                checked, length = recursiveSearch1(img, P, checked, length)
                if (D_DEBUG_1):
                    print("returned !! (%d, %d)" % (p[1], p[0]), flush=True)

    return checked, length


# FIXED (2017/09/13): 端点についての処理を追加
def calcEdgeLength2(img):
    n_of_edge = 0
    total_length = 0

    # 入力画像のチェック
    assert (len(img.shape) == 2), "input image must be 1ch binary image."

    checked = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            length = 0
            endp = 0
            # 注目画素が未探索かつ白画素なら、探索開始
            if (img[i, j] == 255 and [i, j] not in checked):
                # チェック済み配列に注目画素を追加
                checked.append([i, j])
                length += 1
                checked, length, endp = recursiveSearch2(img, [i, j], checked, length, endp)
                # checked, length = recursiveSearch(img, [i, j], checked, length)
                n_of_edge += 1
                total_length += length / endp
                # total_length += length
                if (D_DEBUG_1):
                    print("[%2d, %2d]  length = %2d, n_of_connect = %2d, endp = %2d" % (i, j, length, n_of_edge, endp),
                          flush=True)
                    # print( "[%2d, %2d]  length = %2d, n_of_connect = %2d" % (i, j, length, n_of_edge) , flush=True)

    if (D_DEBUG_1):
        print("n_of_connect = %d, total_length = %f" % (n_of_edge, total_length), flush=True)

    return (0) if (n_of_edge == 0) else (total_length / n_of_edge)


# FIXED (2017/09/13): 端点についての処理を追加
def recursiveSearch2(img, p, checked, length, endp):
    # 近傍8画素(真横から時計回り)
    NEIGHBOR = np.array([(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)])
    # 端点処理用フラグ
    flags = True

    if (D_DEBUG_1):
        print("start! (%d, %d)" % (p[1], p[0]), flush=True)

    for n in NEIGHBOR:
        P = p + n
        if (0 <= P[0] < img.shape[0] and 0 <= P[1] < img.shape[1]):
            if (D_DEBUG_1):
                print("   search... (%d, %d)" % (P[0], P[1]), flush=True)
            if (img[P[0], P[1]] == 255 and P.tolist() not in checked):
                flags = False
                checked.append(P.tolist())
                length += 1
                checked, length, endp = recursiveSearch2(img, P, checked, length, endp)
                # checked, length = recursiveSearch(img, P, checked, length)
                if (D_DEBUG_1):
                    print("returned !! (%d, %d)" % (p[1], p[0]), flush=True)

    # 端点処理
    if (flags):
        endp += 1
        if (D_DEBUG_1):
            print(" edge! (%d, %d) " % (p[1], p[0]), flush=True)

    return checked, length, endp
    # return checked, length


def imgZoom(src_img, scale):
    dst_img = np.zeros((int(src_img.shape[0] * scale), int(src_img.shape[1] * scale), src_img.shape[2]),
                       dtype=src_img.dtype)

    for y in range(dst_img.shape[0]):
        for x in range(dst_img.shape[1]):
            dst_img[y, x] = src_img[int(y / scale), int(x / scale)]

    return dst_img

def proc1(str1, sw):

    sfx = ("fixed_") if (sw == 2) else ("")

    img_src = cv2.imread("img/aerial_only_"+str1+".png", cv2.IMREAD_COLOR)
    img_mask = cv2.imread("img/mask_invert_"+str1+".png", cv2.IMREAD_GRAYSCALE)
    # npy_label = np.load( "data/label.npy" )
    npy_label = ml.getMaskLabel(img_mask)
    # npy_data  = np.load( "data/edge_length_average.npy" )
    src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    npy_data = getAverageEdgeLength(src_gray, img_mask, npy_label, sw)
    # np.save("data/edge_length_average.npy", edge_len)

    npy_data /= npy_data.max()
    # npy_data = sr.extractInRange(npy_data, 0, 40)
    npy_data = sr.percentileModification(npy_data, 10)
    np.savetxt("data/csv_edgelength_"+sfx+str1+".csv", npy_data, delimiter=",", fmt="%4.8f")

    result_img = sr.createResultImg(img_src, npy_data, npy_label, True)
    result_img = cv2.add(img_src, result_img)

    cv2.imwrite("img/result_edgelength_"+sfx+str1+".png", result_img)

def proc2(str1):
    img_src = cv2.imread("img/non_blured/aerial_only_"+str1+".png", cv2.IMREAD_COLOR)
    img_mask = cv2.imread("img/mask_invert_"+str1+".png", cv2.IMREAD_GRAYSCALE)
    # npy_label = np.load( "data/label.npy" )
    npy_label = ml.getMaskLabel(img_mask)
    # npy_data  = np.load( "data/edge_variance.npy" )
    npy_data = getEdgeAngleVariance(getEdgeHSVArray(img_src), npy_label)
    # np.save("data/edge_variance.npy", edge_len)

    npy_data /= npy_data.max()
    # npy_data = sr.extractInRange(npy_data, 0, 40)
    npy_data = sr.percentileModification(npy_data, 10)
    # np.savetxt("data/csv_edge_variance_"+sfx+str1+".csv", npy_data, delimiter=",", fmt="%4.8f")

    result_img = sr.createResultImg(img_src, npy_data, npy_label, True)
    result_img = cv2.add(img_src, result_img)

    cv2.imshow("Test", result_img)
    cv2.waitKey(0)
    cv2.imwrite("img/result_edge_variance_"+str1+".png", result_img)


if __name__ == '__main__':
    # proc1("roi1", 2)
    # proc1("roi2", 2)
    proc2("roi1")