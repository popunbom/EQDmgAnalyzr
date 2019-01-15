import cv2

if __name__ == '__main__':
    inImg = cv2.imread("img/extracted.png")
    inImg = cv2.cvtColor(inImg, cv2.COLOR_RGB2HSV)

    ch = cv2.split(inImg)

    wndName = ["Hue", "Satulation", "Value"]
    for i in range(3):
        cv2.namedWindow(wndName[i])
        cv2.imshow(wndName[i], ch[i])

    cv2.waitKey(0)