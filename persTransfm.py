import numpy as np
import cv2

wndName = "Test"


def getCol(n, N_OF_SPLIT=8):
  hsl = [ ( 360 / N_OF_SPLIT ) * ( n % N_OF_SPLIT), 1.0, 1.0 ]
  rgb = cv2.cvtColor( np.array([[hsl]], dtype=np.float32), cv2.COLOR_HSV2RGB_FULL)
  return (rgb[0][0] * 255).astype(np.uint8).tolist()



def onMouse(event, x, y, flags, param):
  global cnum
  if event == cv2.EVENT_LBUTTONDOWN:
    cv2.circle(img, (x, y), 5, getCol( cnum ), thickness=-1, lineType=cv2.LINE_AA)
    cv2.imshow(wndName, img)
    cnum += 1

if __name__ == '__main__':
  cnum = 0
  img = cv2.imread("img/aerial_4.png", cv2.IMREAD_COLOR)
  
  cv2.namedWindow(wndName, cv2.WINDOW_AUTOSIZE)
  cv2.setMouseCallback(wndName, onMouse)
  cv2.imshow(wndName, img)
  cv2.waitKey(0)