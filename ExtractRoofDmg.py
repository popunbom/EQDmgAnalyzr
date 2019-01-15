import cv2
import numpy as np
from matplotlib import pyplot as plt

def showGraph(data_array):
  plt.plot( data_array, color='b' )
  # plt.hist( img.flatten(), 256, [0, 256], color='r' )
  # plt.xlim( [0, 256] )
  # plt.legend( ('cdf', 'histogram'), loc='upper left' )
  plt.show()
  
def equalizeHistByHSVWithMask( img_rgb, img_mask_bin):
  assert len( img_rgb.shape ) == 3 and img_rgb.shape[2] == 3, "img_rgb must be 3-ch RGB Image !"
  assert len( img_mask_bin.shape ) == 2, "img_mask_bin must be 1-ch Binary Image !"
  img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
  h, s, v = [img_hsv[:, :, i] for i in range( 3 )]

  cv2.imshow( "Test", img_rgb )
  cv2.waitKey( 0 )
  
  for x in [h, s, v]:
    cv2.imshow( "Test", x )
    cv2.waitKey( 0 )
  m = (img_mask_bin / img_mask_bin.max()).flatten()
  
  # Make Mask
  img_mask_bin = img_mask_bin.astype( np.int16 )
  img_mask_bin[img_mask_bin > 0] = 1
  img_mask_bin = img_mask_bin.astype( np.uint8 )
  
  for i, x in enumerate( [s, v] ):
    x = x.flatten()
    x_masked = x[m == 1]
    hist, bins = np.histogram( x_masked, 256, [0, 256] )
    # showGraph(hist)
    # showGraph(bins)
    cdf = hist.cumsum()
    # showGraph(cdf)
    cdf_normalized = cdf * hist.max() / cdf.max()
    # showGraph( cdf_normalized )
    
    cdf_m = np.ma.masked_equal( cdf, 0 )
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    
    cdf = np.ma.filled( cdf_m, 0 ).astype( 'uint8' )
    
    img_hsv[:,:,(i+1)] = cdf[x].reshape( img_rgb.shape[:2] )
    
  img_ret = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
  
  for i in range(3):
    img_ret[:,:,i] = img_ret[:,:,i] * img_mask_bin
    
  
  return img_ret

def equalizeHistWithMask( img_rgb, img_mask_bin ):
  assert len( img_rgb.shape ) == 3 and img_rgb.shape[2] == 3, "img_rgb must be 3-ch RGB Image !"
  assert len( img_mask_bin.shape ) == 2, "img_mask_bin must be 1-ch Binary Image !"
  b, g, r = [img_rgb[:, :, i] for i in range( 3 )]
  m = (img_mask_bin / img_mask_bin.max()).flatten()
  
  # Make Mask
  img_mask_bin = img_mask_bin.astype(np.int16)
  img_mask_bin[img_mask_bin > 0] = 255
  img_mask_bin = np.abs( (img_mask_bin - 255) ).astype( np.uint8 )
  
  equalized = np.zeros(img_rgb.shape, dtype=np.uint8)
  
  for i, x in enumerate([b, g, r]):
    x = x.flatten()
    x_masked = x[m == 1]
    hist, bins = np.histogram(x_masked, 256, [0, 256])
    # showGraph(hist)
    # showGraph(bins)
    cdf = hist.cumsum()
    # showGraph(cdf)
    cdf_normalized = cdf * hist.max() / cdf.max()
    # showGraph( cdf_normalized )
    
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    
    cdf = np.ma.filled(cdf_m, 0). astype('uint8')
    
    equalized[:,:,i] = cdf[x].reshape(img_rgb.shape[:2]) + img_mask_bin
    
  return equalized
  
    
if __name__ == '__main__':
    img = cv2.imread("img/resource/Roof_02.png", cv2.IMREAD_UNCHANGED)
    
    # equalized = equalizeHistWithMask( img[:,:,0:3], img[:,:,3] )
    equalized = equalizeHistByHSVWithMask( img[:,:,0:3], img[:,:,3] )
    cv2.imwrite("Equalized.png", equalized)
    
    # lab = cv2.cvtColor(equalized, cv2.COLOR_BGR2Lab)
    #
    # for i in range(lab.shape[2]):
    #   cv2.imshow("Test", lab[:,:,i])
    #   cv2.waitKey(0)

    cv2.imshow("Test", equalized)
    cv2.waitKey(0)
