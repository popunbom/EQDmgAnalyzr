from skimage.feature import greycomatrix, greycoprops
import numpy as np
import cv2
import CommonProcedures as cp

GLCM_DISTANCE = 1
GLCM_DEGREE   = 45

def calcGLCMFeatures( img_src, npy_label, feature_names, csvSave=False ):
   for feature_name in feature_names:
      
      print( "Calculating '%s' ... " % feature_name, end="", flush=True )
      
      tmp = np.zeros( (len( npy_label )), dtype=np.float32 )
      
      for i in range( 1, len( npy_label ) ):
         P = cp.getRect(img_src.shapem, npy_label[i])
         img = cv2.cvtColor(src_img[P[0]:P[1], P[2]:P[3]], cv2.COLOR_BGR2GRAY)
         glcm = greycomatrix(img, [GLCM_DISTANCE], [GLCM_DEGREE])
         glcm_feature = greycoprops(glcm, feature_name)
         tmp[i] = glcm_feature[0, 0]
      
      tmp /= tmp.max()
      
      if (csvSave):
         file_name = str.format( "data/glcm_%s.csv" % feature_name )
         np.savetxt( file_name, tmp, delimiter=',', fmt='%.8f' )
      else:
         file_name = str.format( "data/glcm_%s.npy" % feature_name )
         np.save( file_name, tmp )
      
      print( "done! saved as \"%s\"" % file_name, flush=True )


if __name__ == '__main__':
   src_img = cv2.imread( "img/aerial_only.png", cv2.IMREAD_COLOR )
   mask_img = cv2.imread( "img/mask_invert.png", cv2.IMREAD_GRAYSCALE )
   label_data = np.load( "data/label.npy" )
   feature_names = ['contrast', 'dissimilarity', 'ASM', 'correlation']
   
   src_img = cv2.bilateralFilter( src_img, 10, 5.90, 33.80 )
   
   calcGLCMFeatures( src_img, label_data, feature_names, False )
   
   # for feature in feature_names:
   #    file_name = str.format( "data/glcm_%s.npy" % feature )
   #
   #    feature_data = np.load( file_name )
   #
   #    quantize_data = sr.percentileModification( feature_data, 10 )
   #
   #    data_img = sr.applyResultByColor( src_img, quantize_data, label_data )
   #    dst_img = sr.createOverwrappedImage( src_img, data_img, mask_img )
   #
   #    out_file = str.format("img/result_glcm_%s.png" % feature)
   #
   #    cv2.imwrite(out_file, dst_img)
