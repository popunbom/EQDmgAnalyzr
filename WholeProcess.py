import cv2
import EdgeProfiler
import MaskLabeling
import SaveResultAsImage
import FeatureAnalyzer
import imgproc.utils

features = ['length', 'endpoints', 'branches', 'passings']

# target_name = os.path.splitext(os.path.basename(path_img_src))[0]
target_name = "expr_2"


if __name__ == '__main__':
  path_img_src = f"img/resource/{target_name}_aerial.png"
  path_img_mask = f"img/resource/{target_name}_mask.png"
  divided_dir = "img/divided/" + target_name
  
  print("    img: ", path_img_src)
  print("   mask: ", path_img_mask)
  print("divided: ", divided_dir)
  
  ########################
  ##### RESOURCE LOAD ####
  ########################
  img_src = cv2.imread(path_img_src, cv2.IMREAD_COLOR)
  img_mask = cv2.imread(path_img_mask, cv2.IMREAD_GRAYSCALE)
  
  # Get Mask Data
  npy_label = MaskLabeling.getMaskLabel( img_mask )
  
  ########################
  # IMAGE PRE-PROCESSING #
  ########################
  img = imgproc.utils.pre_process( img_src )
  
  ########################
  #### IMAGE DIVISION ####
  ########################
  n_of_imgs = imgproc.utils.divide_by_mask( img, npy_label, target_name )
  
  #########################
  ## CALC EDGE FEATURES  ##
  #########################
  edge_features = EdgeProfiler.saveFeatureAsNPY(target_name, n_of_imgs, SHOW_LOG=True)
  # edge_features = np.load("data/"+target_name+"_edge_feat.npy")
  
  #########################
  #### REGION CLASSIFY ####
  #########################
  npy_result = FeatureAnalyzer.clasify( target_name, n_of_pass=2, autoClassify=False )
  # npy_result = np.load("data/final_result_"+target_name+".npy")


  ################################
  ## CREATE RESULT IMAGE (EACH) ##
  ################################
  # for feature in features:
  #   dst_img = SaveResultAsImage.saveEdgeFeatureAsImage(img, img_mask, edge_features, feature,
  #                                                      SHOW_LOG=True, SHOW_IMG=False)
  #
  #   cv2.imwrite("img/result/result_{0}_{1}.png".format(target_name, feature), dst_img)
  
  ################################
  ## CREATE RESULT IMAGE (COMB) ##
  #################################
  dst_img = SaveResultAsImage.createResultImg( img, npy_result, npy_label, raibowColor=True )
  dst_img = SaveResultAsImage.createOverwrappedImage( img_src, dst_img, doGSConv=True )

  cv2.imshow( "Test", dst_img )
  cv2.imwrite( "img/result/final_result_{0}.png".format( target_name ), dst_img )
  cv2.waitKey( 0 )
  
  
  # Classify: feature threshold
  # #  npy_thresholds = { 'feat_name', {'type: 0 or 1, th:[]} }:
  # # if type == 0
  # ##     0.0 < p < th[0]: Not Collapsed
  # ##   th[1] < p < th[1]: Collapsed roughly
  # ##   th[1] < p <   1.0: Collapsed completely
  # # elif type == 1
  # ##     0.0 < p < th[0]: Collapsed completely
  # ##   th[1] < p < th[1]: Collapsed roughly
  # ##   th[1] < p <   1.0: Not Collapsed
  # thresh = {'length'   : {'type': 1, 'th': [0.26, 0.76]},
  #           'endpoints': {'type': 1, 'th': [0.23, 0.60]},
  #           'branches' : {'type': 0, 'th': [0.33, 0.70]},
  #           'passings' : {'type': 0, 'th': [0.45, 0.71]}
  #           }
  # img_dst = SaveResultAsImage.clasify( img_src, img_mask, edge_features, thresh, SHOW_LOG=True, SHOW_IMG=False)
  # cv2.imwrite(str.format("img/result/result_%s_%s.png" % (target_name, "classified")), img_dst)
  
