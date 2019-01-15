# Edge Feature Analyzer using K-Means(scikit-learn)

import numpy as np
import pprint
from sklearn.cluster import KMeans, MeanShift
import matplotlib.pyplot as plt

# Global Values
FEAT_NAMES = ['endpoints', 'branches', 'passings', 'length']
CLS_NAMES = ['Collapsed', 'Half-Collapsed', 'Non-Collapsed']

TARGET_NAME = ""

N_OF_CLASS = 3



def classLabeling(npy_data, km_predict, n_of_class=3):
  assert (len(npy_data) == len(km_predict)), "npy_data and km_predict must be same length"
  assert (np.max(km_predict) == n_of_class - 1), "invalid values: n_of_class=%d" % n_of_class
  
  order = ['asec', 'asec', 'desc', 'desc']
  
  # pprint.pprint(npy_data)
  
  # Make Index Filter
  filt = [ [] for _ in range(n_of_class) ]
  for (i, x) in enumerate(km_predict):
    filt[x].append(i)
    
  result = { k:[ 0.0 for _ in range(n_of_class) ] for k in FEAT_NAMES }
  
  # Summation
  for (ii, feat) in enumerate(FEAT_NAMES):
    for (i, xx) in enumerate(filt):
      for x in xx:
        result[feat][i] += npy_data[x][ii]
  
  # Calc Average
  for vv in result.values():
    for (i, x) in enumerate(vv):
      vv[i] = x / len(filt[i])
  
  
  # Calc Scores
  scores = np.zeros( (n_of_class), dtype=np.float32 )
  for (ii, feat) in enumerate(FEAT_NAMES):
    if order[ii] == 'asec':
      for (i, j) in enumerate(np.argsort(result[feat])):
          scores[j] += 1.0 / (i + 1)
    elif order[ii] == 'desc':
      for (i, j) in enumerate(np.argsort(result[feat])[::-1]):
          scores[j] += 1.0 / (i + 1)
  
  # Make Dict
  # dict_ret = { k:0 for k in CLS_NAMES }
  # for (k, v) in zip(CLS_NAMES, np.argsort(scores)):
  #   dict_ret[k] = v
  dict_ret = {}
  for (k, v) in zip(CLS_NAMES, np.argsort(scores)):
    dict_ret.update({v:k})
  
  return dict_ret


def analyze(npy_data, list_fileNum, n_of_cls):
  km_predict = KMeans(n_clusters=n_of_cls).fit_predict(npy_data)
  # km_predict = MeanShift().fit_predict(npy_data)
  
  result = [[] for _ in range(km_predict.max() + 1)]
  
  for (i, x) in enumerate(km_predict):
    result[x].append(list_fileNum[i])
  
  
  for (i, x) in enumerate(result):
    print("Categorized %d : " % i, x)
  
  return result, km_predict


def showImgs(list_fileNum, title, IMGS_PER_LINE=5, noShowWnd=False):
  wnd = plt.figure()
  wnd.suptitle(title)
  for (idx, img_name) in enumerate(list_fileNum):
    img = plt.imread('img/divided/{0}/{1:05d}.png'.format(TARGET_NAME, img_name))
    # print(int(len(x) // IMGS_PER_LINE))
    sblts = plt.subplot(IMGS_PER_LINE, len(list_fileNum) // IMGS_PER_LINE + 1, idx + 1)
    sblts.tick_params(labelbottom="off", bottom="off")
    sblts.tick_params(labelleft="off", left="off")
    sblts.set_xticklabels([])
    plt.imshow(img)
    
    # if not noShowWnd:
    #   plt.show()
    # else:
    #   return plt


def clasify(target_name, n_of_pass=1, autoClassify=False):
  global TARGET_NAME
  TARGET_NAME = target_name
  
  npy_src = np.load('data/{0}_edge_feat.npy'.format(TARGET_NAME))
  
  # Modifying Data Structure
  npy_src = np.array([[x[1][name] for name in FEAT_NAMES] for x in npy_src], dtype=np.float64)
  
  
  # Init
  list_fileNum = [i for i in range(1, len(npy_src) + 1)]
  final_result = {n: [] for n in CLS_NAMES}
  npy_data = npy_src
  passes = 0
  
  # START :Main Routine
  for passes in range(n_of_pass):
    if (passes > 0):
      if (len(final_result['Half-Collapsed']) == 0):
        break
      npy_data = []
      for x in final_result['Half-Collapsed']:
        npy_data.append(npy_src[x - 1].tolist())
      
      npy_data = np.array(npy_data)
      list_fileNum = final_result['Half-Collapsed']
      final_result['Half-Collapsed'] = []
    
    result, km_predict = analyze(npy_data, list_fileNum, n_of_cls=N_OF_CLASS)
    
    if autoClassify:
      pred_label = classLabeling(npy_data, km_predict, N_OF_CLASS)
      for (i, x) in enumerate( result ):
        showImgs( x, pred_label[i] )
      plt.show()
      for (k, v) in pred_label.items():
        final_result[v] += result[k]
      
    else:
      print("[Number] 0:Collapsed, 1:Half-Collapsed, 2:Non-Collapsed")
      for (i, x) in enumerate( result ):
        showImgs( x, "Class %d" % i )
      plt.show()
      for i in range(N_OF_CLASS):
        clsNum = int(input("Class %d is : " % i))
        final_result[CLS_NAMES[clsNum]] += result[i]
    
    passes += 1
  # END   :MAIN ROUTINE
  
  if passes > 1:
    for cls_name in CLS_NAMES:
      showImgs(final_result[cls_name], cls_name, noShowWnd=True)
    plt.show()
  
  npy_output = [0.0 for _ in range(len(npy_src))]
  # 0:Collapsed, 1:Half-Collapsed, 2:Non-Collapsed
  # ↓
  # 1:Collapsed, 2:Half-Collapsed, 3:Non-Collapsed
  for (idx, cls_name) in enumerate(CLS_NAMES):
    for x in final_result[cls_name]:
      npy_output[x - 1] = float(idx + 1)
  
  npy_output = np.array(npy_output)
  
  # Data Mod for ResultImage
  # 0.30: Collapsed, 0.60:Half-Collapsed, 0.90:Non-Collapsed
  npy_output = np.append(np.array([0.0]), npy_output)
  npy_output = npy_output * 0.30
  
  # pprint.pprint(npy_output)
  
  np.save("data/final_result_{0}.npy".format(TARGET_NAME), npy_output)
  return npy_output


def accuAsses(npy_system, npy_answer):
  assert ( npy_system.shape == npy_answer.shape ), "must be same shape !"
  
  npy_em = np.zeros( (3, 3), dtype=np.int64 )
  
  # 0.30: Collapsed, 0.60:Half-Collapsed, 0.90:Non-Collapsed
  # ↓
  # 0: Collapsed, 1: Half-Collapsed, 2: Non-Collapsed
  npy_system = np.rint(npy_system / 0.30).astype(np.int32) - 1
  npy_answer = np.rint(npy_answer / 0.30).astype(np.int32) - 1
  
  for (s, a) in zip(npy_system, npy_answer):
    npy_em[s, a] += 1
  
  return npy_em
  

if __name__ == '__main__':
  # TARGET_NAME = 'aerial_roi1_customblur'
  # clasify(TARGET_NAME)
  npy_system = np.load("data/final_result_expr_2.npy")
  npy_answer = np.load("data/answer_expr_2.npy")
  npy_em = accuAsses(npy_system, npy_answer)
  print(npy_em)
