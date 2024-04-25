# implemnet jaccard_metric function in utilities.py from scratch
import numpy as np
import cv2
from sklearn.metrics import jaccard_score

def jaccard_metric(y_true, y_pred):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    print(y_true)
    print(y_pred)
    # TP=np.sum(y_pred==y_true)
    jaccard_index= jaccard_score(y_true, y_pred,pos_label=1,zero_division=1)
    return jaccard_index
    # tp = np.sum(y_true*y_pred)
    # fn = y_true-y_pred
    # fn[fn<0]=0
    # fn = np.sum(fn)
    # tn = y_true+y_pred
    # tn[tn>0]=-1
    # tn[tn>=0]=1
    # tn[tn<0]=0
    # tn = np.sum(tn)
    # fp = y_pred - y_true
    # fp[fp<0] = 0
    # fp = np.sum(fp)
    
    # return tp, fp, tn, fn

# # test the function
# result_img = cv2.imread("E:/College/Fourth Year/second term/RSSI/Project/Change-Detection/classical_technique/white.png")
# label_img = cv2.imread("E:/College/Fourth Year/second term/RSSI/Project/Change-Detection/classical_technique/black.png")
# result_img = cv2.imread("E:/College/Fourth Year/second term/RSSI/Project/Change-Detection/trainval/label/0069.png")
# label_img = cv2.imread("E:/College/Fourth Year/second term/RSSI/Project/Change-Detection/trainval/label/0000.png")
y_true = np.array([[0, 1, 0],
                   [0, 0, 0]])
y_pred =  np.array([[1, 0, 0],
                   [0, 0, 0]])
# TP, FP, TN, FN = jaccard_metric(y_true, y_pred)
# iou = TP/(FN+TP+FP+1e-8)
iou = jaccard_metric(y_true, y_pred)
print("Mean Jaccard Index:", iou)