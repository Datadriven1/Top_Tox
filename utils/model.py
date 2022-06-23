from utils.all_utils import *
import numpy as np
import csv
import os
import matplotlib.pyplot as plt

# Importing error metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score, accuracy_score, confusion_matrix, classification_report


def all_classification_model(model_list, x_train, y_train, x_test, y_test, data_used, dataset_name):
    predp_list = []
    for p in model_list:
        p.fit(x_train,y_train)
        print('Score of', p , 'is:' , p.score(x_train,y_train))
        predp=p.predict(x_test)
        print('accuracy_score:', accuracy_score(y_test,predp))
        predp_list.append(accuracy_score(y_test,predp))
        print('confusion_matrix: \n', confusion_matrix(y_test,predp))
        print('classification_report: \n', classification_report(y_test,predp))
        print('********************************************************************************************')
        print('\n')
        headers = ['Test accuracy',"weighted avg precision","weighted avg recall","weighted avg f1-score","True Positive","False Positive","False Negative","True Negative","Sensitivity score", "Specificity","model_name", "features_used"]
        t_s = p.score(x_train,y_train)
        test_s = accuracy_score(y_test,predp)
        c_r = classification_report(y_test,predp).split('\n')
        weighted_avg = c_r[7]
        w = weighted_avg.split('      ')
        tp, fp, fn, tn = confusion_matrix(y_test,predp).ravel()
        Sensitivity = tp/(tp+fn)
        Specificity = tn/(tn+fp)
        f = open(f"result/{dataset_name}.csv", "a", newline="")
        file_is_empty = os.stat(f"result/{dataset_name}.csv").st_size == 0
        tup1 = (test_s,w[1].replace(" ", ""),w[2].replace(" ", ""),w[3].replace(" ", ""),tp,fp,fn,tn,Sensitivity,Specificity,p, data_used)
        writer = csv.writer(f)
        if file_is_empty:
            writer.writerow(headers)
        writer.writerow(tup1)
        f.close()
    return predp_list





def single_classification_model(model_name, x_train, y_train, x_test, y_test):
    b_model = model_name(n_estimators=20000, criterion="friedman_mse", learning_rate=0.001)
    b_model.fit(x_train,y_train)
    print('score of:', model_name, "is" , b_model.score(x_train,y_train))
    b_model_pred = b_model.predict(x_test)
    print('\n')
    print('accuracy_score:', accuracy_score(y_test,b_model_pred))
    print('confusion_matrix: \n', confusion_matrix(y_test,b_model_pred))
    print('classification_report: \n', classification_report(y_test,b_model_pred))
    print('\n')
    return b_model_pred
