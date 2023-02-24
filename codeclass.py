# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 20:36:07 2023

@author: admin
"""
# =============================================================================
import glob
import os
import cv2
from tqdm import tqdm

def file_load(NAME,lead,typ, num_classes):
    image_datas = []
    labels = []

    for data_path in sorted(glob.glob(f"{NAME}\{lead}\{typ}\*")):
        print(data_path)
        label = data_path.split("\\")[-1]
        
        if label == "AF": lb = [0]
        elif label == "AFIB" :lb = [1]
        elif label == "SA": lb = [2]
        elif label == "SB": lb = [3]
        elif label == "SR": lb = [4]
        elif label == "ST": lb = [5]
        elif label == "SVT": lb = [6]
        
        for img_png in tqdm(sorted(glob.glob(os.path.join(data_path, "*.png")))):
            image = cv2.imread(img_png)
            image_datas.append(image)
            labels.append(lb)
    return image_datas, labels

# =============================================================================
from torch.utils.data import Dataset
class ECGDataset(Dataset):
    def __init__(self, image_datas, labels, transform=None):
        self.image_datas = image_datas
        self.y = labels
        self.transform = transform

         
    def __len__(self):
        return (len(self.image_datas))
    
    def __getitem__(self, i):
        data = self.image_datas[i]
        if self.transform:
            data = self.transform(data)
            
        if self.y is not None:
            return (data, self.y[i][0])
# =============================================================================
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score,recall_score,precision_score,roc_auc_score,f1_score, confusion_matrix


def evaluate_test(model, test_loader, device, BATCH, num_classes, PPL=False):
    criterion = nn.CrossEntropyLoss()
    print("model testing...")
    labels = []
    preds = []
    probs = []
    model.eval()
    
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for image, label in tqdm(test_loader):
            a = label.cpu().data.tolist()
            labels.append(a)
            image = image.squeeze().to(device)
            label = label.to(device)
            
            output = model(image)
            
            test_loss += criterion(output, label).item()
            probility = nn.Softmax(dim=1)

            prediction = output.max(1, keepdim = True)[1]
            preds.append(prediction.cpu().data.tolist())
            probs.append(probility(output).cpu().data.tolist())
            correct += prediction.eq(label.view_as(prediction)).sum().item()
    labels = sum(labels, [])
    preds = sum(preds, [])
    probs = sum(probs, [])
    preds = [i[0] for i in preds]
    test_loss /= (len(test_loader.dataset) / BATCH)
    test_accuracy = round(100.*correct / len(test_loader.dataset),2)
    print("\n predicted :  \tLoss: {:.4f}, \tAccuracy: {:.2f}% \n".format(
            test_loss, test_accuracy))
    if PPL==True:
        Li = []
        for j in range(num_classes):
            Li.append([probs[i][j] for i in range(len(probs))])
            
        df_prob = pd.DataFrame(Li).T
            
        df_PPL = pd.concat([pd.DataFrame(labels), df_prob, pd.DataFrame(preds)], axis=1)
        result = [round(accuracy_score(labels,preds)*100,2),
                               round(recall_score(labels, preds,average="weighted"),3),
                               round(precision_score(labels, preds,average="weighted"),3),
                               round(roc_auc_score(labels, probs,multi_class="ovr"),3),
                               round(f1_score(labels, preds,average="weighted"),3)]
        
        return df_PPL, result
    else:
        return test_loss, test_accuracy
    
# =============================================================================

import numpy as np

def cal_LPP(df):
    probs = [np.array(df.iloc[i,1:-1]) for i in range(len(df))]
    probs = np.array(probs)
    preds = df.iloc[:,-1]
    labels = df.iloc[:,0]
    
    return labels, preds, probs

def softmax(x):
    f_x = np.exp(x)/np.sum(np.exp(x))
    return f_x


def ensemble(PATH):
    print("\nsimple averaging...")
    prob_list = []
    result = []
    for path in sorted(glob.glob(f"{PATH}/*.csv")):
        df = pd.read_csv(path)
        labels, preds, probs = cal_LPP(df)
        prob_list.append(probs)
    preds, probs = [], []
    
    for i in range(0,len(df)):
        base = 0
        for j in range(0,len(prob_list)):
            base += np.array(prob_list[j][i])
        preds.append(np.argmax(softmax(base/float(len(prob_list)))))
        probs.append(softmax(base/float(len(prob_list))))
    if len(prob_list)==12:
        method = PATH.split("/")[1]
    else:
        method = "multi"
    print("data Dimension :\t{}    \tdata method : \t{}".format(len(prob_list), method))
    
    #method,lr,name,acc,reca,,pre,auc,f1
    result = [method,PATH.split("/")[2],"simple_averaging",round(accuracy_score(labels,preds)*100,2),
                            round(recall_score(labels, preds,average="weighted"),3),
                                round(precision_score(labels, preds,average="weighted"),3),
                                    round(roc_auc_score(labels, probs,multi_class="ovr",average="weighted"),3),
                                        round(f1_score(labels, preds,average="weighted"),3)]
    return result

# =============================================================================
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold   
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def data_xx_d(path_o):
    print("dimension loading...")
    dataset = pd.DataFrame()
    for n, path in enumerate(glob.glob(f"{path_o}/*.csv")):
        df = pd.read_csv(path)
        if n==0:
            dataset = df.iloc[:,1:-1]
            
        else:
            dataset = pd.concat([dataset, df.iloc[:,1:-1]],axis=1)
    dataset.columns = [str(i) for i in range(0,len(dataset.iloc[0,:]))]

    return dataset, df.iloc[:,0]

def stacking(PATH,test_PATH, num_classes):
    if num_classes == 7:
        s = 84
    elif num_classes ==4:
        s=48
    else:
        print("num_classes error")
        return

    print("\nstacking...")
    result = []
    X, y = data_xx_d(PATH)
    
    if len(X.iloc[0,:])==s:
        method = PATH.split("/")[1]
    else:
        method = "multi"
    print("data Dimension :\t{}    \tdata method : \t{}".format(len(X.iloc[0,:]), method))
    
    X_test, y_test = data_xx_d(test_PATH)
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    #models
    lr = LogisticRegression(random_state=42,n_jobs=-1)
    lr_params = [{'C' : [1e-3,1e-2,1e-1,1,10,100,1000]}]
    
    linear = SVC(random_state=42,kernel="linear",probability=True)
    linear_params = [{'C' : [1e-3,1e-2,1e-1,1,10,100,1000]}]
    
    rbf = SVC(random_state=42,probability=True,kernel="rbf")
    rbf_params = [{'C' : [1e-3,1e-2,1e-1,1,10,100,1000]},
                  {'gamma' : [1e-3,1e-2,1e-1,1,10,100,1000]}]
    
    rf = RandomForestClassifier(random_state=42,n_jobs=-1)
    rf_params = [{'n_estimators' : [100,200,300,500,1000,2000,3000]},
                 {'max_features' : ["sqrt","log2"]},
                 {'max_depth' : [5,10,15,20,None]}]
    
    xgb = XGBClassifier(random_state=42)
    xgb_params = {'n_estimators': [100,300,500,1000],
                      "max_depth": [3, 5, 7, 9],
                          "learning_rate": [0.1, 0.05, 0.01]}

    for model,param,name in [[lr,lr_params,"lr"],[linear,linear_params,"linear"],[rbf,rbf_params,"rbf"],[rf,rf_params,"rf"],[xgb,xgb_params,"xgb"]]:
        grid_cv = GridSearchCV(estimator = model,
                                param_grid = param,
                                scoring='accuracy',
                                cv = kfold,
                                refit = True,
                                n_jobs = -1)
        grid_cv.fit(X, y)
        print(f'{name} :')
        print('best validation score: %.3f' %grid_cv.best_score_)
        print(grid_cv.best_params_)
        
        bestModel = grid_cv.best_estimator_
        

        X_t_pred = bestModel.predict(X_test)
        X_t_prob = bestModel.predict_proba(X_test)
        
        #method,lr,name,acc,recall,pre,auc,f1,params
        result.append([method,PATH.split("/")[2],"stacking",name,
                        round(accuracy_score(y_test,X_t_pred)*100,2),
                            round(recall_score(y_test, X_t_pred,average="weighted"),3),
                                round(precision_score(y_test, X_t_pred,average="weighted"),3),
                                    round(roc_auc_score(y_test, X_t_prob,multi_class="ovr",average="weighted"),3),
                                        round(f1_score(y_test, X_t_pred,average="weighted"),3),
                                            grid_cv.best_params_])
        os.makedirs(f"{num_classes}C_ensemble_stacking_resnet50/ML_model",exist_ok=True)
        joblib.dump(bestModel, f'{num_classes}C_ensemble_stacking_resnet50/ML_model/{method}_{PATH.split("/")[2]}_{name}.pkl') 
        
        LR = PATH.split("/")[2]
        os.makedirs(f"{num_classes}C_ensemble_stacking_resnet50_v2/confusion_matrix/{method}",exist_ok=True)
    
        
        LABELS = ["AF","AFIB","SA","SB","SR","ST","SVT"]
        sns.heatmap(confusion_matrix(y_test, X_t_pred),annot=True,fmt = "d",cbar=False,xticklabels = LABELS, yticklabels = LABELS)
        plt.title(f"{method}_{LR}_{name}_n")
        plt.savefig(f"D:/7class_4class_ensemble/{num_classes}C_ensemble_stacking_resnet50_v2/confusion_matrix/{method}/{method}_{LR}_{name}_n.png")
        plt.close()
        sns.heatmap(confusion_matrix(y_test, X_t_pred,normalize="true")*100,annot=True,fmt = ".2f",cbar=False,xticklabels = LABELS, yticklabels = LABELS)
        plt.title(f"{method}_{LR}_{name}_per")
        plt.savefig(f"D:/7class_4class_ensemble/{num_classes}C_ensemble_stacking_resnet50_v2/confusion_matrix/{method}/{method}_{LR}_{name}_per.png")
        plt.close()
    return result


