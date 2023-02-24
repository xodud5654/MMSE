# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 20:36:20 2023

@author: admin
"""

###############################################################
###########################필요모듈 로드##########################
###############################################################

from numpy import random
import pandas as pd
import numpy as np
import torch, gc
import torch.nn as nn
import torchvision.transforms as T
import os
from codeclass import file_load, ECGDataset,evaluate_test,ensemble,stacking
from torchvision import models


def train(model, train_loader, optimizer, log_interval):
    
    model.train()
    
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.squeeze().to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
                epoch, batch_idx * len(label), 
                len(train_loader.dataset), 100.*batch_idx / len(train_loader), 
                loss.item()))
    return model



def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device

device = get_device()
print(device)


random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
   


transform = T.Compose([
    T.ToTensor(),
    T.Resize(size = (224, 224))
])

BATCH=32
EPOCHS = 30
num_classes = 7

single_ensemble_result, single_stacking_result = [], []
multi_ensemble_result, multi_stacking_result = [], []

re_path = f"{num_classes}C_result_resnet50"
e_s = f"{num_classes}C_ensemble_stacking_resnet50"

for LR in [1e-4, 5e-5, 1e-5]: 
    for image in ["gray", "scalo"]: 
        NAME = f"D:/7class_4class_ensemble/{num_classes}C_dataset/{image}"
        result = []
        for i in range(1,13,1):
            gc.collect()
            torch.cuda.empty_cache()
            
            pre_model = models.resnet50(pretrained=True)
            pre_model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)  
            model = pre_model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr = LR)
            criterion = nn.CrossEntropyLoss()
    
            train_datas, train_labels = file_load(NAME, f"lead_{i}","train", num_classes)
            train_dataset = ECGDataset(train_datas, train_labels,transform)
            train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                        batch_size = BATCH,
                                                        shuffle = True)
            
            
            val_datas, val_labels = file_load(NAME, f"lead_{i}","val", num_classes)
            val_dataset = ECGDataset(val_datas, val_labels,transform)
            val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                        batch_size = BATCH,
                                                        shuffle = False)
            
            
            test_datas, test_labels = file_load(NAME, f"lead_{i}","test", num_classes)
            test_dataset = ECGDataset(test_datas, test_labels,transform) 
            test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                        batch_size = BATCH,
                                                        shuffle = False)
            
            best_acc=0
            for epoch in range(1, EPOCHS + 1):
                
                model = train(model, train_loader, optimizer, log_interval = 50)
                test_loss, test_accuracy = evaluate_test(model, val_loader, device, BATCH, num_classes, PPL=False)
    
                if test_accuracy>best_acc :
                    best_model = model
                    best_acc = test_accuracy
                    
            
            os.makedirs(f"{re_path}/{image}/{LR}/model",exist_ok=True)
            os.makedirs(f"{re_path}/{image}/{LR}/PPL/T",exist_ok=True)
            os.makedirs(f"{re_path}/{image}/{LR}/PPL/t_v",exist_ok=True)
            
            
            torch.save(best_model, f"{re_path}/{image}/{LR}/model/lead{i}.pt")
            
            
            test_PPL, test_result = evaluate_test(best_model, test_loader, device, BATCH, num_classes, PPL=True)  
            result.append(test_result)
            test_PPL.to_csv(f"{re_path}/{image}/{LR}/PPL/T/lead{i}.csv",index=False)
            
            t_v_datas = train_datas + val_datas   
            t_v_labels = train_labels + val_labels  
            t_v_dataset = ECGDataset(t_v_datas, t_v_labels,transform)
            t_v_loader = torch.utils.data.DataLoader(dataset = t_v_dataset,
                                                        batch_size = BATCH,
                                                        shuffle = False)
            
            t_v_PPL, _ = evaluate_test(best_model, t_v_loader, device, BATCH, num_classes, PPL=True)  
            t_v_PPL.to_csv(f"{re_path}/{image}/{LR}/PPL/t_v/lead{i}.csv",index=False)
            
            
            test_total_result = pd.DataFrame(result,columns=["acc","recall","prec","roc","f1"])
        
        test_total_result.to_csv(f"{re_path}/{image}/test_total_result_{LR}.csv",index=False)
        
        #single data simple_averaging & stacking
        single_ensemble_result.append(ensemble(f"{re_path}/{image}/{LR}/PPL/T")) #simple_averaging
        single_stacking_result.extend(stacking(f"{re_path}/{image}/{LR}/PPL/t_v",f"{re_path}/{image}/{LR}/PPL/T", num_classes)) #stacking
        
    #multi data simple_averaging & stacking
    multi_ensemble_result.append(ensemble(f"{re_path}/*/{LR}/PPL/T"))
    multi_stacking_result.extend(stacking(f"{re_path}/*/{LR}/PPL/t_v",f"{re_path}/*/{LR}/PPL/T", num_classes))

#result DataFrame
E_columns = ["method","lr","name","acc","recall","pre","auc","f1"]
S_columns = ["method","lr","name","acc","recall","pre","auc","f1","params"] #method,lr,name,acc,recall,pre,auc,f1,params
single_ensemble_result = pd.DataFrame(single_ensemble_result,columns=E_columns)
multi_ensemble_result = pd.DataFrame(multi_ensemble_result,columns=E_columns)

single_stacking_result = pd.DataFrame(single_stacking_result,columns=S_columns)
multi_stacking_result = pd.DataFrame(multi_stacking_result,columns=S_columns)


# result save
os.makedirs(f"{e_s}",exist_ok=True)
single_ensemble_result.to_csv(f"{e_s}/single_ensemble_result.csv",index=False)
multi_ensemble_result.to_csv(f"{e_s}/multi_ensemble_result.csv",index=False)

single_stacking_result.to_csv(f"{e_s}/single_stacking_result.csv",index=False)
multi_stacking_result.to_csv(f"{e_s}/multi_stacking_result.csv",index=False)


