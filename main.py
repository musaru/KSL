import os
import time
import torch
import pickle
import numpy as np
import copy
import pandas as pd
import torch.nn.functional as F
from shutil import copyfile
from tqdm import tqdm
from torch.utils import data
from torch.optim.adadelta import Adadelta
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    LeaveOneOut,
)
from sklearn.metrics import classification_report
import timm
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from Graph import *
from Models import *

#device = 'cuda'
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("using", device, "device")

epochs = 100
batch_size = 256 #32

def train_valid_split_dataset(data_files, batch_size, test_size=0.2,used_key_points=None):
    features, labels = [], []
    for fil in data_files:
        with open(fil, 'rb') as f:
            fts, lbs = pickle.load(f)
            features.append(fts)
            labels.append(lbs)
        del fts, lbs
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    if used_key_points != None:
        features = features[:,:,:,used_key_points]
    print(features.shape)
    print(labels.shape)
    
    X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=test_size,random_state=0,shuffle=True,stratify=labels)
    train_set = data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),torch.tensor(y_train, dtype=torch.int64))
    valid_set = data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),torch.tensor(y_test, dtype=torch.int64))
    train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
    valid_loader = data.DataLoader(valid_set, batch_size)

    return [train_loader],[valid_loader]
    
def KFold_load_dataset(data_files, batch_size, split_size=0.2,used_key_points=None):#0.2
    """Load data files into torch DataLoader with/without spliting train-test.
    """
    features, labels = [], []
    for fil in data_files:
        with open(fil, 'rb') as f:
            fts, lbs = pickle.load(f)
            features.append(fts)
            labels.append(lbs)
        del fts, lbs
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    if used_key_points != None:
        features = features[:,:,:,used_key_points]
    print(features.shape)
    print(labels.shape)
    
    set_of_train_loader,set_of_valid_loader = [],[]

    if split_size > 0:
        if split_size < 1:
            n_splits = int(1/split_size)
        else:
            n_splits = split_size
        skf = StratifiedKFold(n_splits,random_state=42, shuffle=True)
        
        for train_index, test_index in skf.split(features,labels):
            train_set = data.TensorDataset(torch.tensor(features[train_index], dtype=torch.float32),torch.tensor(labels[train_index], dtype=torch.int64))
            valid_set = data.TensorDataset(torch.tensor(features[test_index], dtype=torch.float32),torch.tensor(labels[test_index], dtype=torch.int64))
            train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
            valid_loader = data.DataLoader(valid_set, batch_size)
            
            set_of_train_loader.append(train_loader)
            set_of_valid_loader.append(valid_loader)
            
    return set_of_train_loader, set_of_valid_loader

def LeaveOneSubject_load_dataset(data_files, batch_size,test_subject_id,used_key_points=None):#0.2
    """Load data files into torch DataLoader with/without spliting train-test.
    """
    features, labels, person_id= [], [], []
    for fil in data_files:
        with open(fil, 'rb') as f:
            fts, lbs, pid = pickle.load(f)
            features.append(fts)
            labels.append(lbs)
            person_id.append(pid)
            
        del fts, lbs, pid
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    person_id = np.concatenate(person_id, axis=0)
    
    if used_key_points != None:
        features = features[:,:,:,used_key_points]

    set_of_train_loader,set_of_valid_loader = [],[]
        
    train_set = data.TensorDataset(torch.tensor(features[person_id!=test_subject_id], dtype=torch.float32),torch.tensor(labels[person_id!=test_subject_id], dtype=torch.int64))
    valid_set = data.TensorDataset(torch.tensor(features[person_id==test_subject_id], dtype=torch.float32),torch.tensor(labels[person_id==test_subject_id], dtype=torch.int64))
    train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
    valid_loader = data.DataLoader(valid_set, batch_size)
    
    print(features[person_id!=test_subject_id].shape)
    print(features[person_id==test_subject_id].shape)
    
    return train_loader, valid_loader


def accuracy_batch(y_pred, y_true):
    # print(y_pred.shape,y_true.shape)
    # return (y_pred.argmax(1) == y_true.argmax(1)).mean()
    return (y_pred.argmax(1) == y_true).mean()


def set_training(model, mode=False):
    for p in model.parameters():
        p.requires_grad = mode
    model.train(mode)
    
    torch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable Parameters :{torch_total_params}')
    return model

save_folder = os.path.join(os.environ['HOME'],"KSL_V2/Outputs")
os.makedirs(save_folder,exist_ok=True)
used_key_points=[0,11,12,13,14]+[i for i in range(33,33+21)] + [i for i in range(54,54+21)] 


# KSL 77 public Dataset
# set_of_train_loader,set_of_valid_loader = KFold_load_dataset([os.path.join("../Datasets/KSL77_25_dataset.pkl")], batch_size,0.2,used_key_points)
set_of_train_loader,set_of_valid_loader = train_valid_split_dataset([os.path.join("../Datasets/KSL77_25_dataset.pkl")], batch_size,0.2,used_key_points)
num_class=77

for i,(train_loader,valid_loader) in enumerate(zip(set_of_train_loader,set_of_valid_loader)):

    
    graph_args = {'layout':'mediapipe_KSL','strategy': 'spatial'}
    
    model = ProposedTwoStreamSpatialTemporalGraph(graph_args, num_class).to(device)
    model_name = 'Proposed_model'
    # model = AblationExperimentModel1(graph_args, num_class).to(device)
    # model = AblationExperimentModel2(graph_args, num_class).to(device)
    # model = AblationExperimentModel3(graph_args, num_class).to(device)
    
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # losser = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    # losser = torch.nn.CrossEntropyLoss()


    loss_list = {'train': [], 'valid': []}
    accu_list = {'train': [], 'valid': []}
    best_acc = -1
    dataloader = {'train': train_loader, 'valid': valid_loader}

    for e in range(epochs):
        print('Epoch {}/{}'.format(e, epochs - 1))
        for phase in ['train', 'valid']:
            if phase == 'train':
                model = set_training(model, True)
                losser = timm.loss.LabelSmoothingCrossEntropy(smoothing=0.01)
            else:
                model = set_training(model, False)
                losser = torch.nn.CrossEntropyLoss()

            run_loss = 0.0
            run_accu = 0.0
            with tqdm(dataloader[phase], desc=phase) as iterator:
                for pts, lbs in iterator:
                    # Create motion input by distance of points (x, y) of the same node
                    # in two frames.
                    mot = pts[:, :, 1:, :] - pts[:, :, :-1, :]

                    mot = mot.to(device)
                    pts = pts.to(device)
                    lbs = lbs.to(device)

                    # Forward.
                    out = model((pts, mot))
                    #print(lbs)

                    #print(out)
                    loss = losser(out, lbs)

                    if phase == 'train':
                        # Backward.
                        model.zero_grad()
                        loss.backward()
                        optimizer.step()

                    run_loss += loss.item()
                    accu = accuracy_batch(out.detach().cpu().numpy(),
                                          lbs.detach().cpu().numpy())
                    run_accu += accu

                    iterator.set_postfix_str(' loss: {:.4f}, accu: {:.4f}'.format(
                        loss.item(), accu))
                    iterator.update()
                    #break
            loss_list[phase].append(run_loss / len(iterator))
            accu_list[phase].append(run_accu / len(iterator))

        if(best_acc < accu_list['valid'][-1]) and phase =='valid':
            best_acc = accu_list['valid'][-1]
            bset_model = copy.deepcopy(model)


        if best_acc==1:
            print("Early Stop beacause of reaching the accuracy to 100%")
            break

        print('Summary epoch:\n - Train loss: {:.4f}, accu: {:.4f}\n - Valid loss:'
              ' {:.4f}, accu: {:.4f}'.format(loss_list['train'][-1], accu_list['train'][-1],
                                             loss_list['valid'][-1], accu_list['valid'][-1]))
    print(best_acc)
    
    torch.save(bset_model.state_dict(), os.path.join(save_folder, f'{model_name}_{i+1}of{len(set_of_train_loader)}_{best_acc:.4f}_KSL77.pth'))
    




    pred,label = [],[]
    model = bset_model
    for phase in ['valid']:
        if phase == 'train':
            model = set_training(model, True)
        else:
            model = set_training(model, False)

        run_loss = 0.0
        run_accu = 0.0
        with tqdm(dataloader[phase], desc=phase) as iterator:
            for pts, lbs in iterator:
                # Create motion input by distance of points (x, y) of the same node
                # in two frames.
                mot = pts[:, :, 1:, :] - pts[:, :, :-1, :]

                mot = mot.to(device)
                pts = pts.to(device)
                lbs = lbs.to(device)

                # Forward.
                out = model((pts, mot))
                #print(lbs)

                #print(out)
                loss = losser(out, lbs)

                if phase == 'train':
                    # Backward.
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                run_loss += loss.item()
                accu = accuracy_batch(out.detach().cpu().numpy(),
                                      lbs.detach().cpu().numpy())


                pred = pred + out.detach().cpu().numpy().argmax(1).tolist()
                label = label +lbs.detach().cpu().numpy().tolist()
                run_accu += accu

                iterator.set_postfix_str(' loss: {:.4f}, accu: {:.4f}'.format(loss.item(), accu))
                iterator.update()
                #break
        loss_list[phase].append(run_loss / len(iterator))
        accu_list[phase].append(run_accu / len(iterator))
        #print(accu_list)
        #print(torch.max(accu_list))
        #break

    print('Summary epoch:\n - Valid loss:'
          ' {:.4f}, accu: {:.4f}'.format(loss_list['valid'][-1], accu_list['valid'][-1]))
    print(best_acc)

    print(classification_report(label, pred,digits=5))
    report=classification_report(label, pred, digits=5,output_dict=True)
    report_df = pd.DataFrame(report).T
    
    report_df.to_csv(os.path.join(save_folder, f'{model_name}_{i+1}of{len(set_of_train_loader)}_{best_acc:.4f}_KSL77_report.csv'))
   
