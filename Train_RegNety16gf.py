import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim as optim


class OnePathologyDataset(Dataset):
    def __init__(self, labels, id_to_path, pathology='No Finding', remote=True):
        # labels is a (n_samples x 3) one-hot encoded numpy array...
        # with columns representing classifications -1, 0, 1
        # id_to_path is a dictionary mapping ids to image path names
        # imdir_path is the path to the directory of all downsampled images
        # pathology is the pathology we are trying to classify
        # remote specifies if this is running locally or on the hpc
        self.labels = labels
        self.id_to_path = id_to_path
        self.remote = remote
        if remote:
            self.imdir_path = "/groups/CS156b/2023/BbbBbbB/Train_200x200"
        else:
            self.imdir_path = "D:\\cs156\\images_200x200"
        self.pathology = pathology
    
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        indx = list(self.id_to_path.keys())[idx]
        impath = os.path.join(self.imdir_path, self.id_to_path[indx])
        im = np.load(impath)
        return torch.from_numpy(im), torch.tensor(self.labels.iloc[idx]).type(torch.long)


if __name__ == '__main__':
    print('Running Train_RegNety16gf.py', file=open('Train_RegNet.out', 'a'))
    
    tic = time.time()

    pathologies = ['No Finding', 'Enlarged Cardiomediastinum','Cardiomegaly',
                   'Lung Opacity','Pneumonia','Pleural Effusion',
                   'Pleural Other','Fracture','Support Devices']

    remote = True
    
    if remote:
        labels = pd.read_csv('/central/groups/CS156b/2023/BbbBbbB/Labels_Onehot.csv', index_col='id')
    else:
        labels = pd.read_csv('./Labels_Onehot.csv', index_col='id')

    if not remote:
        idxs = []
        for i, pth in enumerate(labels['Path200x200']):
            if i != labels.shape[0]-1:
                if int(pth[3:8]) <= 20:
                    idxs.append(i)
        labels = labels.iloc[idxs]

    id_to_path = {}
    for idx in labels.index:
        id_to_path[idx] = labels['Path200x200'][idx]
        
        
    batch_size = 8
    n_epochs = 10
    

    for pathology in pathologies:
        print(f'Starting "{pathology}" training!', file=open('Train_RegNet.out', 'a'))

        
        if remote:
            data = OnePathologyDataset(labels[[pathology+' -1', pathology+' 0', pathology+' 1']],
                                       id_to_path, pathology=pathology, remote=True)
        elif not remote:
            data = OnePathologyDataset(labels[[pathology+' -1', pathology+' 0', pathology+' 1']],
                                       id_to_path, pathology=pathology, remote=False)

        train_test = random_split(data, [int(len(data)*0.7), len(data)-int(len(data)*0.7)])

        train_loader = DataLoader(train_test[0], batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(train_test[1], batch_size=batch_size, shuffle=False)

        ###########################################################################################
        
        repo = 'pytorch/vision'
        model = torch.hub.load(repo, 'regnet_y_16gf', weights='ResNet50_Weights.IMAGENET1K_V1')
        
        ###########################################################################################
        
        for epoch in range(n_epochs):
            for im in train_loader:
                inputs, labels = im
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
             
        correct = 0
        total = 0
        with torch.no_grad():
            for im in test_loader:
                inputs, labels = im
                _, true = torch.max(labels, 1)
                outputs = net(inputs)
                pred = torch.max(torch.max(outputs.data,1)[0], 1)[1]        
                correct += (pred == true).type(torch.int).sum().item()
                total += len(true)
        print(f'{pathology} Accuracy: {correct*100.0 / total}%', file=open('Train_RegNet.out', 'a'))
        print(f'Finished "{pathology}" training!', file=open('Train_RegNet.out', 'a'))


        out_dir = '/groups/CS156b/2023/BbbBbbB/BabyModel_Weights'
        torch.save(net.state_dict(), os.path.join(out_dir, f'Baby_{pathology.replace(" ","_")}_Weights.pth'))

    toc = time.time()
    print(f'Total time: {toc-tic} s', file=open('Train_RegNet.out', 'a'))
    print('Finished running Baby_Model.py', file=open('Train_RegNet.out', 'a'))
    
    os._exit(0)