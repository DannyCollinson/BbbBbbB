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
    def __init__(self, labels, id_to_path, pathology='No Finding'):
        # labels is a (n_samples x 3) one-hot encoded numpy array...
        # with columns representing classifications -1, 0, 1
        # id_to_path is a dictionary mapping ids to image path names
        # imdir_path is the path to the directory of all downsampled images
        # pathology is the pathology we are trying to classify
        # remote specifies if this is running locally or on the hpc
        self.labels = labels
        self.id_to_path = id_to_path
        self.imdir_path = "/groups/CS156b/2023/BbbBbbB/Train_512x512"
        self.pathology = pathology
    
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        indx = list(self.id_to_path.keys())[idx]
        impath = os.path.join(self.imdir_path, self.id_to_path[indx])
        im = np.load(impath)
        return torch.from_numpy(im), torch.tensor(self.labels.iloc[idx]).type(torch.float)



class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv2d(1, 8, kernel_size=3, stride=1, groups=1)
        self.conv12 = nn.Conv2d(8, 8, kernel_size=3, stride=1, groups=2)
        self.conv13 = nn.Conv2d(8, 4, kernel_size=3, stride=1, groups=2)
        self.bnorm1 = nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv21 = nn.Conv2d(4, 8, kernel_size=3, stride=1, groups=2)
        self.conv22 = nn.Conv2d(8, 8, kernel_size=3, stride=1, groups=4)
        self.conv23 = nn.Conv2d(8, 4, kernel_size=3, stride=1, groups=2)
        self.bnorm2 = nn.BatchNorm2d(4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(60516, 10000)
        self.lin2 = nn.Linear(10000, 1000)
        self.lin3 = nn.Linear(1000, 100)
        self.lin4 = nn.Linear(100, 3)
    
    def forward(self, x):
        x = x.view(x.shape[0], 1, 512, 512).type(torch.float)
        x = F.relu(self.conv12(self.conv11(x)))
        x = F.relu(self.conv13(x))
        x = self.pool1(self.bnorm1(x))
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv23(x))
        x = self.pool2(self.bnorm2(x))
        x = F.relu(self.lin1(self.flat(x)))
        x = F.relu(F.dropout(self.lin2(x)))
        x = F.relu(F.dropout(self.lin3(x)))
        x = self.lin4(x)
        return F.softmax(x, dim=1)

###############################################################################

if __name__ == '__main__':
    
    outfile = 'output/Train_BasicCNN_NoFinding.out'
    errfile = 'output/Train_BasicCNN_NoFinding.err'
    
    weights_out_dir = '/groups/CS156b/2023/BbbBbbB/BasicCNN_Weights_Full'

    
    print('Running Basic_CNN.py', file=open(outfile, 'a'))

    
    tic = time.time()
    
    testing = False
    test_max = 100

    pathologies = ['No Finding']#, 'Enlarged Cardiomediastinum','Cardiomegaly',
                   #'Lung Opacity','Pneumonia','Pleural Effusion',
                   #'Pleural Other','Fracture','Support Devices']

    labels = pd.read_csv('/central/groups/CS156b/2023/BbbBbbB/Labels_Onehot.csv', index_col='id')

    train_dir = '/groups/CS156b/2023/BbbBbbB/Train_512x512'
    
    im_paths = []
    for root, subdirs, files in os.walk(train_dir):
        for fname in files:
            im_paths.append(fname)
    
    id_to_path = {}
    for idx in labels.index:
        if testing:
            if idx > test_max:
                continue
        id_to_path[idx] = labels['Short_Path'][idx]

    path_to_id = {}
    for idx in id_to_path:
        if testing:
            if idx not in id_to_path.keys():
                continue
        pth = id_to_path[idx]
        if pth in im_paths:
            path_to_id[pth] = idx

    idxs = []
    for im_path in im_paths:
        if testing:
            if im_path not in path_to_id.keys():
                continue
        idxs.append(path_to_id[im_path])
    
    labels = labels.loc[idxs]
    
        
    batch_size = 32
    n_epochs = 10

    for pathology in pathologies:

        data = OnePathologyDataset(labels[[pathology+' -1', pathology+' 0', pathology+' 1']],
                                   id_to_path, pathology=pathology)

        train_test = random_split(data, [int(labels.shape[0]*0.7), labels.shape[0]-int(labels.shape[0]*0.7)])

        train_loader = DataLoader(train_test[0], batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(train_test[1], batch_size=batch_size, shuffle=False)

        
        CNN = BasicCNN()

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(CNN.parameters(), lr=0.01)


        print(f'Starting "{pathology}" training!', file=open(outfile, 'a'))
        for epoch in range(n_epochs):
            for im in train_loader:
                inputs, labs = im
                optimizer.zero_grad()

                outputs = CNN(inputs)
                loss = loss_fn(outputs, labs)
                loss.backward()
                optimizer.step()
            print(f'\tEpoch {epoch+1}', file=open(outfile, 'a'))
            torch.save(CNN.state_dict(),
                       os.path.join(weights_out_dir, 
                                    f'BasicCNN_Weights_{pathology.replace(" ","_")}_Full_Epoch{epoch}.pth'))

        print(f'Finished "{pathology}" training!', file=open(outfile, 'a'))

        correct = 0
        total = 0
        with torch.no_grad():
            for im in test_loader:
                inputs, labs = im
                _, true = torch.max(labs, dim=1)
                outputs = CNN(inputs)
                _, pred = torch.max(outputs, dim=1)
                correct += (pred == true).type(torch.int).sum().item()
                total += len(true)
            print(f'{pathology} Accuracy: {correct*100.0 / total}%\n',
                  file=open(outfile, 'a'))

        torch.save(CNN.state_dict(), 
                   os.path.join(weights_out_dir,
                                f'BasicCNN_Weights_{pathology.replace(" ","_")}_Full_Final.pth'))

    toc = time.time()
    print(f'Total time: {toc-tic} s', file=open(outfile, 'a'))
    print('Finished running Basic_CNN.py', file=open(outfile, 'a'))
    
    os._exit(0)