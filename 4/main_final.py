import numpy as np
import os
import csv 
from sklearn.model_selection import train_test_split
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch
import torch.utils.data as datatorch
import torch.nn as nn
from sklearn.metrics import f1_score
import torch.backends.cudnn as cudnn
import time
import datetime


#load triplets 
train_triplets = np.loadtxt('train_triplets.txt', dtype= 'str')
test_triplets = np.loadtxt('test_triplets.txt', dtype= 'str')


#splitting the train data
train_triplets , val_triplets = train_test_split(train_triplets, test_size = 0.2)


#half index of the val_triplets 
half_index = np.int64((val_triplets.shape[0]-val_triplets.shape[0]%2)/2)

#prepare val_labels
val_labels = np.int64(np.ones((val_triplets.shape[0],)))

val_triplets[half_index:, 0], val_triplets[half_index:, 1] = val_triplets[half_index:, 1], val_triplets[half_index:, 0].copy()
val_labels[half_index:] = np.int64(0)

train_dir = 'food'
train_files = os.listdir(train_dir)
test_files = os.listdir(train_dir)
number_files = len(test_files)


class ImageTriplesSet(Dataset):
    def __init__(self , file_array, dir, mode='train', transform = None,labels =None):
        self.triple_list = list(map(tuple, file_array))
        self.mode = mode
        self.labels = labels
        self.dir = dir
        self.transform = transform
        
    def __len__(self):
        return len(self.triple_list)
    
    def __getitem__(self,idx):
        img1 = Image.open(os.path.join(self.dir, self.triple_list[idx][0] + '.jpg'))
        img2 = Image.open(os.path.join(self.dir, self.triple_list[idx][1] + '.jpg'))
        img3 = Image.open(os.path.join(self.dir, self.triple_list[idx][2] + '.jpg'))
        
        
        if self.transform is not None:
            img1 = self.transform(img1).numpy()
            img2 = self.transform(img2).numpy()
            img3 = self.transform(img3).numpy()
        if self.labels is None:
            return img1, img2, img3
        else:
            return img1, img2, img3, self.labels[idx]
        

data_transform = transforms.Compose([
        transforms.Resize(228),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

train_dataset = ImageTriplesSet(train_triplets, train_dir, transform = data_transform, labels = None)
val_dataset = ImageTriplesSet(val_triplets, train_dir, transform= data_transform, labels = None)
test_dataset = ImageTriplesSet(test_triplets, train_dir, mode="test" ,transform = data_transform,labels = None)

model = torch.hub.load('pytorch/vision', 'vgg16', pretrained=True)

learning_rate = 0.0001
batch_size = 32
epochs = 4
logstep = int(1000 // batch_size)

train_loader = datatorch.DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=1e-5,nesterov=True)

training_loss_vec = []
training_accuracy_vec = []
val_f1_score = []

start = time.time()
# loop over epochs
model.train()
for e in range(epochs):
    training_loss = 0.
    training_accuracy = 0.
    for idx, (data1, data2, data3) in enumerate(train_loader):
        data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        embedded_a, embedded_p, embedded_n = model(data1), model(data2), model(data3)
        loss = criterion(embedded_a, embedded_p, embedded_n)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        if (idx) % logstep == 0: 
            training_loss_vec.append(training_loss/logstep)
            print('[%d, %5d] training loss: %.5f' %
                  (e + 1, idx + 1, training_loss/logstep))
            training_loss, training_accuracy = 0.,0.

end = time.time()
print(str(datetime.timedelta(seconds= end - start)))

val_loader = datatorch.DataLoader(dataset=val_dataset, shuffle = False, batch_size= 1)
start = time.time()
val_labels_pred = []
model.eval()
for idx, (data1, data2, data3) in enumerate(val_loader):
    data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
    embedded_1, embedded_2, embedded_3 = model(data1), model(data2), model(data3)
    if torch.dist(embedded_1,embedded_3,1)>=torch.dist(embedded_1,embedded_2,1):
        val_labels_pred.append(1)
    else:
        val_labels_pred.append(0)

f1 = f1_score(val_labels_pred, val_labels)
print(f1)

test_loader = datatorch.DataLoader(dataset=test_dataset, shuffle = False, batch_size= 1)
end = time.time()
print(str(datetime.timedelta(seconds= end - start)))

test_triplets_pred = []
model.eval()
start = time.time()
for idx, (data1, data2, data3) in enumerate(test_loader):
    data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
    embedded_1, embedded_2, embedded_3 = model(data1), model(data2), model(data3)
    if torch.dist(embedded_1,embedded_3,1)>=torch.dist(embedded_1,embedded_2,1):
        test_triplets_pred.append(str(1))
    else:
        test_triplets_pred.append(str(0))
end = time.time()
print(str(datetime.timedelta(seconds= end - start)))

file_name = 'submission_Stefano.txt'
with open(file_name, 'w') as f:
    for item in test_triplets_pred:
        f.write(item + '\n')
