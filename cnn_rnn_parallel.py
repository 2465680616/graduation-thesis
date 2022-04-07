import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
LR = 0.005
Epoch = 10
SampleLen = 400
if torch.cuda.is_available() :print('^_^')
device = torch.device("cuda:0")

class TrainDataset(Dataset):
    def __init__(self):
        data = pd.read_csv("./data/data_train.csv")
        self.train_x = data.loc[:,['data']]
        #print(self.train_x)
        self.train_y = data.loc[:,'label']

    def __len__(self):
        return (len(self.train_y))//SampleLen

    def __getitem__(self, item):
        train_x = self.train_x.iloc[SampleLen * (item):SampleLen * (item + 1)]
        t = self.train_y.iloc[SampleLen * (item):SampleLen * (item + 1)]
        # print(t)
        sum = np.sum(t)
        sum = sum/SampleLen
        train_y = np.array([sum]) #self.train_y.iloc[512 * (item):512 * (item + 1)]
        # print(train_y)
        # train_x=torch.reshape(train_x,(512*5))
        return {
            'train_x':torch.from_numpy(train_x.values).float().reshape(1,SampleLen),
            'train_y':torch.from_numpy(train_y).float(),
        }

class TestDataset(Dataset):
    def __init__(self):
        data = pd.read_csv("./data/data_test.csv")
        self.test_x = data.loc[:,['data']]

        self.test_y = data.loc[:,'label']

    def __len__(self):
        return (len(self.test_y))//SampleLen

    def __getitem__(self, item):
        test_x = self.test_x.iloc[SampleLen*(item):SampleLen*(item+1)]
        t = self.test_y.iloc[SampleLen*(item):SampleLen*(item+1)]
        sum = np.sum(t)
        sum = sum/SampleLen
        test_y = np.array([sum])
        # print(train_x.shape)
        # train_x=torch.reshape(train_x,(512*5))
        return {
            'test_x':torch.from_numpy(test_x.values).float().reshape(1,SampleLen),
            'test_y':torch.from_numpy(test_y).float(),
        }

train_dataset = TrainDataset()
train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True,num_workers=0)
# dd1 = next(iter(train_dataset))
# print(dd1['train_x'].shape)

# for data in train_loader:
#     train_x,train_y=data['train_x'],data['train_y']
#     print(train_x.shape)
#     print(train_y)
#     break

total_episodes = Epoch * len(train_loader)

test_dataset = TestDataset()
test_loader = DataLoader(test_dataset,batch_size=4,shuffle=True,num_workers=0)

train_data_size = len(train_dataset)
test_data_size = len(test_dataset)

class CNN_RNN(nn.Module):
    def __init__(self):
        super(CNN_RNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=9,
                stride=1,
                padding=4,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, 5, 1, 2),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2)
                                   )

        self.conv3 = nn.Sequential(nn.Conv1d(64, 10, 9, 1, 4),
                                   nn.BatchNorm1d(10),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2),
                                    nn.Flatten()
                                   )
        self.conv4 = nn.Sequential(nn.Conv1d(16,3,9,1,4),
                                   nn.BatchNorm1d(3),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2),
                                   nn.Flatten())
        self.rnn = nn.GRU(
                    input_size=1,
                    hidden_size=64,
                    num_layers=1,
                    batch_first=True
        )
        self.out = nn.Linear(564,10)
        # self.linear1 = nn.Linear(128*1*40,1024)
        # self.linear2 = nn.Linear(1024, 2)

    def forward(self,input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        x = self.conv3(conv2)

        # print(x.shape)
        y1 = input.reshape(4,400,1)
        # x_out,(h_n,h_c) = self.rnn(y1,None)
        x_out, h_0 = self.rnn(y1, None)
        y = x_out[:,-1,:]
        # print(y.shape)
        # print(out)
        x =torch.concat([x,y],dim=1)
        # print(x.shape)
        out = self.out(x)
        # print(result)
        # print(out.shape)
        # x = x[0].reshape(32,-1)
        # print(x.shape)
        # print(len(x))
        # x = self.linear1(x)
        # x = self.linear2(x)
        return out

cnn_rnn = CNN_RNN()
cnn_rnn = cnn_rnn.float().to(device)
# input = torch.ones(32,1,2560).to(device)
# output = cnn_rnn(input)
# print(output.shape)


epoch = 1000
optimizer = torch.optim.SGD(cnn_rnn.parameters(), lr=LR, weight_decay=1e-5)
lrf = lambda x : (1-0.9*x/epoch)
lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer,lrf)
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)

total_train_step = 0
total_test_step = 0


writer = SummaryWriter("./logs_train4")

for i in range(epoch):
    print("------第 {} 轮训练开始--------".format(i + 1))
    #训练步骤开始
    cnn_rnn.train()
    for data in tqdm(train_loader):
        datas,labels = data['train_x'],data['train_y']
        datas = datas.float().to(device)
        labels = [int(x[0]) for x in labels]
        labels = F.one_hot(torch.tensor(labels),num_classes=10)
        labels = labels.float().to(device)
        # print(labels)
        # if hasattr(torch.cuda,'empty_cache'):
        #     torch.cuda.empty_cache()
        outputs = cnn_rnn(datas)
        # print(outputs)
        # if hasattr(torch.cuda,'empty_cache'):
        #     torch.cuda.empty_cache()
        # print(outputs.shape)
        # print(labels.shape)
        loss = loss_function(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 500 == 0 :
            print("训练次数： {}, Loss: {}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    lr_schedule.step()
    print(optimizer.param_groups[0]['lr'])
        #测试步骤开始
    cnn_rnn.eval()
    total_test_loss = 0
    total_accurary = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            datas,labels = data['test_x'],data['test_y']
            datas = datas.float().to(device)
            labels = [int(x[0]) for x in labels]
            labels = F.one_hot(torch.tensor(labels), num_classes=10)
            labels = labels.float().to(device)
            outputs = cnn_rnn(datas)
            loss = loss_function(outputs,labels)
            total_test_loss = total_test_loss + loss.item()
            # print(outputs)
            # print(labels)
            m = torch.argmax(outputs,axis=1)
            n = torch.argmax(labels,axis =1)
            # print(m)
            # print(labels)
            accurary = (m == n).sum()
            total_accurary += accurary
    # cnn_rnn.train()

    print("整体测试集上的Loss： {}".format(total_test_loss))
    print("整体测试集上的正确率： {}".format(total_accurary / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accurary", total_accurary / test_data_size, total_test_step)
    total_test_step += 1

    # if i % 50 == 0 :
    #     # torch.save(cnn_rnn,"cnn_lstm_{}.pth".format(i))
    #     state = {'model':cnn_rnn.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':i}
    #     torch.save(state,"cnn_lstm_state_{}.pth".format(i))
    #     print("模型已保存")

writer.close()