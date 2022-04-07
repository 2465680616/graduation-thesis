import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR

train_len = 72000
test_len = 36000
valid_len = 7200

df1 = scio.loadmat('./data/0.007-Ball.mat')['X119_DE_time']
data_result0 = pd.DataFrame(df1,columns=['data'])

df1 = scio.loadmat('./data/0.007-InnerRace.mat')['X106_DE_time']
data_result1 = pd.DataFrame(df1,columns=['data'])

df1 = scio.loadmat('./data/0.007-OuterRace6.mat')['X131_DE_time']
data_result2 = pd.DataFrame(df1,columns=['data'])

df1 = scio.loadmat('./data/0.014-Ball.mat')['X186_DE_time']
data_result3 = pd.DataFrame(df1,columns=['data'])

df1 = scio.loadmat('./data/0.014-InnerRace.mat')['X170_DE_time']
data_result4 = pd.DataFrame(df1,columns=['data'])

df1 = scio.loadmat('./data/0.014-OuterRace6.mat')['X198_DE_time']
data_result5 = pd.DataFrame(df1,columns=['data'])

df1 = scio.loadmat('./data/0.021-Ball.mat')['X223_DE_time']
data_result6 = pd.DataFrame(df1,columns=['data'])

df1 = scio.loadmat('./data/0.021-InnerRace.mat')['X210_DE_time']
data_result7 = pd.DataFrame(df1,columns=['data'])

df1 = scio.loadmat('./data/0.021-OuterRace6.mat')['X235_DE_time']
data_result8 = pd.DataFrame(df1,columns=['data'])

df1 = scio.loadmat('./data/Normal.mat')['X098_DE_time']
data_result9 = pd.DataFrame(df1,columns=['data'])



# csv_data = data_result.to_csv("./data/0.csv")

def merge_data_train(df,label):
    df_result = pd.DataFrame(df.iloc[0:train_len])
    df_result['label'] = label
    return df_result

def merge_data_test(df,label):
    df_result = pd.DataFrame(df.iloc[train_len:train_len+test_len,:])
    df_result['label'] = label
    return df_result

def merge_data_valid(df,label):
    df_result = pd.DataFrame(df.iloc[train_len+test_len:train_len+test_len+valid_len,:])
    df_result['label'] = label
    return df_result

data_train_result = pd.DataFrame(columns=['data','label'])
data_test_result = pd.DataFrame(columns=['data','label'])
data_valid_result = pd.DataFrame(columns=['data','label'])

d1 = merge_data_train(data_result0,0)
data_train_result = data_train_result.append(d1,ignore_index=True)
print(len(data_train_result))
d1 = merge_data_train(data_result1,1)
data_train_result = data_train_result.append(d1,ignore_index=True)
print(len(data_train_result))
d1 = merge_data_train(data_result2,2)
data_train_result = data_train_result.append(d1,ignore_index=True)
print(len(data_train_result))
d1 = merge_data_train(data_result3,3)
data_train_result = data_train_result.append(d1,ignore_index=True)
print(len(data_train_result))
d1 = merge_data_train(data_result4,4)
data_train_result = data_train_result.append(d1,ignore_index=True)
print(len(data_train_result))
d1 = merge_data_train(data_result5,5)
data_train_result = data_train_result.append(d1,ignore_index=True)
print(len(data_train_result))
d1 = merge_data_train(data_result6,6)
data_train_result = data_train_result.append(d1,ignore_index=True)
print(len(data_train_result))
d1 = merge_data_train(data_result7,7)
data_train_result = data_train_result.append(d1,ignore_index=True)
print(len(data_train_result))
d1 = merge_data_train(data_result8,8)
data_train_result = data_train_result.append(d1,ignore_index=True)
print(len(data_train_result))
d1 = merge_data_train(data_result9,9)
data_train_result = data_train_result.append(d1,ignore_index=True)
print(len(data_train_result))
print(data_train_result)

print(data_test_result)
d2 = merge_data_test(data_result0,0)
data_test_result = data_test_result.append(d2,ignore_index=True)
print(len(data_test_result))
d2 = merge_data_test(data_result1,1)
data_test_result = data_test_result.append(d2,ignore_index=True)
d2 = merge_data_test(data_result2,2)
data_test_result = data_test_result.append(d2,ignore_index=True)
d2 = merge_data_test(data_result3,3)
data_test_result = data_test_result.append(d2,ignore_index=True)
d2 = merge_data_test(data_result4,4)
data_test_result = data_test_result.append(d2,ignore_index=True)
d2 = merge_data_test(data_result5,5)
data_test_result = data_test_result.append(d2,ignore_index=True)
d2 = merge_data_test(data_result6,6)
data_test_result = data_test_result.append(d2,ignore_index=True)
d2 = merge_data_test(data_result7,7)
data_test_result = data_test_result.append(d2,ignore_index=True)
d2 = merge_data_test(data_result8,8)
data_test_result = data_test_result.append(d2,ignore_index=True)
d2 = merge_data_test(data_result9,9)
data_test_result = data_test_result.append(d2,ignore_index=True)
print(len(data_test_result))
print(data_test_result)

d3 = merge_data_valid(data_result0,0)
data_valid_result = data_valid_result.append(d3,ignore_index=True)
d3 = merge_data_valid(data_result1,1)
data_valid_result = data_valid_result.append(d3,ignore_index=True)
d3 = merge_data_valid(data_result2,2)
data_valid_result = data_valid_result.append(d3,ignore_index=True)
d3 = merge_data_valid(data_result3,3)
data_valid_result = data_valid_result.append(d3,ignore_index=True)
d3 = merge_data_valid(data_result4,4)
data_valid_result = data_valid_result.append(d3,ignore_index=True)
d3 = merge_data_valid(data_result5,5)
data_valid_result = data_valid_result.append(d3,ignore_index=True)
d3 = merge_data_valid(data_result6,6)
data_valid_result = data_valid_result.append(d3,ignore_index=True)
d3 = merge_data_valid(data_result7,7)
data_valid_result = data_valid_result.append(d3,ignore_index=True)
d3 = merge_data_valid(data_result8,8)
data_valid_result = data_valid_result.append(d3,ignore_index=True)
d3 = merge_data_valid(data_result9,9)
data_valid_result = data_valid_result.append(d3,ignore_index=True)
print(data_valid_result)

csv_data_train = data_train_result.to_csv("./data/data_train.csv")
csv_data_test = data_test_result.to_csv("./data/data_test.csv")
csv_data_valid = data_valid_result.to_csv("./data/data_valid.csv")