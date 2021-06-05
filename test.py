#%%
import os
import platform
import torch.nn as nn
import torch
if platform.system() == 'Windows':
  ROOT_DIR = r'D:\GitHub\Fashion_Search'
else:
  ROOT_DIR = '/content/Fashion_Search'
os.chdir(ROOT_DIR)

import pandas as pd
train_label_df  = pd.read_csv('data/cloth_train.csv')
MasterDict = {}
for index, row in train_label_df.iterrows():
  key = tuple(row['image_name'].split(r'/')[1:])
  MasterDict[key] = row['category_name']
  
train_df = pd.read_csv('data/test/cloth_train_features.csv')
train_gr_cnt = train_df.group.value_counts()
train_catena_cnt = train_label_df.category_name.value_counts()

test_df = pd.read_csv('data/test/cloth_test_features.csv')
#%%
class Info():
  def __init__(self, df, idx):
    self.group = df.iloc[idx].group
    self.name = df.iloc[idx].name_img
    self.feature = torch.FloatTensor(list(map(float, df.iloc[idx].feature[1:-1].split(','))))
    self.label = MasterDict[(self.group, self.name)]
source_info = Info(train_df, 24)
cossim = nn.CosineSimilarity(dim=0)

import matplotlib.pyplot as plt
import heapq
G = 3
lim_G = G * G

def cal_acc(source_info, df):
  lis = []
  # size = train_gr_cnt[source_info.group]
  size = train_catena_cnt[source_info.label]
  from tqdm import tqdm
  for index in tqdm(range(len(df)), position=0, leave=True):
    target_info = Info(df, index)
    heapq.heappush(lis, (cossim(source_info.feature, target_info.feature), 
                         target_info.group, 
                         target_info.name, 
                         target_info.label))
                         
    while len(lis) > size:
      heapq.heappop(lis)
  lis = sorted(lis, key=lambda tup: tup[0], reverse=True)
  print(source_info.group, source_info.name, source_info.feature)
  for L in lis[:lim_G]:
    print(L)
  list_of_path_tosubplot(source_info, lis)
  
  return 1.0 * list(map(lambda x:x[3], lis)).count(source_info.label) / size
  # return 1.0 * list(map(lambda x:x[1], lis)).count(source_info.group) / size

import skimage.io as io
def list_of_path_tosubplot(source_info, lis):

  fig, axes = plt.subplots(G, G, figsize=(32, 32))
  img = io.imread(os.path.join('img', source_info.group, source_info.name))
  axes[0, 0].axis('off')
  axes[0, 0].imshow(img)
  for i in range(0, len(lis[:G * G]) - 1):
    j = i  + 1
    axes[j // G, j % G].axis('off')
    print(lis[i][1], "##", lis[j][2], "##", lis[j][3][:lim_G])
    img = io.imread(os.path.join('img', lis[j][1], lis[j][2]))
    axes[j // G, j % G].imshow(img)
    
acc = cal_acc(source_info, train_df)
print(acc)
  
    
    


# %%
