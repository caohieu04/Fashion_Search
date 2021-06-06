    # print(DF.csv_file.describe())
    # print(DF[5]['image'])
    # te = DF[5]['image'].numpy().transpose((1, 2, 0))
    # print(te.shape)
    # plt.imshow(te)
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched['image'].size())

    # df = pd.read_csv('data/cloth.csv')
    # df = df[df['category_name'] != 'Hoodi']

    # print(len(df['category_name'].unique()))
    # n_classes = len(df['category_name'].unique())
    # print(df['category_name'].value_counts())

    # df.describe()
    
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        scale = 4
        self.inc = DoubleConv(n_channels, scale)
        self.down1 = Down(scale, scale * 2)
        self.down2 = Down(scale * 2, scale * 4)
        self.down3 = Down(scale * 4, scale * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(scale * 8, scale * 16 // factor)
        self.up1 = Up(scale * 16, scale * 8 // factor, bilinear)
        self.up2 = Up(scale * 8, scale * 4 // factor, bilinear)
        self.up3 = Up(scale * 4, scale * 2 // factor, bilinear)
        self.up4 = Up(scale * 2, scale, bilinear)
        self.rcstr = DoubleConv(scale, 3)
        # self.outc = OutConv(64, n_classes)  
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.rcstr(x)
        return x
        # logits = self.outc(x)
        # return logits
        
# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
#         self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.rcstr = DoubleConv(64, 3)
#         # self.outc = OutConv(64, n_classes)  
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)

#         x = self.rcstr(x)
#         return x
#         # logits = self.outc(x)
#         # return logits

#%%
import os
import platform
import torch.nn as nn
import torch
import pickle
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

lists = []
infile = open('latent', 'rb')
cnt = 3200
while 1:
    try:
        lists.append(pickle.load(infile))
        cnt += 3200
        if (cnt >= 32000):
          break
    except (EOFError, UnpicklingError):
        break
dic = [(k[0], k[1], v) for element in lists for k,v in element.items()]
infile.close()
print(dic)

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
