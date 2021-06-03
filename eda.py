#%%
import argparse
import logging
import os
import sys
import platform
if platform.system() == 'Windows':
    ROOT_DIR = r'D:\GitHub\Fashion_Search'
else:
    ROOT_DIR = '/content/Fashion_Search'
os.chdir(ROOT_DIR)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.transforms.functional import crop
import sys
import skimage.io as io



HEIGHT = 128
WIDTH = 128
#plot grid xy
PGX = 2
PGY = 8


class dataset(Dataset):
    def __init__(self, csv_file=None, root_dir=None, img_width=300, img_height=300):
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
    def __len__(self):
        return len(self.csv_file)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.csv_file.iloc[idx, 0])
        name = self.csv_file.iloc[idx, 0]
        abstr_gr = self.csv_file.iloc[idx, 0].split(r'/')[1]
        image = io.imread(img_name)
        x1, y1, x2, y2 = map(int, self.csv_file.iloc[idx, 4:8])
        image = image[y1:y2, x1:x2]
        # image = image.transpose((2, 0, 1))

        # image = torch.tensor(image).byte()
        image = transforms.functional.to_tensor(image)
        data_transform = transforms.Compose([
            transforms.Resize((HEIGHT, WIDTH)),
            transforms.RandomHorizontalFlip(),
            
        ])
        image = data_transform(image)
        sample = {'image': image, 'abstr': abstr_gr, 'name':name}
        return sample



class ARGS():
    batch_size = 16
    epochs = 10
args = ARGS()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(net):
    DF = dataset(csv_file='data/cloth_train.csv', root_dir=ROOT_DIR)
    dataloader = DataLoader(DF, batch_size = args.batch_size, shuffle=True)
    EPOCHS = args.epochs

    optimizer = optim.Adam(net.parameters(), lr=3e-3)
    criterion = nn.MSELoss()
    
    fig, axes = plt.subplots(PGX * 2, PGY, figsize=(16, 16))
    # print(net)
    for epoch in range(EPOCHS):
        loss = 0
        net.train()
        # pbar = tqdm(enumerate(dataloader), total = len(dataloader))
        lim = 0
        print('=' * 160)
        first = True
        for i_batch, sample_batched in enumerate(dataloader):
            x_batch = sample_batched['image'].type(torch.FloatTensor)
            x_batch = x_batch.to(device=device, dtype=torch.float32)
            abs_batch = sample_batched['abstr']
            optimizer.zero_grad()
            outputs = net(x_batch)

            if first:
                x_db = x_batch
                y_db = outputs
                first = False
            
            train_loss = criterion(outputs, x_batch)
            train_loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.05)
            optimizer.step()
            loss += train_loss.item()
            if lim > 0 and i_batch > lim:
                break
            if i_batch > 0 and i_batch * args.batch_size % 4000 == 0 :
                print(f"    With number of images {i_batch * args.batch_size} current loss is {loss}")

        dir_checkpoint = 'checkpoints/'
        save_cp = True
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
        print("Epoch : {}/{}, loss = {:.6f}".format(epoch + 1, EPOCHS, loss))
            
    for i in range(args.batch_size):
        axes[i // PGY, i % PGY].axis('off')
        axes[i // PGY + PGX, i % PGY].axis('off')
        axes[i // PGY, i % PGY].imshow(x_db[i].cpu().detach().numpy().transpose((1, 2, 0)))
        db_img = y_db[i].cpu().detach().numpy().transpose((1, 2, 0))
        db_img = (db_img * 255).astype(np.uint8)
        axes[i // PGY + PGX, i % PGY].imshow(db_img)
    loss = loss / len(dataloader)
    plt.subplots_adjust(left=0,
                    bottom=0, 
                    right=1, 
                    top=0.5, 
                    wspace=0, 
                    hspace=0.001)
    plt.show()

if __name__ == '__main__':
  net = UNet(3, n_classes=1, bilinear=True)
  if os.path.exists(r'./models'):
    net.load_state_dict(torch.load(r'./models/CP_epoch10.pth'))
    net.cuda()
  else:
    net.cuda()
    train(net)

# %%
handle = net.down4.register_forward_hook(func)
down4hook = 0
def func(self, input, output):
    down4hook = torch.flatten(output[0])

MasterDict = {}

def extract_vector(net):
    args = ARGS()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DF = dataset(csv_file='data/cloth_train.csv', root_dir=ROOT_DIR)
    dataloader = DataLoader(DF, batch_size = args.batch_size, shuffle=True)
    EPOCHS = args.epochs

    net.eval()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i_batch, sample_batched in pbar:
        x_batch = sample_batched['image'].type(torch.FloatTensor)
        x_batch = x_batch.to(device=device, dtype=torch.float32)

        abstr = sample_batched['abstr']
        name = sample_batched['name']
        outputs = net(x_batch)
        
        for i in range(len(abstr)):
          if not abstr[i] in MasterDict:
              MasterDict[abstr[i]] = []
        MasterDict[abstr[i]] = (name[i].split(r'/')[2], down4hook)
extract_vector(net)
print(len(MasterDict.keys()))
handle.remove()