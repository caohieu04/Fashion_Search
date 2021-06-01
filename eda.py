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
        sample = {'image': image, 'abstr': abstr_gr}
        return sample



class ARGS():
    batch_size = 16
    epochs = 40


def train(net):
    args = ARGS()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DF = dataset(csv_file='data/cloth_train.csv', root_dir=ROOT_DIR)
    dataloader = DataLoader(DF, batch_size = args.batch_size, shuffle=True)
    EPOCHS = args.epochs

    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    fig, axes = plt.subplots(8, 4, figsize=(32, 32))
    # print(net)
    max_i_batch = 10
    for epoch in range(EPOCHS):
        loss = 0
        net.train()
        # pbar = tqdm(enumerate(dataloader), total = len(dataloader))
        for i_batch, sample_batched in enumerate(dataloader):
            x_batch = sample_batched['image'].type(torch.FloatTensor)
            x_batch = x_batch.to(device=device, dtype=torch.float32)
            abs_batch = sample_batched['abstr']
            optimizer.zero_grad()
            outputs = net(x_batch)
            
            train_loss = criterion(outputs, x_batch)
            train_loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()
            loss += train_loss.item()
            
            if i_batch > max_i_batch:
              break
            # if i_batch > 0 and i_batch % 50 == 0 :
            #     print(f"With batch {i_batch} current loss is {loss / (i_batch * args.batch_size)}")
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, EPOCHS, loss))
            
    for i in range(16):
        axes[i // 4, i % 4].axis('off')
        axes[i // 4 + 4, i % 4].axis('off')
        axes[i // 4, i % 4].imshow(x_batch[i].cpu().detach().numpy().transpose((1, 2, 0)))
        db_img = outputs[i].cpu().detach().numpy().transpose((1, 2, 0))
        db_img = (db_img * 255).astype(np.uint8)
        axes[i // 4 + 4, i % 4].imshow(db_img)
    loss = loss / len(dataloader)
    plt.show()

if __name__ == '__main__':
    net = UNet(3, n_classes=1, bilinear=True)
    net.cuda()
    train(net)
# %%
