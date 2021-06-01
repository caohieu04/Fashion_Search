#%%
import argparse
import logging
import os
import sys

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

import platform
if platform.system() == 'Windows':
    ROOT_DIR = r'D:\GitHub\Fashion_Search'
else:
    ROOT_DIR = '/content/Fashion_Search'
os.chdir(ROOT_DIR)

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
        sample = {'image': image}
        return sample



class ARGS():
    batch_size = 64
    epochs = 10


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = ARGS()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DF = dataset(csv_file='data/cloth.csv', root_dir=ROOT_DIR)
    dataloader = DataLoader(DF, batch_size = args.batch_size, shuffle=True)
    EPOCHS = args.epochs

    model = UNet(3, n_classes=1, bilinear=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    fig, axes = plt.subplots(1, 10)
    # print(model)
    for epoch in range(EPOCHS):
        loss = 0

        for i_batch, sample_batched in enumerate(dataloader):
            x_batch = sample_batched['image'].type(torch.FloatTensor)

            optimizer.zero_grad()
            outputs = model(x_batch)

            print(outputs.shape)
            db_img = outputs[0].detach().numpy().transpose((1, 2, 0))
            db_img = (db_img * 255).astype(np.uint8)
            print(db_img.shape)
            axes[epoch].imshow(db_img)
            break
            train_loss = criterion(outputs, x_batch)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()
            
        loss = loss / len(dataloader)
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, EPOCHS, loss))
    plt.show()



# %%
