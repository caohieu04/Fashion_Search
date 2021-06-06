import seaborn as sns
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

from unet import UNet

from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.transforms.functional import crop
import sys
import skimage.io as io
import torch.nn.functional as F
import time
import cv2
import pickle

RETRAIN = False
if platform.system() == 'Windows':
    ROOT_DIR = r'D:\GitHub\Fashion_Search'
else:
    ROOT_DIR = '/content/Fashion_Search'
os.chdir(ROOT_DIR)
HEIGHT = 128
WIDTH = 128
IMAGE_SIZE = 128
#plot grid xy
PGX = 2
PGY = 8
MEGA_BACTH_SIZE = 3200

class SquarePad:
  def __call__(self, image):
    w, h = image.shape[0], image.shape[1]
    max_wh = np.max([w, h])
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    padding = (hp, vp, hp, vp)
    return F.pad(image, padding, 'constant', 0)
  
class dataset(Dataset):
    def __init__(self, csv_file=None, root_dir=None, img_width=300, img_height=300):
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
    def __len__(self):
        return len(self.csv_file)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        csv1 = str(self.csv_file.iloc[idx, 1])
        if (csv1.split(r'/')[1] == 'Striped_A-line_Dress'):
          csv1 = r'img/Striped_A-Line_Dress/' + csv1.split(r'/')[2]
        img_name = os.path.join(self.root_dir, csv1)
        name = self.csv_file.iloc[idx, 1]
        abstr_gr = self.csv_file.iloc[idx, 1].split(r'/')[1]
        image = io.imread(img_name)
        x1, y1, x2, y2 = map(int, self.csv_file.iloc[idx, 5:9])
        image = image[y1:y2, x1:x2, :]

        # image = image.transpose((2, 0, 1))

        # image = torch.tensor(image).byte()
        image = transforms.functional.to_tensor(image)
        data_transform = transforms.Compose([
            # transforms.Resize((HEIGHT, WIDTH)),
            SquarePad(),
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
        ])
        image = data_transform(image)
        sample = {'image': image, 'abstr': abstr_gr, 'name':name}
        return sample



class ARGS():
    batch_size = 32
    epochs = 10
args = ARGS()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(net):
    DF = dataset(csv_file='data/cloth_train.csv', root_dir=ROOT_DIR)
    dataloader = DataLoader(DF, batch_size = args.batch_size, shuffle=True)
    EPOCHS = args.epochs

    optimizer = optim.Adam(net.parameters(), lr=3e-3)
    criterion = nn.MSELoss()
    

    # print(net)

    for epoch in range(EPOCHS):
        start_time_epoch = time.time()
        loss = 0
        net.train()
        # pbar = tqdm(enumerate(dataloader), total = len(dataloader))
        lim = 0
        print('=' * 160)
        first = True
        start_time = time.time()
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
            if lim > 0 and i_batch >= lim:
                break
            if i_batch > 0 and i_batch * args.batch_size % MEGA_BACTH_SIZE == 0 :
                end_time = time.time()
                print(f"    With numbers of images {i_batch * args.batch_size} current loss is {loss:.6f} and elapsed time is {end_time - start_time}")
                start_time = time.time()

        dir_checkpoint = 'checkpoints/'
        save_cp = True
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
        end_time_epoch = time.time()
        print("EPOCH : {}/{}, LOSS = {:.6f}, TIME = {:.6f}".format(epoch + 1, EPOCHS, loss, end_time_epoch - start_time_epoch))
def visualize(fname):
    DF = dataset(csv_file=f'data/cloth_{fname}.csv', root_dir=ROOT_DIR)
    dataloader = DataLoader(DF, batch_size = args.batch_size, shuffle=True)
    first = True
    for i_batch, sample_batched in enumerate(dataloader):
          x_batch = sample_batched['image'].type(torch.FloatTensor)
          x_batch = x_batch.to(device=device, dtype=torch.float32)
          abs_batch = sample_batched['abstr']
          with torch.no_grad():
            outputs = net(x_batch)

          if first:
              x_db = x_batch
              y_db = outputs
              first = False
              break

    fig, axes = plt.subplots(PGX * 2, PGY, figsize=(16, 16))  
    plot_batch_size = 16
    for i in range(plot_batch_size):
        axes[i // PGY, i % PGY].axis('off')
        axes[i // PGY + PGX, i % PGY].axis('off')
        axes[i // PGY, i % PGY].imshow(x_db[i].cpu().detach().numpy().transpose((1, 2, 0)))
        db_img = y_db[i].cpu().detach().numpy().transpose((1, 2, 0))
        db_img = (db_img * 255).astype(np.uint8)
        axes[i // PGY + PGX, i % PGY].imshow(db_img)
    plt.subplots_adjust(left=0,
                    bottom=0, 
                    right=1, 
                    top=0.5, 
                    wspace=0, 
                    hspace=0.001)
    plt.show()
if __name__ == '__main__':
  net = UNet(3, n_classes=1, bilinear=True)
  if os.path.exists(r'./model') and RETRAIN == False:
    net.to('cpu')
    net.load_state_dict(torch.load(r'./model/CP_epoch4.pth', map_location=torch.device('cpu')))
    visualize('test')
  else:
    net.cuda()
    train(net)
    visualize()

os.chdir(ROOT_DIR)
class Hook():
  def __init__(self, module, backward=False):
      if backward==False:
          self.hook = module.register_forward_hook(self.hook_fn)
      else:
          self.hook = module.register_backward_hook(self.hook_fn)
  def hook_fn(self, module, input, output):
      self.input = input
      self.output = output
  def close(self):
      self.hook.remove()


from tqdm import tqdm
def extract_vector(net, fname):
    DF = dataset(csv_file=f'data/cloth_{fname}.csv', root_dir=ROOT_DIR)
    dataloader = DataLoader(DF, batch_size = args.batch_size, shuffle=True)

    god_dic = {}
    net.eval()
    with torch.no_grad():
      pbar = tqdm(enumerate(dataloader), total=len(dataloader), position=0, leave=True)
      # lim_i_batch = 100
      for i_batch, sample_batched in pbar: 
          x_batch = sample_batched['image'].type(torch.FloatTensor)
          x_batch = x_batch.to(device=device, dtype=torch.float32)
          abstr = sample_batched['abstr']
          name = sample_batched['name']
          hook = Hook(net.down4)
          outputs = net(x_batch)
          latent_vec = hook.output
          latent_vec = torch.flatten(latent_vec, start_dim=1)
          hook.close()

          for i in range(len(abstr)):
            col0 = name[i].split(r'/')[1]
            col1 = name[i].split(r'/')[2]
            col2 = latent_vec[i].type(torch.HalfTensor).tolist()
            god_dic[(col0, col1)] = col2
          if i_batch * args.batch_size % MEGA_BACTH_SIZE == 0:
            latent_fpath = 'latent'
            with open(latent_fpath, "ab+") as f:
              pickle.dump(god_dic, f)
            god_dic = {}
          del outputs
          del latent_vec
          del x_batch
          torch.cuda.empty_cache()
    # F_csv['group'] = col0
    # F_csv['name'] = col1
    # F_csv['feature'] = col2 
    # F_csv.to_csv(f'cloth_{fname}_features.csv', index=False)
    # return god_dic
extract_vector(net, 'train')