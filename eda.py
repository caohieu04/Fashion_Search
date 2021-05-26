import pandas as pd
# from pandas_profiling import ProfileReport
import pandas_profiling
import os 
import torch
from unet import UNet
from torch.utils.data import Dataset, DataLoader
import skimage.io as io

class dataset(Dataset):
    def __init__(self, csv_file=None, root_dir='/content/Fashion_Search', img_width=300, img_height=300):
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
    def __len__(self):
        return len(self.csv_file)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.csv_file.iloc[idx, 0])
        image = io.imread(img_name)


os.chdir('d:/GitHub/Fashion_Search/')
df = pd.read_csv('data/cloth.csv')
df = df[df['category_name'] != 'Hoodi']

print(len(df['category_name'].unique()))
n_classes = len(df['category_name'].unique())
print(df['category_name'].value_counts())

df.describe()
net = UNet(n_channels=3, n_classes=n_classes, bilinear=True)




