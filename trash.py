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
