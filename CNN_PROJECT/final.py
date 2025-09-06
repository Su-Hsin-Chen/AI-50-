# %% [markdown]
# ### 下載資料集(需要有git)

# %%
#!git clone https://github.com/inoueMashuu/hiragana-dataset

# %% [markdown]
# ### Import 函示庫

# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# %% [markdown]
# ### 標籤與數字對應

# %%
label_dict = {
    'A': 0, 'BA': 1, 'CHI': 2, 'DA': 3, 'E': 4, 'FU': 5, 'HA': 6, 'HE': 7, 'HI': 8, 'HO': 9, 'I': 10, 'JI': 11,
    'KA': 12, 'KE': 13, 'KI': 14, 'KO': 15, 'KU': 16, 'MA': 17, 'ME': 18, 'MI': 19, 'MO': 20, 'MU': 21, 'N': 22,
    'NA': 23, 'NE': 24, 'NI': 25, 'NO': 26, 'NU': 27, 'O': 28, 'PI': 29, 'RA': 30, 'RE': 31, 'RI': 32, 'RO': 33,
    'RU': 34, 'SA': 35, 'SE': 36, 'SHI': 37, 'SO': 38, 'SU': 39, 'TA': 40, 'TE': 41, 'TO': 42, 'TSU': 43, 'U':
    44, 'WA': 45, 'WO': 46, 'YA': 47, 'YO': 48, 'YU': 49
}

label_dict_inv = {v: k for k, v in label_dict.items()}
label_dict_inv

# %% [markdown]
# ### 建構資料集

# %%
class MyDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = Path(path)
        self.transform = transform
        self.images, self.labels = self.readData()

    def readData(self):
        images = []
        labels = []

        for path in self.path.glob('**/*'):
            if path.suffix in ['.jpg']:
                images.append(Image.open(path).copy())
                img_name = path.stem
                if img_name[4:7].isalpha() == True:
                    label_name = img_name[4:7]
                elif img_name[4:6].isalpha() == True:
                    label_name = img_name[4:6]
                else:
                    label_name = img_name[4]
              
            
                labels.append(label_dict[label_name])

        ## 使用for 迴圈 從self.path 讀取所有的圖片(*.jpg)檔案路徑 #TODO
            # Image.open() 讀取圖片 #TODO
            
            # 將圖片加入 images 列表 #TODO
            
            # 假設檔名格式為 'kanaYA1.jpg' 提取YA 作為標籤 #TODO
            
            # 將標籤轉換為數字並加入 labels 列表 #TODO
            
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]

        if self.transform:
            img = self.transform(img)
        return img, label

# %%
dataset_path = "C:/Users/User/Documents/hiragana-dataset-master/hiragana_images"
batch_size = 16
img_size=(28, 28) 
## dataloader #TODO 資料前處理
transform = transforms.Compose([ 
    transforms.Resize(img_size),
    transforms.ToTensor(),   # [0, 1]
    # transforms.Normalize((0.5), (0.5))  # [-1, 1]
    ])

dataset = MyDataset(path=dataset_path, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
## 分成訓練集和測試集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# %% [markdown]
# #### 測試資料是否成功讀取

# %%
def show_images(images, labels):
    plt.figure(figsize=(12, 12))
    for i, (img, label) in enumerate(zip(images, labels)):
        plt.subplot(8, 8, i + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(label_dict_inv[label])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_one_per_label(dataset, label_min=0, label_max=32):
    seen = set()
    imgs, labels = [], []
    for img, label in dataset:
        if label_min <= label <= label_max and label not in seen:
            imgs.append(img)
            labels.append(label)
            seen.add(label)
        if len(seen) == label_max - label_min + 1:
            break
    print(f"Found {len(imgs)} images with labels {label_min}~{label_max}")
    show_images(imgs, labels)

# 使用
show_one_per_label(dataset, 0, 31)


# %% [markdown]
# ### 建構模型

# %%
class Net(nn.Module):
    def __init__(self,in_channels=1, num_classes=50):
        super(Net, self).__init__()
        ## 定義卷積層、池化層、全連接層等 #TODO

        self.cnn1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=5, stride=1, padding=0) # out_shape=(16,24,24)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) # out_shape=(16,12,12)

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0) # out_shape=(32,8,8)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) # out_shape=(32,4,4)

        self.fc1 = nn.Linear(in_features=32 * 4 * 4, out_features=num_classes)  # in_shape=(32*4*4)  # TODO: adjust in_features

    def forward(self, x):
        ## 定義前向傳播過程 #TODO
        x = self.cnn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.cnn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        return x


# %% [markdown]
# ### 設定參數

# %%
## 參數自己定義 #TODO
epochs = 20
learning_rate = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [markdown]
# ### Build
# (1)model, (2)loss-function, (3)optimizer, (4)dataloader

# %%
## TODO build model
model = Net()
model.to(device)
## TODO loss function 
criterion = nn.CrossEntropyLoss()
## TODO optimizer
optimizer =torch.optim.SGD(model.parameters(), lr=learning_rate)



# %% [markdown]
# #### 列出模型架構

# %%
try:
    summary(model, (1, img_size[0], img_size[1]), device=str(device))
except:
    summary(model.cpu(), (1, img_size[0], img_size[1]))


# %% [markdown]
# ### 訓練

# %%
def accuracy(pred: torch.Tensor, label: torch.Tensor):
    _, pred_label = pred.max(1)
    num_correct = (pred_label == label).sum().item()
    acc = num_correct / label.shape[0]
    return acc
metric = {'loss': [], 'acc': [],'test_loss':[], 'test_acc': []}
for i_epoch in tqdm(range(epochs)):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    model.train(mode=True)
    for i_batch, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)

        pred = model.forward(image)  # inference
        loss = criterion(pred, label)  # calculate loss

        ## TODO inference 

        ## TODO calculate loss
        
        optimizer.zero_grad()  # reset gradient to zero

        loss.backward()  # calculate gradient
        optimizer.step()  # optimize weight (using gradient)

        ## TODO calculate gradient
        
        ## TODO optimize weight (using gradient)

        train_loss += [loss.item()]
        train_acc += [accuracy(pred, label)]
    model.eval()
    with torch.no_grad():
        for i_batch, (image, label) in enumerate(test_loader):    
            image = image.to(device)
            label = label.to(device)

            pred = model.forward(image)  # inference
            loss = criterion(pred, label)  # calculate loss

            test_loss += [loss.item()]
            test_acc += [accuracy(pred, label)]
    metric['loss'] += [sum(train_loss)/ len(train_loader)]
    metric['acc'] += [sum(train_acc)/ len(train_loader)]
    metric['test_loss'] += [sum(test_loss)/ len(test_loader)]
    metric['test_acc'] += [sum(test_acc)/ len(test_loader)]
    #print(f'Epoch[{i_epoch+1}/{epochs}] loss: {metric["loss"][-1]}, acc: {metric["acc"][-1]} testloss: {metric["test_loss"][-1]}, testacc: {metric["test_acc"][-1]}')

# %% [markdown]
# ### 視覺化結果

# %%
plt.plot(range(len(metric["loss"])), metric["loss"])
plt.plot(range(len(metric["test_loss"])), metric["test_loss"])
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train Loss', 'Test Loss'])
plt.show()
plt.plot(range(len(metric["acc"])), metric["acc"])
plt.plot(range(len(metric["test_acc"])), metric["test_acc"])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train Accuracy', 'Test Accuracy'])
plt.show()
print(f'loss: {metric["loss"][-1]}, acc: {metric["acc"][-1]} testloss: {metric["test_loss"][-1]}, testacc: {metric["test_acc"][-1]}')

# %% [markdown]
# ### 儲存模型

# %%
model_scripted = torch.jit.script(model.cpu())
model_scripted.save('model_scripted_32.pt')


