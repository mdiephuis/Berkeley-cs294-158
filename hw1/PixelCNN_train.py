import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import utils
from PixelCNN_model import PixelCNN

import torch
import torch.nn as nn

from torch.utils.data import DataLoader


import torchvision.transforms as T
eps = np.finfo(float).eps

import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(3)
use_cuda = True
use_cuda = use_cuda and torch.cuda.is_available()
print(use_cuda)



batch_size = 64
train_valid, x_test = utils.load_data('../data/mnist-hw1.pkl')

x_train = train_valid[:int(len(train_valid)*0.8)]
x_valid = train_valid[int(len(train_valid)*0.8):]

transform = T.Compose([T.ToTensor()])

dataloader_train = DataLoader(utils.CMNIST(x_train, transform=transform), batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

dataloader_val = DataLoader(utils.CMNIST(x_valid, transform=transform), batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

dataloader_test = DataLoader(utils.CMNIST(x_test, transform=transform), batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)


def train_validate(model, dataloader, optim, loss_fn, train):
    model.train() if train else model.eval()
    total_loss = 0
    for batch_idx, x in enumerate(dataloader):


        x = x.float().cuda() if use_cuda else x.float()

        x_hat = model(x)
        loss = loss_fn(x_hat, x.long())
        if batch_idx % 50 == 0:
            print('\n batch:{} ---- loss:{}'.format(batch_idx, loss.item()))

        if train:
            optim.zero_grad()
            loss.backward()
            optim.step()

        total_loss += loss.item()
    return total_loss / len(dataloader.dataset)

model = PixelCNN(128)
model = model.cuda() if use_cuda else model
model.apply(utils.init_weights)

optim = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=45, gamma=0.1)

# loss_fn = F.cross_entropy
loss_fn = nn.CrossEntropyLoss(reduction='mean')
n_epochs = 30

train_loss = []
val_loss = []
best_loss = np.inf

for epoch in tqdm.tqdm_notebook(range(0, n_epochs)):
    scheduler.step(epoch)
    t_loss = train_validate(model, dataloader_train, optim, loss_fn, train=True)
    print('\n epoch:{} ---- training loss:{}'.format(epoch, t_loss))
    train_loss.append(t_loss)

    if epoch % 5 == 0:
        v_loss = train_validate(model, dataloader_val, optim, loss_fn, train=False)
        print('\n epoch:{} ---- validation loss:{}'.format(epoch, v_loss))
        val_loss.append(v_loss)
    if v_loss < best_loss:
        best_loss = v_loss
        print('Writing model checkpoint')
        torch.save(model.state_dict(), 'models/pixelcnn_'+str(n_epochs)+ 'epochs_batch_'+str(batch_size)+'.pt')


sns.set()
plt.rcParams['figure.figsize'] = 5, 5
plt.plot(np.arange(len(train_loss)), train_loss, label='train')
plt.plot(np.arange(0, len(val_loss) * 5, 5), val_loss, label='val')
plt.title('Training')
plt.xlabel('Step')
plt.ylabel('BCE')
plt.legend()
plt.grid(True)
plt.savefig('Figs/training_'+str(n_epochs)+ 'epochs.png')


