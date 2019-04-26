import torch
import models
from pathlib import Path
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F
import random
import torch.nn as nn
from torch.optim import Adam
import statistics as stats
from customdataset import MyCustomDataset
from torchvision.transforms import ToTensor, ToPILImage
# from matplotlib import pyplot as plt
# from torchvision.utils import save_image
# import numpy as np
# from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64

# image size 3, 32, 32
# batch size must be an even number
# shuffle must be True
cifar_10_train_dt = CIFAR10(r'data', download=False, transform=ToTensor())
#dev = Subset(cifar_10_train_dt, range(128))
cifar_10_train_l = DataLoader(cifar_10_train_dt, batch_size=batch_size, shuffle=False,
                              pin_memory=torch.cuda.is_available())


# cifar_10_train_dt = CIFAR10(r'data1', train=False, download=True, transform=ToTensor())
# cifar_10_train_l = DataLoader(cifar_10_train_dt, batch_size=batch_size,  shuffle=False)


class Classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(64, 200)
        self.l2 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)        
        return F.log_softmax(x, dim=1)

encoder = models.Encoder()
classification = Classification().to(device)

root = Path(r'baseline/models')
model_path = root / Path(r'encoder320.wgt')
encoder.load_state_dict(torch.load(str(model_path)))
encoder.to(device)
classification_optim = Adam(classification.parameters(), lr=1e-4)


epoch_restart = 100
root_classification_model = Path(r'classification_model')

if epoch_restart > 0 and root is not None:
    classification_loss_file = root_classification_model / Path('classification_loss' + str(epoch_restart) + '.wgt')
    classification.load_state_dict(torch.load(str(classification_loss_file)))

for epoch in range(epoch_restart + 1, 201):
    batch = tqdm(cifar_10_train_l, total=len(cifar_10_train_dt) // batch_size)
    train_loss = []
    correct = 0
    for images, target in batch:
        images = images.to(device)
        target = target.to(device)
        encoded, features = encoder(images)
        classification_optim.zero_grad()
        predicted_value = classification(encoded)
        criterion = nn.CrossEntropyLoss()
        loss_total = criterion(predicted_value, target)
        train_loss.append(loss_total.item())
        loss_total.backward()
        classification_optim.step()
        batch.set_description(str(epoch) + ' Loss: ' + str(stats.mean(train_loss[-20:])))
        pred = predicted_value.data.max(1, keepdim=True)[1] 
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        if epoch % 5 == 0:
            root = Path(r'classification_model')
            classification_loss_file = root / Path('classification_loss' + str(epoch) + '.wgt')
            classification_loss_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(classification.state_dict(), str(classification_loss_file))
    print('\n Training Accuracy: {}/{} ({:.4f}%)\n'.format(
        correct, len(cifar_10_train_dt),
        100. * correct / len(cifar_10_train_dt)))



