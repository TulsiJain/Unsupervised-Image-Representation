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
from classification import Classification
# from customdataset import MyCustomDataset
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
cifar_10_train_dt = CIFAR10(r'data', train=False, download=False, transform=ToTensor())
#dev = Subset(cifar_10_train_dt, range(128))
cifar_10_train_l = DataLoader(cifar_10_train_dt, batch_size=batch_size, shuffle=False,
                              pin_memory=torch.cuda.is_available())

encoder = models.Encoder()
classification = Classification()

root = Path(r'modified/models')
encoder_path = root / Path(r'encoder500.wgt')
encoder.load_state_dict(torch.load(str(encoder_path)))
encoder.to(device)
root_classification = Path(r'classification_model_baseline_modified')
classification_model_path = root_classification / Path(r'classification_loss50.wgt')
classification.load_state_dict(torch.load(str(classification_model_path)))
classification.to(device)

batch = tqdm(cifar_10_train_l, total=len(cifar_10_train_dt) // batch_size)
correct = 0
for images, target in batch:
    images = images.to(device)
    target = target.to(device)
    encoded, features = encoder(images)
    predicted_value = classification(encoded)
    batch.set_description("hello")
    pred = predicted_value.data.max(1, keepdim=True)[1] 
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
print('\n Testing Accuracy: {}/{} ({:.4f}%)\n'.format(
    correct, len(cifar_10_train_dt),
    100. * correct / len(cifar_10_train_dt)))



