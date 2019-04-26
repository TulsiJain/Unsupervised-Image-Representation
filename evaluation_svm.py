import torch
import models
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from torchvision.datasets.cifar import CIFAR10
from tqdm import tqdm
import statistics as stats
from torchvision.transforms import ToTensor, ToPILImage
from sklearn.svm import SVC
import numpy as np
# from matplotlib import pyplot as plt
# from torchvision.utils import save_image
# import numpy as np
# from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64

# image size 3, 32, 32
# batch size must be an even number
# shuffle must be True
cifar_10_train_dt = CIFAR10(r'data', train=True, download=False, transform=ToTensor())
#dev = Subset(cifar_10_train_dt, range(128))
cifar_10_train_l = DataLoader(cifar_10_train_dt, batch_size=batch_size, shuffle=False,
                              pin_memory=torch.cuda.is_available())
encoder = models.Encoder()
root = Path(r'baseline_jsn/models')
encoder_path = root / Path(r'encoder320.wgt')
encoder.load_state_dict(torch.load(str(encoder_path)))
encoder.to(device)
encoder.eval()
batch = tqdm(cifar_10_train_l, total=len(cifar_10_train_dt) // batch_size)
correct = 0
train_data = []
labels = []
for images, target in batch:
    images = images.to(device)
    target = target.to(device)
    encoded, features = encoder(images)
    encoded = torch.Tensor.cpu(encoded).detach().numpy()
    target = torch.Tensor.cpu(target).detach().numpy()
    train_data.append(encoded)
    labels.append(target)
    batch.set_description("hello")

train_data = np.concatenate(train_data, axis=0)
labels = np.concatenate(labels, axis=0)
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(train_data, labels)

cifar_10_test_dt = CIFAR10(r'data', train=False, download=False, transform=ToTensor())
cifar_10_test_l = DataLoader(cifar_10_test_dt, batch_size=batch_size, shuffle=False,
                              pin_memory=torch.cuda.is_available())
batch = tqdm(cifar_10_test_l, total=len(cifar_10_test_dt) // batch_size)
correct = 0
test_data = []
test_labels = []
for images, target in batch:
    images = images.to(device)
    target = target.to(device)
    encoded, features = encoder(images)
    encoded = torch.Tensor.cpu(encoded).detach().numpy()
    target = torch.Tensor.cpu(target).detach().numpy()
    test_data.append(encoded)
    test_labels.append(target)
    batch.set_description("hello")

test_data = np.concatenate(test_data, axis=0)
test_labels = np.concatenate(test_labels, axis=0)
correct = 0
svm_predictions = svm_model_linear.predict(test_data) 
correct = np.sum(test_labels == svm_predictions)
print('\n Training Accuracy: {}/{} ({:.4f}%)\n'.format(
    correct, len(cifar_10_test_dt),
    100. * correct / len(cifar_10_test_dt)))