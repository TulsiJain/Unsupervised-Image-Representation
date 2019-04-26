import torch
import models
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from torchvision.datasets.cifar import CIFAR10
from tqdm import tqdm
import statistics as stats
from torchvision.transforms import ToTensor
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.utils.linear_assignment_ import linear_assignment
# from matplotlib import pyplot as plt
# from torchvision.utils import save_image
# import numpy as np
# from PIL import Image


def NMI(y_true,y_pred):
    return metrics.normalized_mutual_info_score(y_true, y_pred)

def ARI(y_true,y_pred):
    return metrics.adjusted_rand_score(y_true, y_pred)

def ACC(y_true,y_pred):
    Y_pred = y_pred
    Y = y_true
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
        ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size,ind

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

kmeans = KMeans(n_clusters=10, random_state=0).fit(train_data)

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
Y = np.concatenate(test_labels, axis=0)
Y_pred = kmeans.fit_predict(test_data) 

print(NMI(Y, Y_pred))
print(ARI(Y, Y_pred))
print(ACC(Y, Y_pred))