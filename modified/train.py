import torch
from models import Encoder, GlobalDiscriminator, LocalDiscriminator, PriorDiscriminator, Classification
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
import statistics as stats
import argparse
import numpy as np
import math
from customdataset import MyCustomDataset

class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0, gamma=0.1):
        super().__init__()
        self.global_d = GlobalDiscriminator()
        self.local_d = LocalDiscriminator()
        self.prior_d = PriorDiscriminator()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y, M, M_prime):

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        y_exp = y.unsqueeze(-1).unsqueeze(-1)
        y_exp = y_exp.expand(-1, -1, 26, 26)

        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        a = self.local_d(y_M);
        Ej = -F.softplus(-a).mean()
        b = self.local_d(y_M_prime);
        Em = F.softplus(b).mean()
        LOCAL = (Em - Ej) * self.beta

        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha

        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
        return LOCAL + GLOBAL + PRIOR


if __name__ == '__main__':

    torch.manual_seed(1);

    parser = argparse.ArgumentParser(description='DeepInfomax pytorch')
    parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size

    # image size 3, 32, 32
    # batch size must be an even number
    # shuffle must be True

    cifar_10_train_dt = MyCustomDataset('data',  download=True, transform=ToTensor())
    cifar_10_train_l = DataLoader(cifar_10_train_dt, batch_size=batch_size, shuffle=True, drop_last=True,
                                  pin_memory=torch.cuda.is_available())

    loss_fn = DeepInfoMaxLoss(0, 1, 0.1).to(device)
    loss_fn = DeepInfoMaxLoss().to(device)
    classification = Classification().to(device)
    encoder_optim = Adam(encoder.parameters(), lr=1e-4)
    loss_optim = Adam(loss_fn.parameters(), lr=1e-4)
    classification_optim = Adam(classification.parameters(), lr=1e-4)

    epoch_restart = 0
    root = Path(r'models')

    if epoch_restart > 0 and root is not None:
        enc_file = root / Path('encoder' + str(epoch_restart) + '.wgt')
        loss_file = root / Path('loss' + str(epoch_restart) + '.wgt')
        classification_loss_file = root / Path('classification_loss' + str(epoch_restart) + '.wgt')
        encoder.load_state_dict(torch.load(str(enc_file)))
        loss_fn.load_state_dict(torch.load(str(loss_file)))
        classification.load_state_dict(torch.load(str(classification_loss_file)))

    for epoch in range(epoch_restart + 1, 501):
        batch = tqdm(cifar_10_train_l, total=len(cifar_10_train_dt) // batch_size)
        train_loss = []
        for x, target, rot in batch:
            x = x.to(device)
            rot = rot.to(device)
            encoder_optim.zero_grad()
            loss_optim.zero_grad()
            classification_optim.zero_grad()
            y, M = encoder(x)
            predicted_value = classification(y)
            criterion = nn.CrossEntropyLoss()
            loss_ss = criterion(predicted_value, rot)
            M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
            loss_mutual_information = loss_fn(y, M, M_prime)
            loss_classification = loss_ss
            loss_total = loss_mutual_information + loss_classification
            train_loss.append(loss_total.item())
            batch.set_description(str(epoch) + ' Loss: ' + str(stats.mean(train_loss[-1:])))
            loss_total.backward()
            encoder_optim.step()
            loss_optim.step()
            classification_optim.step()

        if epoch % 10 == 0:
            root = Path(r'models')
            enc_file = root / Path('encoder' + str(epoch) + '.wgt')
            loss_file = root / Path('loss' + str(epoch) + '.wgt')
            classification_loss_file = root / Path('classification_loss' + str(epoch) + '.wgt')
            enc_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(encoder.state_dict(), str(enc_file))
            torch.save(loss_fn.state_dict(), str(loss_file))
            torch.save(classification.state_dict(), str(classification_loss_file))
