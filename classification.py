import torch.nn as nn
import torch.nn.functional as F

class Classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(64, 200)
        self.dropout = nn.Dropout(0.1)
        self.l2 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.dropout(x)
        x = self.l2(x)        
        return F.log_softmax(x, dim=1)
