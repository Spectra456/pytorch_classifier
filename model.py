from torch import nn, sigmoid, softmax 
import torch.nn.functional as F


class ImageClassifier(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding = 1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding= 1)
        self.conv2_drop = nn.Dropout2d()
        input_size = int((128*(n/4)*(n/4)))
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 5)
 
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return softmax(x, dim=1)