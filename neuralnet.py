import torch
from torch import optim
import time
import torchvision
import torch.nn as nn
import torch.nn.functional as F




config= {'simple_cnn': {'mnist':[(1,32,3), (2,2), (32, 64, 5), (1024, 200), (200, 84), (84, 10)], 'cifar10': [(3,6,5), (2,2), (6,16,5), (16*5*5, 120), (120, 84), (84,10)]},'vgg': {'cifar10': [64, 64, 'M', 128,128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 'cifar100':[]},'resnet': {'mnist': [], 'cifar10': [], 'cifar100': []}}



class VGG(nn.Module):
    def __init__(self, dataset):
        super(VGG, self).__init__()
        self.param = config['vgg'][dataset]
        self.in_channels = 3
        self.num_classes = 10 if dataset == 'cifar10' else 100
        self.conv_layers = self.create_conv_layers(config['vgg'][dataset])

        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [nn.Conv2d(in_channels, out_channels, kernel_size = (3,3), stride=(1,1), padding=(1,1)), nn.BatchNorm2d(x), nn.ReLU()]

                in_channels = x
            
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]

        return nn.Sequential(*layers)



class CNN(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.param = config['simple_cnn'][dataset]
        self.conv1 = nn.Conv2d(*self.param[0])
        self.pool = nn.MaxPool2d(*self.param[1])
        self.conv2 = nn.Conv2d(*self.param[2])
        self.fc1 = nn.Linear(*self.param[3])
        self.fc2 = nn.Linear(*self.param[4])
        self.fc3 = nn.Linear(*self.param[5])

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def create_model(model_name, dataset):
    if model_name == 'simple_cnn':
        return CNN(dataset)
    elif model_name == 'vgg':
        return VGG(dataset)
    elif model_name == 'resnet':
        return Resnet(dataset)






def create_cnn11(dataset):

    model = nn.Sequential(
        nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1),
        nn.ReLU(),
        nn.MaxPool2d(2, stride = 2),

        nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1), 
        nn.ReLU(),
        nn.MaxPool2d(2, stride = 2),

        nn.Flatten(),
        nn.Linear(1024, 200),
        nn.ReLU(),
        nn.Linear(200, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
)
    return model





