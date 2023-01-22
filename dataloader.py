import torch, torchvision
from torchvision.datasets import CIFAR100, CIFAR10
import copy
import matplotlib.pyplot as plt

class data():
  def __init__(self, dataset, num_clients, model):
    self.dataset = dataset
    self.num_clients = num_clients
    self.train_data = None
    self.test_data = None
    self.BATCH_SIZE = 64
    self.call = 0
    self.model = model


  def split_data(self, dataSplit):
    self.call += 1
    if self.call == 1:
      name_DS = 'train'
    elif self.call == 2:
      name_DS = 'test'
    else:
      name_DS = 'dataset'
    train_cpy = copy.deepcopy(self.num_clients)

    if dataSplit.data.shape[0]%train_cpy != 0:
      for _ in range(dataSplit.data.shape[0]):
        train_cpy += 1
        if dataSplit.data.shape[0]%train_cpy == 0:
          print(f"The {name_DS} set is divided into {train_cpy} parts")
          break
        else:
          continue
      if dataSplit.data.shape[0]%train_cpy != 0:
        print("Sorry we couldn't divide your data properly with that number of clients")
    else:
      print(f"No change made, {train_cpy} divides evenly into {dataSplit.data.shape[0]}")

    return train_cpy

  def returnData(self):

    train_split_num = self.split_data(self.train_data)
    print(f"there are: {self.test_data.data.shape[0]} test samples and {self.train_data.data.shape[0]} train samples")
    test_split_num = self.split_data(self.test_data)
    train_dataCopy = copy.deepcopy(self.train_data)
    test_dataCopy = copy.deepcopy(self.test_data)

    traindata_split = torch.utils.data.random_split(train_dataCopy, [int(train_dataCopy.data.shape[0]/train_split_num) for _ in range(train_split_num)])

    testdata_split = torch.utils.data.random_split(test_dataCopy, [int(test_dataCopy.data.shape[0]/test_split_num) for _ in range(test_split_num)])


    #pin_memory automatically puts fetched data tensors in pinned memory
    train_dl = [torch.utils.data.DataLoader(x, 
    batch_size = self.BATCH_SIZE,
    shuffle = True,
    pin_memory=True) for x in traindata_split]

    test_dl = [torch.utils.data.DataLoader(y, 
    batch_size = self.BATCH_SIZE,
    shuffle = False,
    pin_memory=True) for y in testdata_split]

    return train_dl, test_dl

  def loadData(self):
    tt = torchvision.transforms

    if self.dataset == 'mnist':
      if self.model == 'vgg':
        train_transform = tt.Compose([tt.Resize(size=(224, 224)), tt.ToTensor(), tt.Normalize((0.1307,), (0.3081,))])
        test_transform  = tt.Compose([tt.Resize(size=(224, 224)), tt.ToTensor(), tt.Normalize((0.1307,), (0.3081))])
      else:
        train_transform = tt.Compose([tt.ToTensor(), tt.Normalize((0.1307,), (0.3081,))])
        test_transform  = tt.Compose([tt.ToTensor(), tt.Normalize((0.1307,), (0.3081))])
      train_data = torchvision.datasets.MNIST(root = './mnist', train = True, download = True, transform = train_transform)
      test_data = torchvision.datasets.MNIST(root = './mnist', train = False, download = True, transform = test_transform)

    elif self.dataset == 'cifar10':
      if self.model == 'vgg':
        train_transform = tt.Compose([tt.Resize(size=(224,224)),
        tt.ToTensor(),
        tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])

        test_transform = tt.Compose([tt.Resize(size=(224,224)),
        tt.ToTensor(),
        tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])
      else:
        train_transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])

        test_transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])

      train_data = CIFAR10(download=True, root ="./cifar10", transform = train_transform)
      test_data = CIFAR10(root ="./cifar10", train = False, transform = test_transform)

    elif self.dataset == 'cifar100':
      #online calculated stats for cifar100
      stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
      
      if self.model == 'vgg':
        train_transform = tt.Compose([tt.Resize(size=(224,224)),
        tt.RandomHorizontalFlip(),
        tt.RandomCrop(32, padding =4, padding_mode = "reflect"),
        tt.ToTensor(),
        tt.Normalize(*stats)
      ])
      

        test_transform = tt.Compose([tt.Resize(size=(224,224)),
        tt.ToTensor(),
        tt.Normalize(*stats)
      ])
      else:
      #read documentation on each function below
      #it's to prevent overfitting by adding noise
        train_transform = tt.Compose([
        tt.RandomHorizontalFlip(),
        tt.RandomCrop(32, padding =4, padding_mode = "reflect"),
        tt.ToTensor(),
        tt.Normalize(*stats)
      ])
      

        test_transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize(*stats)
      ])

      train_data = CIFAR100(download=True, root ="./cifar100", transform = train_transform)
      test_data = CIFAR100(root = "./cifar100", train=False, transform=test_transform)

    else:
      print(f"You have entered {self.dataset} which is an invalid dataset")

    print(f"For the {self.dataset} dataset, we are trying to classify {len(train_data.classes)} different classes of the following type:")
    print(train_data.classes)

    self.train_data = train_data
    self.test_data = test_data

    return self.returnData()


  





# def load_data2(num_clients):
#   BATCH_SIZE = 512
#   train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
#   test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

#   train_data = torchvision.datasets.MNIST('mnist_data', train= True, download = True, transform = train_transform)
#   test_data = torchvision.datasets.MNIST('mnist_data', train = False, download = True, transform = test_transform)
  
#   train_dl = []
#   test_dl = []
  
#   for x in range(num_clients):
#     train_dl.append(torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True))
#     test_dl.append(torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = True))
#   return train_dl, test_dl