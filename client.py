import torch
from torch import nn
from torch import optim
from numba import jit, cuda
import numpy as np



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.001
lf = nn.CrossEntropyLoss()

class Client:
    def __init__(self, model, client_id):
        self.model = model
        self.client_id = client_id
        self.train_dl = None
        self.test_dl = None
        self.epochs = 5
        self.optimizer = optim.SGD(self.model.parameters(), lr)

    
    def client_training(self):
        self.model.train()
        for e in range(0,self.epochs):
            
            ttl = 0
            crct = 0
            trainAccuracy = 0.0
            loss = 0
            prediction = None
            for batch_index, (data, target) in enumerate(self.train_dl):
                data, target = data.to(device), target.to(device)
                self.optimizer.zero_grad()
                prediction = self.model(data)
                ttl += prediction.size(0)
                loss = lf(prediction, target)
                loss.backward()
                self.optimizer.step()
                crct += (prediction.argmax(1) == target).sum()
            trainAccuracy = crct*100./ttl
            with torch.no_grad():
                self.model.eval()
                total = 0
                correct = 0
                validationAccuracy = 0
                prediction2 = None
                for i, (x, y) in enumerate(self.test_dl):
                    (x,y) = (x.to(device), y.to(device))
                    prediction2 = self.model(x)
                    total += prediction2.size(0)
                    correct += (prediction2.argmax(1) == y).sum()

            validationAccuracy = correct*100./total
            print(f"EPOCH {e+1}/{self.epochs} SUMMARY FOR CLIENT {self.client_id}:")
            print(f"Train Accuracy: {trainAccuracy}%, and Validation Accuracy is {validationAccuracy}%")
            print("==============================================")
        
        return self.model.state_dict()

              
