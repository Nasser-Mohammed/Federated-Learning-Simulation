import torch
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
import random
import numpy as np 
torch.backends.cudnn.benchmark = True
import dataloader
import server
from memory_profiler import profile
from multiprocessing import Pool
from timeit import default_timer as timer
import client

#import torch.multiprocessing as mp
import multiprocessing

def train(client):
    client_to_train = client[1].client_list[client[0]]
    client_to_train.client_training()
def main():
    
    multiprocessing.set_start_method('spawn')
    nameList = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"We are using: {device}")
    client_list = {}

    print(f"Choose one of the following options:")
    print("[1] MNIST on simple CNN")
    print("[2] CIFAR10 on simple CNN")
    print("[3] CIFAR10 on VGG")
    #resnet and cifar100 dataset not compatible yet
    #print("[4] CIFAR10 on ResNet")
    #print("[5] CIFAR100 on VGG")
    #print("[6] CIFAR100 on ResNet")

    choice = int(input())
    
    if choice == 1:
        op1, op2 = 'mnist', 'simple_cnn'
    elif choice == 2:
        op1, op2 = 'cifar10', 'simple_cnn'
    elif choice == 3:
        op1, op2 = 'cifar10', 'vgg'
    # elif choice == 4:
    #     op1, op2 = 'cifar10', 'resnet'
    # elif choice == 5:
    #     op1, op2 = 'cifar100', 'vgg'
    # elif choice == 6:
    #     op1, op2 = 'cifar100', 'resnet'
    else:
        print("Invalid option")

    print("Enter the number of clients to connect to the server and number of rounds to train the models for. Separated by '-'")
    num_clients, num_rounds = input().split('-')
    num_clients = int(num_clients)
    num_rounds = int(num_rounds)
    if num_clients < 1 or num_rounds < 1:
        print(f"You entered bad parameters")
        exit()


    model_type = op2


    datasetName = op1
    dataObj = dataloader.data(datasetName, num_clients, model_type)
    data = dataObj.loadData()
    
    train_dl = data[0]
    test_dl = data[1]



    print("Lastly, enter '1' for series processing, and any other number for parallel processses")
    mode = int(input())


    serv = server.Server(num_clients, num_rounds, device, client_list, data[1], datasetName, model_type)
    serv.initialize_clients()

    z = 0
    for client in range(num_clients):
        cpyStr = "client " + str(z)
        #looping through the server's dictionary of clients, then accessing each client
        #based off of their key 'cpyStr' which returns a client at that key, then we access
        #their dataloaders with .train_dl and .test_dl and we set them equal to one of the #shards of the dataset we split
        serv.client_list[cpyStr].train_dl = train_dl[client]
        serv.client_list[cpyStr].test_dl = test_dl[client]
        #you can use the commented print statements below to verify each client, and their
        #dataset is unique
        #print(f"Gathering training data for {cpyStr} at location {serv.client_list[cpyStr]}")
        #print(f"We gave {cpyStr} the data loader: {train_dl[client]}, and {test_dl[client]}")
        z+=1

    print("-------------------------------------------------------------------------------------------------")
    
    #@jit(target_backend = 'cuda')
    #def train(key):
       # print(f"Training {key} at address: {serv.client_list[key]}, Round: {round+1}")
        #serv.client_list[key].client_training()
    #modelDict = {}
    allClient = 0
    acc = {}
    avgTime = 0
    rndLength = 0
    for rnd in range(num_rounds):
        print(f"ROUND: {rnd+1}/{num_rounds}")
        print("-------------------------------------------------------------------------------------------------")
        C = random.random()
        print(f"We are choosing {round(C,2)}% of clients for this round")
        #num_selected = random.randint(1, int(float(serv.num_clients)*C))
        num_selected = max(int(float(num_clients)*C), 1)
        client_index = np.random.permutation(num_clients)[:num_selected]
        #uncomment below to verify list of client indices
        #print(f"Order of clients is: {client_index}")
        processes = []
        
        start = timer()
        nameList = []
        if mode == 1:
            print(f"-----------TRAINING-----------")
            for index, client in enumerate(client_index):
                key = 'client ' + str(client)
                #uncomment statement below to verify client object is training
                #print(f"training {key} at address: {serv.client_list[key]}, Round: {round +1}")
                serv.client_list[key].client_training()
                nameList.append(key)
                
        else:
            print(f"using parallel processes. Spawning Processes.......")
            if len(client_index) <= 3:
                processes = 1
            elif len(client_index) < 15:
                processes = 3
            else:
                processes = 5
            pool = Pool(processes = processes)
            args = ['client ' + str(x) for x in client_index]
            #args2 = [serv.client_list for x in args]
            nameList = [x for x in args]
        

            servIt = [serv for y in args]
            arg = zip(args, servIt)
           # print(arg)
        #cant do it this way try another
            pool.map(train, arg)
            pool.close()
            pool.join()


        finishTime = timer()
        allClient += len(client_index)
        print(f"Round {rnd+1} took: {finishTime - start} seconds")
        rndLength += finishTime - start
        
        #aggregate all trained models
        serv.server_merge(nameList)
        #test new global model accuracy
        tmp = serv.test()
        print(f"Accuracy of new global model is {tmp}%")

        acc[rnd] = tmp.to('cpu')
    rndAvg = rndLength/serv.num_rounds



    # print(f"Average round time: {rndAvg}")
    # print(f"Average time per client: {rndLength/allClient}")
    # print(f"Average number of clients per round: {allClient/serv.num_rounds}")
    # print(f"Total number of trained clients: {allClient}")
    print(f"Total Execution Time: {rndLength}")
    # print(f"Total number of rounds: {serv.num_rounds}")
    newAcc = sorted(acc.items())
    x,y = zip(*newAcc)
    plt.plot(x,y)
    plt.xlabel("Number of Rounds")
    plt.ylabel("Accuracy of Global Model")
    plt.title("Communication Rounds Effect on Accuracy")
    plt.savefig(op1+op2+".png")
    #plt.show()

if __name__ == "__main__":
    main()
