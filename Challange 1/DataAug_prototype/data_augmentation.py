import logging
import sys
import torch
import torch.nn as nn 
import torch.nn.functional as F
from numpy.random import RandomState
import numpy as np
import torch.optim as optim
from torch.utils.data import Subset
from torchvision import datasets, transforms


# configure the logging module
# logger = logging.getLogger('my_logger')
# logging.basicConfig(filename='training.log',  filemode='a', level=logging.INFO ,format='%(asctime)s %(message)s')
# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.FileHandler('challange1-dataaug.log')
# Create STDERR handler
# handler = logging.StreamHandler(sys.stderr)
# ch.setLevel(logging.DEBUG)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Set STDERR handler as the only handler 
logger.handlers = [handler]



def train(model, device, train_loader, optimizer, epoch, growth=4, display=True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        data_aug = []
        for i in range(growth):
            data_aug.append(torch.cat([transform_augment(data[i]).unsqueeze(0) for i in range(data.shape[0])]))
        data_aug = torch.cat(data_aug, dim=0).to(device)

        target_aug = target.repeat(growth).to(device)
        
        optimizer.zero_grad()
        output = model(data_aug)
        # print(output.shape)
        loss = F.cross_entropy(output, target_aug)
        loss.backward()
        optimizer.step()
    if display:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
        logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        
        self.layers+=[nn.Conv2d(3, 16,  kernel_size=3) , 
                      nn.ReLU(inplace=True)]
        self.layers+=[nn.Conv2d(16, 16,  kernel_size=3, stride=2), 
                      nn.ReLU(inplace=True)]
        self.layers+=[nn.Conv2d(16, 32,  kernel_size=3), 
                      nn.ReLU(inplace=True)]
        self.layers+=[nn.Conv2d(32, 32,  kernel_size=3, stride=2), 
                      nn.ReLU(inplace=True)]
        self.fc = nn.Linear(32*5*5, 1)
    def forward(self, x):
        for i in range(len(self.layers)):
          x = self.layers[i](x)
        x = x.view(-1, 32*5*5)
        x = self.fc(x)
        return x
	
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))

transform_val = transforms.Compose([transforms.ToTensor(), normalize]) #careful to keep this one same
transform_train = transforms.Compose([transforms.ToTensor()]) 
transform_augment = transforms.Compose([transforms.ToPILImage(),
                                        transforms.RandomHorizontalFlip(),
                                      transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                      transforms.ToTensor(),
                                      normalize])

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

##### Cifar Data
cifar_data = datasets.CIFAR10(root='.',train=True, transform=transform_train, download=True)
    
#We need two copies of this due to weird dataset api 
cifar_data_val = datasets.CIFAR10(root='.',train=True, transform=transform_val, download=True)

import itertools

def generate_hyperparam_permutations():
    lr = [1e-2, 1e-3, 5e-3, 1e-4, 3e-4, 5e-4]
    epochs = [10,20,30,40,50,100,150]
    batch_size = [8,16,32,64]
    growth_rate = [1,2,4,8]
    param_permutations = list(itertools.product(lr, epochs, batch_size, growth_rate))
    return param_permutations

for (lr, epochs, batch_size, growth_rate) in generate_hyperparam_permutations():
    print(f"{'-='*50}\nTraining with lr={lr}, epochs={epochs}, batch_size={batch_size}, growth_rate={growth_rate}")
    logger.info(f"{'-='*50}\nTraining with lr={lr}, epochs={epochs}, batch_size={batch_size}, growth_rate={growth_rate}")

    accs = []
    for seed in range(1, 5):
        prng = RandomState(seed)
        random_permute = prng.permutation(np.arange(0, 1000))
        classes =  prng.permutation(np.arange(0,10))
        indx_train = np.concatenate([np.where(np.array(cifar_data.targets) == classe)[0][random_permute[0:25]] for classe in classes[0:2]])
        indx_val = np.concatenate([np.where(np.array(cifar_data.targets) == classe)[0][random_permute[25:525]] for classe in classes[0:2]])

        train_data = Subset(cifar_data, indx_train)
        val_data = Subset(cifar_data_val, indx_val)

        # print('Num Samples For Training %d Num Samples For Val %d'%(train_data.indices.shape[0],val_data.indices.shape[0]))

        train_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=batch_size, 
                                                    shuffle=True)

        val_loader = torch.utils.data.DataLoader(val_data,
                                            batch_size=batch_size, 
                                            shuffle=False)
    

        model = Net()
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(),lr=lr, momentum=0.9,
                                    weight_decay=0.0005)
        for epoch in range(epochs):
            train(model, device, train_loader, optimizer, epoch, growth=growth_rate, display=epoch%10==0)
        
        accs.append(test(model, device, val_loader))

    accs = np.array(accs)
    print('Acc over 5 instances: %.2f +- %.2f'%(accs.mean(),accs.std()))
    logger.info('Acc over 5 instances: %.2f +- %.2f'%(accs.mean(),accs.std()))
