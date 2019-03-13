# Not mine, source: https://github.com/pytorch/examples/blob/master/mnist/main.py

from __future__ import print_function

import argparse, os, torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter

from logger import Logger

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def train(args, model, device, train_loader, optimizer, epoch, train_fraction):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), train_fraction*len(train_loader.dataset),
                100. * batch_idx * len(data) / ( train_fraction * len(train_loader.dataset) ), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= ( test_loader.batch_size * len(test_loader) )
    accuracy = 100. * correct / ( test_loader.batch_size * len(test_loader) )
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_loader.batch_size * len(test_loader),
        #100. * correct / len(test_loader.dataset)))
        accuracy))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dataset = datasets.MNIST(os.path.join( '..', 'data' ), train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    dataset_size = len( dataset )
    indices = list( range( dataset_size ) )
    validation_split = 0.1
    split = int( np.floor( validation_split * dataset_size ) )
    
    np.random.seed( args.seed )
    np.random.shuffle( indices )
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler( train_indices )
    valid_sampler = SubsetRandomSampler( val_indices )

    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler
        # shuffle=True # already shuffled
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size,
        sampler=valid_sampler,
        # shuffle=True # already shuffled
    )

    '''   
    # original stuff, had this for train_loader as well
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join( '..', 'data' ), train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    '''
    
    print( f"train_loader has a size of {len(train_loader)}\n" )
    print( f"validation_loader has a size of {len(validation_loader)}\n" )

    logger = Logger('./logs')

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, train_fraction=( 1-validation_split ) )
        test(args, model, device, validation_loader) # was test_loader

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
