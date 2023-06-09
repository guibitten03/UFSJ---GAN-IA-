import torch
from torchvision import datasets
import torchvision.transforms as transforms


class DataLoader():
    def __init__(self, num_workers:int=0):
        # convert data to torch.FloatTensor
        transform = transforms.ToTensor()
        # Download and load the training data
        dataset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
        
        self.dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,  num_workers=num_workers)