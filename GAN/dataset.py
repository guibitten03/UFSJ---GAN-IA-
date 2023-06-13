from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class data():
    # Numero de subprocessos que carregarão os dados
    num_workers = 0

    # Converter a data para um tensor de acordo com parametros
    transform = transforms.ToTensor()
    # Download and load the training data
    dataset = datasets.FashionMNIST('../dataset', download=True, train=True, transform=transform)
    dataset_loader = DataLoader(dataset, batch_size=64, shuffle=True,  num_workers=num_workers)