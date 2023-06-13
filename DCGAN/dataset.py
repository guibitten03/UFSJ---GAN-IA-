from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

class data():
    # define the transform
    data_transforms = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
    ])
    # load the dataset 
    dataset = FashionMNIST(root="../dataset", train=True, transform=data_transforms, download=True)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)