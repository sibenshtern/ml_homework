from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    transforms.Resize((224, 224))
])


def prepare_train():
    return CIFAR10("CIFAR10/train", download=True, transform=transform)


def prepare_test():
    return CIFAR10("CIFAR10/test", train=False, download=True,
                   transform=transform)
