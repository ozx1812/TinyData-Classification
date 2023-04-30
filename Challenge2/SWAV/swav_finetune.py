from numpy.random import RandomState
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset


from torchvision import datasets, transforms
import itertools
import logging
import sys

# configure the logging module
# logger = logging.getLogger('my_logger')
# logging.basicConfig(filename='training.log',  filemode='a', level=logging.INFO ,format='%(asctime)s %(message)s')
# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.FileHandler("challange-2-ft.log")
# Create STDERR handler
# handler = logging.StreamHandler(sys.stderr)
# ch.setLevel(logging.DEBUG)

# Create formatter and add it to the handler
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Set STDERR handler as the only handler
logger.handlers = [handler]


def train(model, device, train_loader, optimizer, epoch, display=True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    if display:
        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item(),
            )
        )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(
                output, target, size_average=False
            ).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return 100.0 * correct / len(test_loader.dataset)


crop = transforms.RandomResizedCrop(224)
hflip = transforms.RandomHorizontalFlip()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


transform = transforms.Compose(
    [
        crop,
        transforms.RandomApply([hflip], p=0.5),
        normalize,
    ]
)


# We resize images to allow using imagenet pre-trained models, is there a better way?
resize = transforms.Resize(224)

transform_val = transforms.Compose(
    [resize, transforms.ToTensor(), transform]
)  # careful to keep this one same
transform_train = transforms.Compose([resize, transforms.ToTensor(), transform])

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)  # you will really need gpu's for this part

##### Cifar Data
cifar_data = datasets.CIFAR10(
    root=".", train=True, transform=transform_train, download=True
)

# We need two copies of this due to weird dataset api
cifar_data_val = datasets.CIFAR10(
    root=".", train=True, transform=transform_val, download=True
)


def generate_hyperparam_permutations():
    lr = [1e-2, 1e-3, 5e-3, 1e-4, 3e-4, 5e-4]
    epochs = [10, 20, 30, 40, 50, 100, 150]
    batch_size = [8, 16, 32, 64]
    param_permutations = list(itertools.product(lr, epochs, batch_size))
    return param_permutations


for lr, epochs, batch_size in generate_hyperparam_permutations():
    print(f"{'-='*50}\nTraining with lr={lr}, epochs={epochs}, batch_size={batch_size}")
    logger.info(
        f"{'-='*50}\nTraining with lr={lr}, epochs={epochs}, batch_size={batch_size}"
    )
    accs = []
    for seed in range(1, 25):
        prng = RandomState(seed)
        random_permute = prng.permutation(np.arange(0, 5000))
        classes = prng.permutation(np.arange(0, 10))
        indx_train = np.concatenate(
            [
                np.where(np.array(cifar_data.targets) == classe)[0][
                    random_permute[0:25]
                ]
                for classe in classes[0:2]
            ]
        )
        indx_val = np.concatenate(
            [
                np.where(np.array(cifar_data.targets) == classe)[0][
                    random_permute[25:225]
                ]
                for classe in classes[0:2]
            ]
        )

        train_data = Subset(cifar_data, indx_train)
        val_data = Subset(cifar_data_val, indx_val)

        # print('Num Samples For Training %d Num Samples For Val %d'%(train_data.indices.shape[0],val_data.indices.shape[0]))

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=False
        )

        # ORIGINAL #
        # model = models.alexnet(pretrained=True)
        # model.classifier = nn.Linear(256 * 6 * 6, 10)
        ############
        model = torch.hub.load("facebookresearch/swav:main", "resnet50")
        model.fc = nn.Linear(2048, 10)

        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

        optimizer = torch.optim.SGD(
            model.fc.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
        )
        model.to(device)
        for epoch in range(epochs):
            train(
                model,
                device,
                train_loader,
                optimizer,
                epoch,
                display=epoch % (epochs // 10) == 0,
            )

        accs.append(test(model, device, val_loader))

    accs = np.array(accs)
    print("Acc over 25 instances: %.2f +- %.2f" % (accs.mean(), accs.std()))
    logger.info("Acc over 25 instances: %.2f +- %.2f" % (accs.mean(), accs.std()))

