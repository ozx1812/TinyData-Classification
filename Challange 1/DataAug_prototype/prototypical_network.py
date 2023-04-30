# Proto-typical network
import os
import re
import zipfile
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import logging

from typing import List
from torch.utils.data import Subset
import numpy as np

np.random.seed(41)

from torch.utils.data import Subset
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# configure the logging module
# logger = logging.getLogger('my_logger')
# logging.basicConfig(filename='training.log',  filemode='a', level=logging.INFO ,format='%(asctime)s %(message)s')
# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.FileHandler("training1-prototypical.log")
# Create STDERR handler
# handler = logging.StreamHandler(sys.stderr)
# ch.setLevel(logging.DEBUG)

# Create formatter and add it to the handler
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Set STDERR handler as the only handler
logger.handlers = [handler]


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def load_protonet_conv(**kwargs):
    """
    Loads the prototypical network model
    Arg:
        x_dim (tuple): dimension of input image
        hid_dim (int): dimension of hidden layers in conv blocks
        z_dim (int): dimension of embedded image
    Returns:
        Model (Class ProtoNet)
    """
    x_dim = kwargs["x_dim"]
    hid_dim = kwargs["hid_dim"]
    z_dim = kwargs["z_dim"]

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten(),
    )

    return ProtoNet(encoder)


def euclidean_dist(x, y):
    """
    Computes euclidean distance btw x and y
    Args:
        x (torch.Tensor): shape (n, d). n usually n_way*n_query
        y (torch.Tensor): shape (m, d). m usually n_way
    Returns:
        torch.Tensor: shape(n, m). For each query, the distances to each centroid
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class ProtoNet(nn.Module):
    def __init__(self, encoder):
        """
        Args:
            encoder : CNN encoding the images in sample
            n_way (int): number of classes in a classification task
            n_support (int): number of labeled examples per class in the support set
            n_query (int): number of labeled examples per class in the query set
        """
        super(ProtoNet, self).__init__()
        self.encoder = encoder.to(device)
        self.prototypes = None

    def save_prototypes(self, prototypes):
        if self.prototypes is None:
            self.prototypes = prototypes
        else:
            self.prototypes += prototypes
            self.prototypes /= 2

    def set_forward_loss(self, sample, target_inds, is_train=True):
        """
        Computes loss, accuracy and output for classification task
        Args:
            sample (torch.Tensor): shape (n_way, n_support+n_query, (dim))
        Returns:
            torch.Tensor: shape(2), loss, accuracy and y_hat
        """
        sample_images = sample.to(device)
        n_way = sample.size(0)
        n_support = sample.size(1) // 2
        n_query = sample.size(1) // 2

        x_support = sample_images[:, :n_support]
        x_query = sample_images[:, n_support:]

        target_inds = target_inds.to(device)

        # encode images of the support and the query set
        x = torch.cat(
            [
                x_support.contiguous().view(n_way * n_support, *x_support.size()[2:]),
                x_query.contiguous().view(n_way * n_query, *x_query.size()[2:]),
            ],
            0,
        )
        # print(f"before encoder {x.shape}")
        z = self.encoder.forward(x)
        z_dim = z.size(-1)  # usually 64
        z_proto = z[: n_way * n_support].view(n_way, n_support, z_dim).mean(1)
        z_query = z[n_way * n_support :]

        # print(z_proto.shape)
        # If the model is in training mode, save/update the prototypes
        if is_train:
            if self.prototypes is None:
                self.prototypes = z_proto.detach().clone()
            else:
                self.prototypes = torch.mean(
                    torch.stack([self.prototypes, z_proto.detach().clone()]), dim=0
                )

        # print(z_query.shape, self.prototypes.shape)
        # compute distances
        dists = euclidean_dist(z_query, self.prototypes)

        # compute probabilities
        log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            "loss": loss_val.item(),
            "acc": acc_val.item(),
            "y_hat": y_hat,
        }

    def predict_class_with_min_distance(self, sample):
        """
        given the sample - which is query at test time, calculate the distance to
        prototypes saved while training and then returning the minimum distance class.
        """
        with torch.no_grad():
            z_query = self.encoder.forward(sample)
            dists = euclidean_dist(z_query, self.prototypes)
            _, min_class_inds = torch.min(dists, dim=1)
            return min_class_inds


normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))

transform_val = transforms.Compose(
    [transforms.ToTensor(), normalize]
)  # careful to keep this one same

transform_train = transforms.Compose(
    [
        transforms.ToTensor(),
        normalize,
    ]
)

transform_augment = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        normalize,
    ]
)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

##### Cifar Data
cifar_data = datasets.CIFAR10(
    root=".", train=True, transform=transform_train, download=True
)

# We need two copies of this due to weird dataset api
cifar_data_val = datasets.CIFAR10(
    root=".", train=True, transform=transform_val, download=True
)




def select_two_classes_randomaly(class_ids: List):
    selected = np.random.choice(class_ids, size=2, replace=False)
    return selected


def get_random_train_test_data(cifar_data):
    train_X, train_Y, test_X = None, None, None
    classes = select_two_classes_randomaly(range(len(cifar_data.classes)))
    label_map = {classes[0]: 0, classes[1]: 1}

    targets = np.array(cifar_data.targets)
    subset_idx_0 = np.where(targets == classes[0])[0].tolist()
    subset_idx_1 = np.where(targets == classes[1])[0].tolist()

    # Shuffle the indices and split into train and test sets
    rng = np.random.default_rng()
    rng.shuffle(subset_idx_0)
    rng.shuffle(subset_idx_1)
    train_idx = subset_idx_0[:25] + subset_idx_1[:25]
    test_idx = subset_idx_0[25:525] + subset_idx_1[25:525]

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    # Create train and test subsets from indices
    train_data = Subset(cifar_data, train_idx)
    test_data = Subset(cifar_data, test_idx)

    # Extract inputs and targets from train and test subsets
    train_X = torch.stack([train_data[i][0] for i in range(len(train_data))])
    train_Y = torch.tensor(
        [label_map[train_data[i][1]] for i in range(len(train_data))]
    )
    test_X = torch.stack([test_data[i][0] for i in range(len(test_data))])
    test_Y = torch.tensor([label_map[test_data[i][1]] for i in range(len(test_data))])

    return train_X, train_Y, test_X, test_Y


train_X, train_Y, test_X, test_Y = get_random_train_test_data(cifar_data)
print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)


def generate_samples(
    n_support,
    n_query,
    datax,
    datay,
    transform_lambda,
    n_transformation=4,
    n_samples=1000,
):
    """
    given images and labels, it will generate samples of shape (n_way, n_support+n_query, image-dim)
    per image we will apply n_transformation and will use same label for all transformation
    image -> transform (will be randomaly applied) -> transformed_image
    """
    t_images = []
    labels = []
    for idx, (img, label) in enumerate(zip(datax, datay)):
        for i in range(n_transformation):
            t_img = transform_lambda(img)
            t_images.append(t_img.unsqueeze(0))
            labels.append(label.unsqueeze(0))

    t_images = torch.cat(t_images, dim=0)
    labels = torch.cat(labels, dim=0)

    n_shot = n_support + n_query

    class_0_idx = torch.nonzero(labels == 0)
    class_1_idx = torch.nonzero(labels == 1)

    img_class_0_subset = t_images[class_0_idx.squeeze()]
    img_class_1_subset = t_images[class_1_idx.squeeze()]

    class_0_rand_subset_idx = torch.randint(
        low=0, high=img_class_0_subset.size(0), size=(n_samples, n_shot)
    )
    class_1_rand_subset_idx = torch.randint(
        low=0, high=img_class_1_subset.size(0), size=(n_samples, n_shot)
    )

    img_class_0_rand_subset = img_class_0_subset[class_0_rand_subset_idx]
    img_class_1_rand_subset = img_class_1_subset[class_1_rand_subset_idx]

    data = torch.cat(
        [img_class_0_rand_subset.unsqueeze(1), img_class_1_rand_subset.unsqueeze(1)],
        dim=1,
    )
    target = (
        torch.arange(0, 2)
        .view(2, 1, 1)
        .expand(2, n_query, 1)
        .long()
        .unsqueeze(0)
        .repeat(*(n_samples, 1, 1, 1))
    )

    # data = data.reshape((data.size(0), data.size(1), data.size(2), -1))
    # target = target.reshape((target.size(0), target.size(1), data.size(2), -1))

    # print(f"data {data.shape}, target {target.shape}")

    return data, target


def get_samples_with_n_way_n_support_n_query(
    trainx, trainy, n_support, n_query, n_transformation, n_samples
):
    transform_augment = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    samples = generate_samples(
        n_support=n_support,
        n_query=n_query,
        datax=trainx,
        datay=trainy,
        transform_lambda=transform_augment,
        n_transformation=n_transformation,
        n_samples=n_samples,
    )
    return samples


# testing sampling function
# get_samples_with_n_way_n_support_n_query(train_X, train_Y, n_support=5, n_query=5, n_transformation=8, n_samples=1000)


def train(
    model,
    optimizer,
    train_x,
    train_y,
    n_support,
    n_query,
    max_epoch,
    n_transformation,
    epoch_size,
):
    # divide the learning rate by 2 at each epoch, as suggested in paper
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)

    # n_way will always be 2 (binary classificaiton)
    epoch = 0  # epochs done so far
    stop = False  # status to know when to stop
    epoch_acc = 0
    while epoch < max_epoch and not stop:
        running_loss = 0.0
        running_acc = 0.0

        samples_data, samples_labels = get_samples_with_n_way_n_support_n_query(
            train_x, train_y, n_support, n_query, n_transformation, n_samples=epoch_size
        )
        for s_data, s_labels in zip(samples_data, samples_labels):
            # print(f"sample shape {s_data.shape}")
            optimizer.zero_grad()
            loss, output = model.set_forward_loss(s_data, s_labels)
            running_loss += output["loss"]
            running_acc += output["acc"]
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / epoch_size
        epoch_acc = running_acc / epoch_size
        print(
            "Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}".format(
                epoch + 1, epoch_loss, epoch_acc
            )
        )
        epoch += 1
        scheduler.step()

    return epoch_acc


def predict(
    model,
    test_data,
    test_target,
    device=None,
    data_transform=transform_val,
    data_desc="prediction set",
):
    device = torch.device(device or "cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = []
    correct = 0
    with torch.no_grad():
        for data, target in zip(test_data, test_target):
            # print(data.shape, target.shape)
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)
            data = data.to(device)
            if len(data.shape) < 4:
                data = data.unsqueeze(0)
            output = model.predict_class_with_min_distance(data)
            pred = output.item()
            predictions.append(pred)
            correct += 1 if pred == target else 0

    print(f"Test accuracy : {correct / len(test_data)}")
    return predictions, correct / len(test_data)


# Load datasets
train_splits = []
train_targets = []
test_splits = []
test_targets = []
for i in range(5):
    train_X, train_Y, test_X, test_Y = get_random_train_test_data(cifar_data)
    train_splits.append(torch.tensor(train_X))
    train_targets.append(torch.tensor(train_Y))
    test_splits.append(torch.tensor(test_X))
    test_targets.append(torch.tensor(test_Y))


models = []
train_accuracies = []
test_accuracies = []
eval_predictions = []
for num_epoch in [3, 4, 5]:
    for epoch_size in [10, 50, 100, 500]:
        for num_of_aug in [2, 4, 8]:
            print(
                f"{'-='*50}\nTraining with epochs={num_epoch}, batch_size={epoch_size}, augmentations={num_of_aug}"
            )
            logger.info(
                f"{'-='*50}\nTraining with epochs={num_epoch}, batch_size={epoch_size}, augmentations={num_of_aug}"
            )
            for i, (train_X, train_Y, test_X, test_Y) in enumerate(
                zip(train_splits, train_targets, test_splits, test_targets)
            ):
                ## --- START of model definition ---
                ## EDIT the following lines with your own
                ## model/optimizer/[scheduler] implementation.
                ## (here we just used the baseline provided above)

                model = load_protonet_conv(
                    x_dim=(3, 28, 28),
                    hid_dim=64,
                    z_dim=64,
                )

                n_way = 2
                n_support = 5
                n_query = 5

                model = model.to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                ## --- END of model definition ---
                acc = train(
                    model,
                    optimizer,
                    train_X,
                    train_Y,
                    n_support,
                    n_query,
                    max_epoch=num_epoch,
                    n_transformation=num_of_aug,
                    epoch_size=epoch_size,
                )
                print(f"final train acc for model {i+1}: {acc:.2%}")
                train_accuracies.append(acc)
                models.append(model)
                # evaluate model
                pred_Y, test_acc = predict(
                    model, test_X, test_Y, data_desc=f"CODALab eval instance {i+1}"
                )
                eval_predictions.append(pred_Y)
                test_accuracies.append(test_acc)

            print(
                f"Mean Train Acc over {len(train_splits)} models: "
                f"{np.mean(train_accuracies):.2%} "
                f"+- {np.std(train_accuracies):.2}"
            )

            print(
                f"Mean Test Acc over {len(test_splits)} models: "
                f"{np.mean(test_accuracies):.2%} "
                f"+- {np.std(test_accuracies):.2}"
            )

            logger.info(
                f"Mean Train Acc over {len(train_splits)} models: "
                f"{np.mean(train_accuracies):.2%} "
                f"+- {np.std(train_accuracies):.2}"
            )

            logger.info(
                f"Mean Test Acc over {len(test_splits)} models: "
                f"{np.mean(test_accuracies):.2%} "
                f"+- {np.std(test_accuracies):.2}"
            )

