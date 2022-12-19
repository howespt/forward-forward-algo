from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader


def get_data_loaders(train_batch_size, test_batch_size):
    mnist_transform = Compose(
        [
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
            Lambda(lambda x: torch.flatten(x)),
        ]
    )

    train_dataset = MNIST(
        root="./data", train=True, download=True, transform=mnist_transform
    )
    test_dataset = MNIST(
        root="./data", train=False, download=True, transform=mnist_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


class DenseFFLayer(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        goodness_threshold: float = 5,
        activation_fn: Callable = nn.ReLU,
    ):
        super().__init__(in_features, out_features)
        self.goodness_threshold = goodness_threshold

        # What is the best activation function to use? So far, only ReLUs have
        # been explored. There are many other possibilities whose behaviour is
        # unexplored in the context of FF. Making the activation be the negative
        # log of the density under a t-distribution is an interesting
        # possibility(Osindero et al., 2006).
        self.activation = activation_fn()
        self.optimizer = Adam(self.parameters(), lr=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise the input: This removes all of the information that was used
        # to determine the goodness in the previous hidden layer and forces
        # this hidden layer to use information in the relative activities of the
        # neurons in the previous hidden layer. These relative activities are
        # unaffected by the layer-normalization. To put it another way, the
        # activity vector in the previous hidden layer has a length and an
        # orientation. The length is used to define the goodness for that layer
        # and only the orientation is passed to the this layer.
        x = x / (
            x.norm(p=2, dim=1, keepdim=True) + 1e-8
        )  # L2-normalise with a small epsilon to avoid division by zero
        x = F.linear(x, self.weight, self.bias)
        return self.activation(x)

    def update(self, x: torch.Tensor, x_is_good: bool) -> None:
        outputs = self.forward(x)
        squared_activations = outputs.pow(2).mean(1)
        goodness = squared_activations - self.goodness_threshold
        loss_sign = -1 if x_is_good else 1
        # loss = loss_sign * goodness.mean()
        loss = torch.log(1 + torch.exp(loss_sign * goodness)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Detach the outputs from the graph so that they don't accumulate gradients
        return outputs.detach()


class ForwardForwardNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        goodness_threshold: float = 5,
        num_classes: int = 10,
    ):
        super().__init__()
        self.goodness_threshold = goodness_threshold
        self.num_classes = num_classes
        self.layers = nn.ModuleList()
        self.layers.append(
            DenseFFLayer(input_size, hidden_sizes[0], goodness_threshold)
        )
        for i in range(1, len(hidden_sizes)):
            self.layers.append(
                DenseFFLayer(hidden_sizes[i - 1], hidden_sizes[i], goodness_threshold)
            )

    def train_step(self, x: torch.Tensor, x_is_good: bool) -> None:
        for i, layer in enumerate(self.layers):
            x = layer.update(x, x_is_good)

    def train(
        self, train_dataloader: DataLoader, test_dataloader: DataLoader, epochs: int
    ):
        for epoch in tqdm(range(epochs)):
            for x, y in train_dataloader:
                # Insert y into x
                x_good = insert_target_into_input(x, y)
                # Train the network on good examples
                self.train_step(x_good, True)
                # Train the network on bad examples
                x_bad = insert_target_into_input(x, y, make_bad=True)
                self.train_step(x_bad, False)
            self.test(test_dataloader)

    def test(self, test_dataloader: DataLoader):
        predictions = torch.zeros(0)
        trues = torch.zeros(0)
        for x, y in test_dataloader:
            y_hat = self.evaluate_labels(x)
            predictions = torch.cat((predictions, y_hat))
            trues = torch.cat((trues, y))
        accuracy = (predictions == trues).float().mean()
        print(f"Accuracy: {accuracy}")
        return accuracy, predictions, trues

    def evaluate_labels(self, x: torch.Tensor) -> torch.Tensor:
        goodness_by_label = torch.zeros(x.shape[0], self.num_classes)
        for label in range(self.num_classes):
            x_copy = insert_target_into_input(x, torch.tensor([label] * x.shape[0]))
            sum_of_goodness = torch.zeros(x.shape[0])
            for i, layer in enumerate(self.layers):
                x_copy = layer.forward(x_copy)
                #  It is better to run the net with a particular label as part
                #  of the input and accumulate the goodnesses of all but the
                #  first hidden layer.
                if i != 0:
                    sum_of_goodness += x_copy.pow(2).mean(1)
            goodness_by_label[:, label] = sum_of_goodness
        return goodness_by_label.argmax(1)


def insert_target_into_input(x, y, num_classes=10, make_bad=False):
    # Insert y into x
    # make a copy of x
    x_copy = x.clone()
    # Set the first 10 elements of each row to the corresponding y value
    x_copy[:, :num_classes] = torch.zeros_like(x[:, :num_classes])
    y_copy = y.clone()
    if make_bad:
        # If we want to make a bad example, we need to change the target to anything but the correct class
        y_copy = torch.randint(num_classes, (y.shape[0],))
        y_copy[y_copy == y] += torch.randint(
            1, num_classes, (y_copy[y_copy == y].shape[0],)
        )
        y_copy = y_copy % num_classes

    x_copy[torch.arange(len(x)), y_copy] = x.max()
    return x_copy
