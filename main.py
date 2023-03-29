import logging
from datetime import datetime

import torch
from params import *
from rich.logging import RichHandler
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Lambda, Normalize, ToTensor

torch.manual_seed(SEED)

LOGGER = logging.getLogger("FF")
LOGGER.addHandler(RichHandler())
LOGGER.addHandler(
    logging.FileHandler(f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
)
LOGGER.setLevel(LOG_LEVEL)


def getMNISTDataLoader(batch_size: int = 128, train: bool = True, shuffle: bool = True):
    return DataLoader(
        MNIST(
            "./data",
            train=train,
            download=True,
            transform=Compose(
                [
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,)),
                    Lambda(lambda x: x.view(-1)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def appendLabel(x: torch.Tensor, y: torch.Tensor):
    """Concatenate label to input tensor as one-hot vector."""
    return torch.cat(
        [x, x.max() * F.one_hot(y.to(torch.long), num_classes=10)],
        dim=1,
    )


class Forward(nn.Linear):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(in_dim, out_dim, bias, device, dtype)
        self.optimizer = Adam(self.parameters(), lr=LEARNING_RATE)

    def forward(self, x: torch.Tensor):
        x = x / (x.norm(p=2, dim=1, keepdim=True) + EPSILON)
        return F.relu_(super().forward(x))

    def trainOn(self, x: torch.Tensor, positive: bool = True):
        LOGGER.debug(
            f"Training layer with {'positive' if positive else 'negative'} samples"
        )
        # G = sum_i (y_i ** 2) / N
        g = self.forward(x).pow(2).mean(dim=1)
        # Smooth loss function
        loss = torch.log(
            1 + torch.exp((THRESHOLD - g) if positive else (g - THRESHOLD))
        )
        # Make local param update (NO BACKPROP ACROSS LAYERS)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        # Return the new prediction for the next layer to use
        return self.forward(x).detach()


class FFNet(nn.Module):
    def __init__(
        self,
        dims: list[int],
        bias: bool = True,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.layers = [
            Forward(dims[i], dims[i + 1], bias, device, dtype).to(DEVICE)
            for i in range(len(dims) - 1)
        ]

    def predict(self, x: torch.Tensor):
        LOGGER.debug("Predicting labels")
        label_evaluations = []
        for label in range(10):
            # Append label to input as one-hot vector
            out = appendLabel(x, torch.tensor([label] * x.shape[0]).to(DEVICE))
            # Store the goodness of each layer
            goodness = torch.zeros(x.shape[0]).to(DEVICE)
            # Forward pass through layers
            for layer in self.layers:
                out = layer(out)
                goodness += out.pow(2).mean(dim=1)
            # Store the evaluation of the label
            label_evaluations.append(goodness.unsqueeze(-1))
        return torch.cat(label_evaluations, dim=-1).argmax(-1)

    def trainOn(self, x: torch.Tensor, y: torch.Tensor, positive: bool = True):
        LOGGER.debug(
            f"Training model with {'positive' if positive else 'negative'} samples"
        )
        # Append label to input as one-hot vector
        x = appendLabel(x, y)
        # Forward pass through layers
        for layer in self.layers:
            x = layer.trainOn(x, positive)
        return x


if __name__ == "__main__":
    # Get data
    train_loader = getMNISTDataLoader(TRAIN_BATCH_SIZE)
    test_loader = getMNISTDataLoader(TEST_BATCH_SIZE, train=False)

    # Initialize model
    model = FFNet([784 + 10, *DIMS]).to(DEVICE)

    # Train model
    for e, epoch in enumerate(range(EPOCH)):
        for i, (x, y) in enumerate(train_loader):
            x: torch.Tensor
            y: torch.Tensor
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            # Making fast alternation between postive and negative
            model.trainOn(x, y)
            model.trainOn(x, y[torch.randperm(x.shape[0])], False)
            if i == 0 or (i + 1) % LOG_INTERVAL == 0 or i + 1 == len(train_loader):
                error_rate = (model.predict(x) != y).float().mean()
                LOGGER.info(
                    f"[{e+1}/{EPOCH}] [{i+1}/{len(train_loader)}] Train error rate: {error_rate.item():.5f}"
                )

    # Test model
    for i, (x, y) in enumerate(test_loader):
        x: torch.Tensor
        y: torch.Tensor
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        error_rate = (model.predict(x) != y).float().mean()
        LOGGER.info(
            f"[{i+1}/{len(test_loader)}] Test error rate: {error_rate.item():.5f}"
        )
