"""AutoML class for regression tasks.

This module contains an example AutoML class that simply returns predictions of a quickly trained MLP.
You do not need to use this setup, and you can modify this however you like.
"""
from __future__ import annotations

from typing import Any, Tuple
import torch
import random
import numpy as np
import logging

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from automl.dummy_model import DummyNN
from automl.utils import calculate_mean_std
from bayes_opt import BayesianOptimization

logger = logging.getLogger(__name__)


class AutoML:

    def __init__(
        self,
        seed: int,
    ) -> None:
        self.seed = seed
        self._model: nn.Module | None = None
        self._transform = None

    def fit(
        self,
        dataset_class: Any,
        lr: float,  # Example hyperparameter to optimize
    ) -> float:
        """A reference/toy implementation of a fitting function for the AutoML class.
        """
        # set seed for pytorch training
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        # Ensure deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(*calculate_mean_std(dataset_class)),
            ]
        )
        dataset = dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=self._transform
        )
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        input_size = dataset_class.width * dataset_class.height * dataset_class.channels

        model = DummyNN(input_size, dataset_class.num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)  # Use lr provided by Bayesian optimization
        
        model.train()
        for epoch in range(5):
            loss_per_batch = []
            for _, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                loss_per_batch.append(loss.item())
            logger.info(f"Epoch {epoch + 1}, Loss: {np.mean(loss_per_batch)}")
        model.eval()
        self._model = model

        # Return validation accuracy as the score to maximize
        return self.evaluate(dataset_class)

    def evaluate(self, dataset_class) -> float:
        """Evaluate the model on a validation set and return the accuracy."""
        dataset = dataset_class(
            root="./data",
            split='valid',  # Assuming you have a validation split
            download=True,
            transform=self._transform
        )
        data_loader = DataLoader(dataset, batch_size=100, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in data_loader:
                output = self._model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total
        return accuracy

    def optimize_hyperparameters(self, dataset_class: Any):
        """Perform Bayesian optimization to find optimal hyperparameters."""
        pbounds = {'lr': (0.001, 0.01)}  # Example: lr as a hyperparameter to optimize

        optimizer = BayesianOptimization(
            f=lambda lr: self.fit(lr, dataset_class),
            pbounds=pbounds,
            verbose=2,
            random_state=1,
        )

        optimizer.maximize(
            init_points=2,
            n_iter=3,
        )

        best_params = optimizer.max['params']
        best_lr = best_params['lr']

        logger.info(f"Best hyperparameters found: LR={best_lr}")

        # Now train the model with the best hyperparameters found
        self.fit(dataset_class, lr=best_lr)

    def predict(self, dataset_class) -> Tuple[np.ndarray, np.ndarray]:
        """A reference/toy implementation of a prediction function for the AutoML class."""
        dataset = dataset_class(
            root="./data",
            split='test',
            download=True,
            transform=self._transform
        )
        data_loader = DataLoader(dataset, batch_size=100, shuffle=False)
        predictions = []
        labels = []
        self._model.eval()
        with torch.no_grad():
            for data, target in data_loader:
                output = self._model(data)
                predicted = torch.argmax(output, 1)
                labels.append(target.numpy())
                predictions.append(predicted.numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        return predictions, labels
