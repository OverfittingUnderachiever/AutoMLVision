import torch
import random
import numpy as np
import logging

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Any, Tuple

from automl.dummy_model import DummyNN
from automl.utils import calculate_mean_std
from bayes_opt import BayesianOptimization

logger = logging.getLogger(__name__)

class AutoML:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._model: nn.Module | None = None
        self._transform = None
        self._initialize_random_seeds()
        self.optimizer = None #initialize later

    def _initialize_random_seeds(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Check if CUDA is available and set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def _create_model(self, input_size: int, num_classes: int) -> nn.Module:
        return DummyNN(input_size, num_classes)

    def _train_model(self, model: nn.Module, train_loader: DataLoader, lr: float, epochs: int) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        model.train()
        for epoch in range(epochs):
            loss_per_batch = []
            for _, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                loss_per_batch.append(loss.item())
            logger.info(f"Epoch {epoch + 1}, Loss: {np.mean(loss_per_batch)}")

    def fit(self, dataset_class: Any, epochs: int = 5, lr: float = 0.003, batch_size: int = 64) -> None:
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(*calculate_mean_std(dataset_class)),
            ]
        )
        dataset = dataset_class(root="./data", split='train', download=True, transform=self._transform)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        input_size = dataset_class.width * dataset_class.height * dataset_class.channels
        self._model = self._create_model(input_size, dataset_class.num_classes)
        self._train_model(self._model, train_loader, lr, epochs)
        return None

    def evaluate_on_validation(self) -> float:
        dataset = self.dataset_class(root="./data", split='val', download=True, transform=self._transform)
        data_loader = DataLoader(dataset, batch_size=100, shuffle=False)
        predictions, labels = self._evaluate_model(self._model, data_loader)
        validation_loss = self._calculate_loss(predictions, labels)  # Example of calculating validation loss
        return -validation_loss  # Return negative for minimization
    
    def black_box_function(self, lr: float, batch_size: int) -> float:
        batch_size = int(round(batch_size))
        self.fit(self.dataset_class, epochs=5, lr=lr, batch_size=batch_size)
        validation_loss = self.evaluate_on_validation()
        return validation_loss

    def predict(self, dataset_class):
        test_dataset = dataset_class(root="./data", split='test', download=True, transform=self._transform)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

        self._model.eval()
        predictions = []
        labels = []

        with torch.no_grad():
            for data, target in test_loader:
                output = self._model(data)
                predicted = torch.argmax(output, dim=1)
                predictions.append(predicted.numpy())
                labels.append(target.numpy())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        return predictions, labels

    def _evaluate_model(self, model: torch.nn.Module, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        logits, labels = [], []
        with torch.no_grad():
            for data, target in data_loader:
                output = model(data)  # Get raw logits
                logits.append(output.numpy())  # Collect logits
                labels.append(target.numpy())  # Collect labels
        return np.concatenate(logits), np.concatenate(labels)
    
    def _calculate_loss(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        # Convert labels to torch LongTensor
        labels = torch.from_numpy(labels).long()

        # Convert predictions to torch tensor and ensure float32
        predictions = torch.tensor(predictions, dtype=torch.float32)

        # Compute cross-entropy loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(predictions, labels)

        return loss.item()


    def optimize_hyperparameters(self, dataset_class: Any, pbounds: dict, init_points: int = 2, n_iter: int = 10) -> None:
        self.dataset_class = dataset_class
        self.optimizer = BayesianOptimization(
            f=self.black_box_function,
            pbounds=pbounds,
            verbose=2,
            random_state=self.seed,
        )
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)
        logger.info(f"Best hyperparameters: {self.optimizer.max}")

        