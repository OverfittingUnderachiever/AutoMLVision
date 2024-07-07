import torch
import random
import numpy as np
import logging

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Any, Tuple

from automl.dummy_model import DummyNN
from automl.efficient_net_model import EfficientNetModel
from automl.utils import calculate_mean_std

import ray
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.search.bayesopt import BayesOptSearch

from bayes_opt import BayesianOptimization

logger = logging.getLogger(__name__)

class AutoML:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._model: nn.Module | None = None
        self._transform = None
        self._initialize_random_seeds()
        self.optimizer = None  # initialize later

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

    def _create_model(self, input_size: int, num_classes: int, dropout_rate: float) -> nn.Module:
        model = EfficientNetModel(self.dataset_class, output_size=num_classes, dropout_rate=dropout_rate)
        return model.to(self.device)  # Move model to the device

    def _train_model(self, model: nn.Module, train_loader: DataLoader, lr: float, weight_decay: float, epochs: int) -> None:
        criterion = nn.CrossEntropyLoss().to(self.device)  # Move loss function to the device
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # Include weight_decay
        model.train()
        for epoch in range(epochs):
            loss_per_batch = []
            for _, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)  # Move data and target to the device
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                loss_per_batch.append(loss.item())
            logger.info(f"Epoch {epoch + 1}, Loss: {np.mean(loss_per_batch)}")

    def fit(self, dataset_class: Any, epochs: int = 7, lr: float = 0.003, batch_size: int = 64, dropout_rate: float = 0.5, weight_decay: float = 0.0) -> None:
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(*calculate_mean_std(dataset_class)),
            ]
        )
        dataset = dataset_class(root="./data", split='train', download=True, transform=self._transform)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        input_size = dataset_class.width * dataset_class.height * dataset_class.channels
        self._model = self._create_model(input_size, dataset_class.num_classes, dropout_rate)  # Include dropout_rate
        self._train_model(self._model, train_loader, lr, weight_decay, epochs)  # Include weight_decay
        return None

    def evaluate_on_validation(self) -> float:
        dataset = self.dataset_class(root="./data", split='val', download=True, transform=self._transform)
        data_loader = DataLoader(dataset, batch_size=100, shuffle=False)
        predictions, labels = self._evaluate_model(self._model, data_loader)
        validation_loss = self._calculate_loss(predictions, labels)  # Example of calculating validation loss
        return -validation_loss  # Return negative for minimization

    def black_box_function (self, config: dict) -> None:
        lr = config['lr']
        batch_size = int(config['batch_size'])
        dropout_rate = config['dropout_rate']
        weight_decay = config['weight_decay']
        self.fit(self.dataset_class, epochs=7, lr=lr, batch_size=batch_size, dropout_rate=dropout_rate, weight_decay=weight_decay)
        validation_loss = self.evaluate_on_validation()
        tune.report(loss=validation_loss)
        #return validation_loss


    def predict(self, dataset_class):
        test_dataset = dataset_class(root="./data", split='test', download=True, transform=self._transform)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

        self._model.eval()
        predictions = []
        labels = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)  # Move data and target to the device
                output = self._model(data)
                predicted = torch.argmax(output, dim=1)
                predictions.append(predicted.cpu().numpy())  # Move predictions to CPU for numpy conversion
                labels.append(target.cpu().numpy())  # Move labels to CPU for numpy conversion

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        return predictions, labels

    def _evaluate_model(self, model: torch.nn.Module, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        logits, labels = [], []
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)  # Move data and target to the device
                output = model(data)  # Get raw logits
                logits.append(output.cpu().numpy())  # Move logits to CPU for numpy conversion
                labels.append(target.cpu().numpy())  # Move labels to CPU for numpy conversion
        return np.concatenate(logits), np.concatenate(labels)

    def _calculate_loss(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        # Convert labels to torch LongTensor
        labels = torch.from_numpy(labels).long().to(self.device)  # Move labels to the device

        # Convert predictions to torch tensor and ensure float32
        predictions = torch.tensor(predictions, dtype=torch.float32).to(self.device)  # Move predictions to the device

        # Compute cross-entropy loss
        criterion = nn.CrossEntropyLoss().to(self.device)  # Move loss function to the device
        loss = criterion(predictions, labels)

        return loss.item()

    def optimize_hyperparameters(self, dataset_class: Any, pbounds: dict, init_points: int = 2, n_iter: int = 10) -> None:
        self.dataset_class = dataset_class
        scheduler = HyperBandScheduler(metric="loss", mode="min")
        
        bayesopt_search = BayesOptSearch(
            pbounds,
            random_state=self.seed,
            metric="loss",
            mode="min"
        )
        
        analysis = tune.run(
            tune.with_parameters(self.black_box_function),
            name="hyperband_optimization",
            search_alg=bayesopt_search,
            scheduler=scheduler,
            num_samples=n_iter,
            config=pbounds,
            resources_per_trial={"cpu": 0, "gpu": 1}, 
            verbose=1,
        )
        
        best_config = analysis.get_best_config(metric="loss", mode="min")
        print(f"Best hyperparameters: {best_config}")
        self.best_params = best_config

