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

from freeze_thaw import FreezeThaw
import itertools

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

    def black_box_function(self, lr: float, batch_size: int, dropout_rate: float, weight_decay: float, epochs: int) -> float:
        batch_size = int(round(batch_size))
        # TODO: implement freeze-thawing of models (in .fit)
        self.fit(self.dataset_class, epochs=epochs, lr=lr, batch_size=batch_size, dropout_rate=dropout_rate, weight_decay=weight_decay)
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
    
    def freeze_thaw(self,pbounds: dict, init_points: int = 2, n_iter: int = 10, init_epochs = 2, pred_epochs: int = 2):
        bounds=pbounds
        # Start by observing n_init_configs random configurations for init_epochs epoch each
        observed_configs_dicts={}
        observed_configs_list=np.empty((0,len(bounds.keys())))
        for _ in range(init_points):
            while True:
                new_config=np.empty(0)
                for key,_ in bounds.items():
                    new_config=np.append(new_config,np.round(np.random.uniform(bounds[key][0],bounds[key][1]),2))
                if not np.any(np.all(np.isin(observed_configs_list,new_config),axis=1)):
                    break
            # Observe the new configuration for init_epochs epochs
            f_space = np.linspace(1,init_epochs,init_epochs)
            experimental_data=self.black_box_function(lr=new_config[0],batch_size=new_config[1],dropout_rate=new_config[2],weight_decay=new_config[3],epochs=f_space[-1]) #Adapt black_box_function to take dict?
            observed_configs_dicts['_'.join([str(config) for config in new_config])]=(f_space,experimental_data)
            observed_configs_list=np.vstack([new_config,observed_configs_list])
        observed_configs_list=np.array(observed_configs_list)

        for _i in range(n_iter):
            ft=FreezeThaw(bounds,observed_configs_list,observed_configs_dicts)
            # Find the next configuration to evaluate
            new_config,new_epochs=ft.iterate(pred_epoch=pred_epochs)
            # Evaluate the new configuration
            if np.any(np.all(np.isin(observed_configs_list,new_config),axis=1)):
                results=self.black_box_function(lr=new_config[0],batch_size=new_config[1],dropout_rate=new_config[2],weight_decay=new_config[3],epochs=new_epochs[-1])
                old_config_entry=observed_configs_dicts['_'.join([str(c) for c in new_config])]
                observed_configs_dicts['_'.join([str(c) for c in new_config])]=(np.append(old_config_entry[0],new_epochs),np.append(old_config_entry[1],results))
            else:
                results=self.black_box_function(lr=new_config[0],batch_size=new_config[1],dropout_rate=new_config[2],weight_decay=new_config[3],epochs=new_epochs[-1])
                observed_configs_list=np.vstack([observed_configs_list,new_config])
                observed_configs_dicts['_'.join([str(c) for c in new_config])]=(new_epochs,results)
        
        # Find the best configuration in the GP
        num_steps=100
        param_values = [list(np.linspace(lower, upper, num_steps)) for lower, upper in bounds.values()]
        entire_config_space = list(itertools.product(*param_values))
        mean_prediction,_std_prediction=ft.predict_global(entire_config_space)
        best_config=entire_config_space[np.argmin(mean_prediction)]
        best_config_dict={}
        for key in bounds:
            best_config_dict[key]=best_config
        return {"params":best_config_dict}


    def optimize_hyperparameters(self, dataset_class: Any, pbounds: dict, init_points: int = 2, n_iter: int = 10, freeze_thaw=False) -> None:
        self.dataset_class = dataset_class
        if freeze_thaw:
            result=self.freeze_thaw(pbounds,init_points,n_iter)
            logger.info(f"Best hyperparameters: {result}")
        else:
            self.optimizer = BayesianOptimization(
                f=self.black_box_function,
                pbounds=pbounds,
                verbose=2,
                random_state=self.seed,
            )
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)
        logger.info(f"Best hyperparameters: {self.optimizer.max}")
