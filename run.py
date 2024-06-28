"""An example run file which trains a dummy AutoML system on the training split of a dataset
and logs the accuracy score on the test set.

In the example data you are given access to the labels of the test split, however
in the test dataset we will provide later, you will not have access
to this and you will need to output your predictions for the images of the test set
to a file, which we will grade using github classrooms!
"""
from __future__ import annotations

from pathlib import Path
from sklearn.metrics import accuracy_score
import numpy as np

# Determine the path to the src directory
import sys
import os
src_path = os.path.join(os.path.dirname(__file__), 'src')

# Add the src directory to the system path if it's not already there
if src_path not in sys.path:
    sys.path.append(src_path)

from automl.automl import AutoML

import argparse

import logging

from automl.datasets import FashionDataset, FlowersDataset, EmotionsDataset, FashionMiniDataset, FlowersMiniDataset, EmotionsMiniDataset

logger = logging.getLogger(__name__)


def main(
    dataset: str,
    output_path: Path,
    seed: int,
):
    match dataset:
        case "fashion":
            dataset_class = FashionDataset
        case "flowers":
            dataset_class = FlowersDataset
        case "emotions":
            dataset_class = EmotionsDataset
        case "fashion_mini":
            dataset_class = FashionMiniDataset
        case "flowers_mini":
            dataset_class = FlowersMiniDataset
        case "emotions_mini":
            dataset_class = EmotionsMiniDataset
        case _:
            raise ValueError(f"Invalid dataset: {args.dataset}")

    logger.info("Fitting AutoML")

    # You do not need to follow this setup or API it's merely here to provide
    # an example of how your automl system could be used.
    # As a general rule of thumb, you should **never** pass in any
    # test data to your AutoML solution other than to generate predictions.
    automl = AutoML(seed=seed)
    # Define the hyperparameter search space
    pbounds = {
        'lr': (1e-5, 1e-2),
        'batch_size': (256, 256),
        'dropout_rate': (0.1, 0.5),
        'weight_decay': (0, 0.1)
    }
    # Perform hyperparameter optimization
    automl.optimize_hyperparameters(dataset_class, pbounds=pbounds, init_points=2, n_iter=20)
    logger.info(f"Best hyperparameters: {automl.optimizer.max}")

    # Get the best hyperparameters found
    best_params = automl.optimizer.max['params']
    best_lr = best_params['lr']
    best_batch_size = int(round(best_params['batch_size']))  # no bo support for ints :(
    best_dropout_rate = best_params['dropout_rate']
    best_weight_decay = best_params['weight_decay']

    # Fit the best model on the entire dataset
    automl.fit(dataset_class, epochs=7, lr=best_lr, batch_size=best_batch_size, dropout_rate=best_dropout_rate, weight_decay=best_weight_decay)
    test_preds, test_labels = automl.predict(dataset_class)
    # Write the predictions of X_test to disk
    # This will be used by github classrooms to get a performance
    # on the test set.
    logger.info("Writing predictions to disk")
    with output_path.open("wb") as f:
        np.save(f, test_preds)

    # check if test_labels has missing data


    if not np.isnan(test_labels).any():
            # Ensure both predictions and labels are in the correct format (1-dimensional arrays)
        test_preds = test_preds.astype(np.int64)  # Ensure predictions are integers
        test_labels = test_labels.astype(np.int64)  # Ensure labels are integers
        acc = accuracy_score(test_labels, test_preds)
        logger.info(f"Accuracy on test set: {acc}")
    else:
        # This is the setting for the exam dataset, you will not have access to the labels
        logger.info(f"No test split for dataset '{dataset}'")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of the dataset to run on.",
        choices=["fashion", "flowers", "emotions", "fashion_mini", "flowers_mini", "emotions_mini"]
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("predictions.npy"),
        help=(
            "The path to save the predictions to."
            " By default this will just save to './predictions.npy'."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for reproducibility if you are using and randomness,"
            " i.e. torch, numpy, pandas, sklearn, etc."
        )
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Whether to log only warnings and errors."
    )

    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    logger.info(
        f"Running dataset {args.dataset}"
        f"\n{args}"
    )

    main(
        dataset=args.dataset,
        output_path=args.output_path,
        seed=args.seed,
    )