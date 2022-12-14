"""
This script contains the routine of converting vanilla CIFAR10 from torchvision
to the format acceptable by this repo.
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torchvision


class _CIFAR10(torchvision.datasets.CIFAR10):
    """
    Slightly modified CIFAR10 datasets from torchvision. Modification consists
    of returning the index of the item and the image as a numpy arrays. It's
    the simplest way to convert torchvision CIFAR10 to the desired format.
    """

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, int]:
        """
        Obtain single item of the dataset.

        Parameters:
            index: The index of the item.

        Returns:
            Item of the dataset that is suitable for convertion the format.
        """
        image, label = super().__getitem__(index)
        return np.array(image), label, index


def prepare_data(dataset: _CIFAR10, prefix_path: str, subset: str) -> pd.DataFrame:
    """
    Prepare data to the desired format.

    Parameters:
        dataset: Intance of CIFAR10 dataset wrapper;
        prefix_path: Images will be saved at path with this prefix;
        subset: train or test.

    Returns:
        Dataframe with metadata.
    """
    images, labels, indices = list(zip(*dataset))
    paths = list(
        map(
            lambda index: os.path.join(prefix_path, f"{subset}_{index}.jpg"),
            indices[2],
        ),
    )
    df = pd.DataFrame(
        data={
            "path": paths,
            "label": labels[1],
            "subset": [subset] * len(paths),
        }
    )
    for image, path in zip(images, paths):
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download_path", required=True, type=str, help="Path to download CIFAR10 to"
    )
    parser.add_argument(
        "--target_path",
        required=True,
        type=str,
        help="Path to target folder to store your converted CIFAR10 dataset",
    )
    args = parser.parse_args()

    # Let's create needed paths.
    train_image_path = os.path.join(args.target_path, "images", "train")
    test_image_path = os.path.join(args.target_path, "images", "test")
    os.makedirs(train_image_path, exist_ok=True)
    os.makedirs(test_image_path, exist_ok=True)
    os.makedirs(args.download_path, exist_ok=True)

    # Convert torchvision CIFAR10 to the desired format.
    train_dataset = _CIFAR10(root=args.download_path, download=True)
    test_dataset = _CIFAR10(root=args.download_path, train=False, download=False)
    train_df = prepare_data(train_dataset, train_image_path, "train")
    test_df = prepare_data(test_dataset, test_image_path, "test")
    df = pd.concat((train_df, test_df))
    df.to_csv(os.path.join(args.target_path, "train.csv"), index=False)
