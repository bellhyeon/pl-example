import glob
import os
import shutil
from typing import List, Tuple
from torchvision import datasets, transforms


def save_cifar100_images(
    output_root="./dataset/cifar100",
) -> None:
    """
    Save CIFAR-100 dataset to the given output directory.

    Args:
        output_root (str): Path to the output directory.
    """

    os.makedirs(output_root, exist_ok=True)

    to_pil = transforms.ToPILImage()

    cifar100_meta = datasets.CIFAR100(root=output_root, download=True)
    class_names = cifar100_meta.classes

    for split in ["train", "test"]:
        is_train = split == "train"
        dataset = datasets.CIFAR100(
            root=output_root,
            train=is_train,
            download=True,
            transform=transforms.ToTensor(),
        )
        split_dir = os.path.join(output_root, split)
        os.makedirs(split_dir, exist_ok=True)

        print(f"Saving {split} images...")

        counters = {name: 0 for name in class_names}

        for idx in range(len(dataset)):
            img, label = dataset[idx]
            label_name = class_names[label]
            label_dir = os.path.join(split_dir, label_name)
            os.makedirs(label_dir, exist_ok=True)

            counters[label_name] += 1
            filename = f"{counters[label_name]:05d}.png"

            img_pil = to_pil(img)
            img_pil.save(os.path.join(label_dir, filename))

        print(f"Saved {sum(counters.values())} images to {split_dir}")

    # remove temp files
    os.remove(os.path.join(output_root, "cifar-100-python.tar.gz"))
    shutil.rmtree(os.path.join(output_root, "cifar-100-python"))


def load_cifar100(split_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load CIFAR-100 dataset from the given split directory.

    Args:
        split_dir (str): Path to the split directory containing the dataset.

    Returns:
        Tuple[List[str], List[int]]: List of image paths and corresponding labels
    """
    paths, labels = [], []

    class_names = sorted(os.listdir(split_dir))
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        imgs = sorted(glob.glob(os.path.join(class_dir, "*.png")))
        paths.extend(imgs)
        labels.extend([idx] * len(imgs))

    return paths, labels


if __name__ == "__main__":
    save_cifar100_images()
