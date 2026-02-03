import os
import random
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ..basic_dataset import FedDataset

# 支持的图像后缀集合
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_image_file(filename: str) -> bool:
    """判断文件是否为图像文件。"""
    return os.path.splitext(filename.lower())[1] in IMG_EXTENSIONS


def _collect_class_dirs(root_dir: str, dataset_name: str) -> List[str]:
    """递归收集符合“数字_数据集名”格式的类目录。"""
    pattern = re.compile(r"^\d+_" + re.escape(dataset_name) + r"$")
    matched_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if pattern.match(os.path.basename(dirpath)):
            matched_dirs.append(dirpath)
    matched_dirs.sort()
    return matched_dirs


def _collect_images(class_dir: str) -> List[str]:
    """收集类目录下的所有图像文件路径。"""
    files = []
    for name in os.listdir(class_dir):
        if _is_image_file(name):
            files.append(os.path.join(class_dir, name))
    files.sort()
    return files


def _build_pairs(
    class_dirs: List[str],
    max_pairs_per_class: Optional[int],
    negative_ratio: float,
    rng: random.Random,
) -> Tuple[List[Tuple[str, str]], List[int]]:
    """
    构建图片对与标签：
    - 正样本：同一类目录内的两张图片，label=1
    - 负样本：不同类目录的两张图片，label=0
    """
    class_to_files: Dict[str, List[str]] = {}
    for class_dir in class_dirs:
        files = _collect_images(class_dir)
        if files:
            class_to_files[class_dir] = files

    # 正样本：同类目录内的两两组合
    pos_pairs: List[Tuple[str, str]] = []
    for class_dir, files in class_to_files.items():
        if len(files) < 2:
            continue
        all_pairs = list(combinations(files, 2))
        if max_pairs_per_class is not None and len(all_pairs) > max_pairs_per_class:
            all_pairs = rng.sample(all_pairs, max_pairs_per_class)
        pos_pairs.extend(all_pairs)

    pos_labels = [1] * len(pos_pairs)

    # 负样本：不同类目录随机抽样
    neg_pairs: List[Tuple[str, str]] = []
    class_dirs_list = list(class_to_files.keys())
    if len(class_dirs_list) >= 2 and len(pos_pairs) > 0:
        target_neg = int(len(pos_pairs) * negative_ratio)
        while len(neg_pairs) < target_neg:
            class_a, class_b = rng.sample(class_dirs_list, 2)
            file_a = rng.choice(class_to_files[class_a])
            file_b = rng.choice(class_to_files[class_b])
            neg_pairs.append((file_a, file_b))

    neg_labels = [0] * len(neg_pairs)

    pairs = pos_pairs + neg_pairs
    labels = pos_labels + neg_labels
    combined = list(zip(pairs, labels))
    rng.shuffle(combined)
    if combined:
        pairs, labels = zip(*combined)
        return list(pairs), list(labels)
    return [], []


class PairConcatDataset(Dataset):
    """将两张图片拼接为一个样本，输出二分类标签。"""
    def __init__(self, pairs: List[Tuple[str, str]], labels: List[int], transform=None, concat_dim="channel"):
        self.pairs = pairs
        self.target = np.array(labels)
        self.data = pairs
        self.transform = transform
        self.concat_dim = concat_dim

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        """读取两张图片，做 transform，然后按指定维度拼接。"""
        path1, path2 = self.pairs[index]
        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # 通道拼接：输出形状为 (6, H, W)
        if self.concat_dim == "channel":
            img = torch.cat([img1, img2], dim=0)
        # 高度拼接：输出形状为 (3, 2H, W)
        elif self.concat_dim == "height":
            img = torch.cat([img1, img2], dim=1)
        # 宽度拼接：输出形状为 (3, H, 2W)
        elif self.concat_dim == "width":
            img = torch.cat([img1, img2], dim=2)
        else:
            raise ValueError("concat_dim must be one of ['channel', 'height', 'width']")

        label = int(self.target[index])
        return img, label


class LocalFingerprintPairs(FedDataset):
    """
    本地数据集读取与划分：
    - 从 root_dir 中筛选形如 “数字_数据集名” 的类目录
    - 先按客户端数分配类目录
    - 再在客户端内划分 train/val/test
    - 最后为每个子集构建正负样本对，并保存为 .pkl
    """
    def __init__(
        self,
        root_dir: str,
        save_dir: str,
        num_clients: int,
        dataset_name: str,
        seed: int = 42,
        img_size: int = 128,
        max_pairs_per_class: int = 50,
        negative_ratio: float = 1.0,
        concat_dim: str = "channel",
    ):
        # 原始数据根目录（相对路径时，以项目根目录为基准）
        project_root = Path(__file__).resolve().parents[3]
        root_path = Path(os.path.expanduser(root_dir))
        if not root_path.is_absolute():
            root_path = project_root / root_path
        self.root = str(root_path)

        # 保存目录（相对路径时，以项目根目录为基准）
        save_path = Path(os.path.expanduser(save_dir))
        if not save_path.is_absolute():
            save_path = project_root / save_path
        self.path = str(save_path)
        self.num = num_clients
        self.dataset_name = dataset_name
        self.seed = seed
        # 统一缩放尺寸，保持模型输入一致
        self.img_size = img_size
        self.max_pairs_per_class = None if max_pairs_per_class <= 0 else max_pairs_per_class
        self.negative_ratio = negative_ratio
        self.concat_dim = concat_dim

        os.makedirs(self.path, exist_ok=True)
        os.makedirs(os.path.join(self.path, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.path, "val"), exist_ok=True)
        os.makedirs(os.path.join(self.path, "test"), exist_ok=True)

        # 基础预处理：Resize + ToTensor
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ]
        )

    def preprocess(self):
        """执行本地数据划分与配对构建。"""
        print("########### Preprocessing local fingerprint pairs ###########")
        # 1) 收集类目录，并打乱
        class_dirs = _collect_class_dirs(self.root, self.dataset_name)
        rng = random.Random(self.seed)
        rng.shuffle(class_dirs)

        total_dirs = len(class_dirs)
        if total_dirs == 0:
            raise RuntimeError(f"No class directories found under: {self.root}")

        # 2) 平均分配类目录给各客户端（前 extra_dirs 个客户端多一个）
        base_dirs = total_dirs // self.num
        extra_dirs = total_dirs % self.num

        start = 0
        for client_idx in range(self.num):
            count = base_dirs + (1 if client_idx < extra_dirs else 0)
            end = start + count
            client_dirs = class_dirs[start:end]
            start = end

            # 3) 按客户端划分 train/val/test
            if not client_dirs:
                train_pairs, train_labels = [], []
                val_pairs, val_labels = [], []
                test_pairs, test_labels = [], []
            else:
                num_train_val = max(1, int(len(client_dirs) * 0.8))
                train_val_dirs = client_dirs[:num_train_val]
                test_dirs = client_dirs[num_train_val:]

                num_train = max(1, int(len(train_val_dirs) * 0.8))
                train_dirs = train_val_dirs[:num_train]
                val_dirs = train_val_dirs[num_train:]

                # 4) 在各子集内构建正负样本对
                train_pairs, train_labels = _build_pairs(
                    train_dirs,
                    self.max_pairs_per_class,
                    self.negative_ratio,
                    rng,
                )
                val_pairs, val_labels = _build_pairs(
                    val_dirs,
                    self.max_pairs_per_class,
                    self.negative_ratio,
                    rng,
                )
                test_pairs, test_labels = _build_pairs(
                    test_dirs,
                    self.max_pairs_per_class,
                    self.negative_ratio,
                    rng,
                )

            # 5) 保存为二次处理后的可直接读取数据集
            train_dataset = PairConcatDataset(
                train_pairs, train_labels, transform=self.transform, concat_dim=self.concat_dim
            )
            val_dataset = PairConcatDataset(
                val_pairs, val_labels, transform=self.transform, concat_dim=self.concat_dim
            )
            test_dataset = PairConcatDataset(
                test_pairs, test_labels, transform=self.transform, concat_dim=self.concat_dim
            )

            torch.save(train_dataset, os.path.join(self.path, "train", f"data{client_idx}.pkl"))
            torch.save(val_dataset, os.path.join(self.path, "val", f"data{client_idx}.pkl"))
            torch.save(test_dataset, os.path.join(self.path, "test", f"data{client_idx}.pkl"))

    def get_dataset(self, id, type="train"):
        dataset = torch.load(os.path.join(self.path, type, f"data{id}.pkl"))
        return dataset

    def get_data_loader(self, id, batch_size=None, type="train"):
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader
