import os
import queue as Queue
import threading

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from triditional_method import EBOCV
from utils.rand_aug import RandAugment


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super().__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super().__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super().__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for idx in range(len(self.batch)):
                self.batch[idx] = self.batch[idx].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class _BasePalmVeinDataset(Dataset):
    def __init__(self, root_dir, local_rank, transform):
        super().__init__()
        self.root_dir = root_dir
        self.local_rank = local_rank
        self.transform = transform
        self.imgidx, self.labels = self.scan(root_dir)
        self.dfe = EBOCV.EBOCV()

    def scan(self, root):
        imgidx = []
        labels = []
        label = -1
        subdirs = sorted(os.listdir(root))
        for subdir in subdirs:
            image_names = os.listdir(os.path.join(root, subdir))
            label += 1
            for image_name in image_names:
                imgidx.append(os.path.join(subdir, image_name))
                labels.append(label)
        return imgidx, labels

    def read_image(self, path):
        return cv2.imread(os.path.join(self.root_dir, path))

    def __getitem__(self, index):
        path = self.imgidx[index]
        img = self.read_image(path)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        sample = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tri_feature = self.dfe.get_feature(img)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label, tri_feature

    def __len__(self):
        return len(self.imgidx)


class FaceDatasetFolder(_BasePalmVeinDataset):
    def __init__(self, root_dir, local_rank, tm="EBOCV"):
        if tm != "EBOCV":
            raise ValueError(f"Only EBOCV is supported in this project cleanup, got: {tm}")

        image_size = 128
        resize_ratio = 0.1
        aug = 2
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize([128, 128]),
                RandAugment(aug, aug),
                transforms.Resize((int(image_size * (1 + resize_ratio)), int(image_size * (1 + resize_ratio)))),
                transforms.RandomCrop((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )
        super().__init__(root_dir=root_dir, local_rank=local_rank, transform=transform)


class valDatasetFolder(_BasePalmVeinDataset):
    def __init__(self, root_dir, local_rank, tm="EBOCV"):
        if tm != "EBOCV":
            raise ValueError(f"Only EBOCV is supported in this project cleanup, got: {tm}")

        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize([128, 128]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        super().__init__(root_dir=root_dir, local_rank=local_rank, transform=transform)
