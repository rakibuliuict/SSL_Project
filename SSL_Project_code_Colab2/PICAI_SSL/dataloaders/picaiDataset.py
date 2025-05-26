import os
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
from torchvision.transforms import Compose
import logging

class PICAIDataset(Dataset):
    """ PICAI Dataset with multi-modal MRI and segmentation mask """

    def __init__(self, data_dir, list_dir, split, reverse=False, logging=logging):
        self.data_dir = data_dir
        self.list_dir = list_dir
        self.split = split
        self.reverse = reverse

        tr_transform = Compose([
            RandomCrop((256, 256, 16)),
            ToTensor()
        ])
        test_transform = Compose([
            CenterCrop((256, 256, 16)),
            ToTensor()
        ])

        if split == 'train_lab':
            data_path = os.path.join(self.list_dir, 'train_lab.txt')
            self.transform = tr_transform
        elif split == 'train_unlab':
            data_path = os.path.join(self.list_dir, 'train_unlab.txt')
            self.transform = tr_transform
            print("unlab transform")
        else:
            data_path = os.path.join(self.list_dir, 'test.txt')
            self.transform = test_transform

        # print(f"Reading file: {data_path}")  # Debug log

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path does not exist: {data_path}")

        with open(data_path, 'r') as f:
            self.image_list = f.read().splitlines()

        self.image_list = [os.path.join(self.data_dir, f"{pid}", f"{pid}.h5") for pid in self.image_list]

        logging.info("{} set: total {} samples".format(split, len(self.image_list)))
        logging.info("total {} samples".format(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx % len(self.image_list)]
        if self.reverse:
            image_path = self.image_list[len(self.image_list) - idx % len(self.image_list) - 1]

        with h5py.File(image_path, 'r') as h5f:
            t2w = h5f['image']['t2w'][:]
            adc = h5f['image']['adc'][:]
            hbv = h5f['image']['hbv'][:]
            seg = h5f['label']['seg'][:].astype(np.float32)

        image = np.stack([t2w, adc, hbv], axis=0)  # [3, H, W, D]
        samples = (image, seg)

        if self.transform:
            image_, label_ = self.transform(samples)
        else:
            image_, label_ = image, seg

        image_ = image_.permute(0, 3, 1, 2)  # [3, D, H, W]
        label_ = label_.permute(2, 0, 1).unsqueeze(0)  # [1, D, H, W]

        return image_.float(), label_.long()


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size  # (H, W, D)

    def __call__(self, samples):
        image, label = samples
        _, H, W, D = image.shape
        oH, oW, oD = self.output_size

        if D < oD:
            pad_d = oD - D
            image = np.pad(image, ((0, 0), (0, 0), (0, 0), (pad_d // 2, pad_d - pad_d // 2)), mode='constant')
            label = np.pad(label, ((0, 0), (0, 0), (pad_d // 2, pad_d - pad_d // 2)), mode='constant')
            D = oD

        start_h = max((H - oH) // 2, 0)
        start_w = max((W - oW) // 2, 0)
        start_d = max((D - oD) // 2, 0)

        image_cropped = image[:, start_h:start_h + oH, start_w:start_w + oW, start_d:start_d + oD]
        label_cropped = label[start_h:start_h + oH, start_w:start_w + oW, start_d:start_d + oD]

        return image_cropped, label_cropped


class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, samples):
        image, label = samples
        _, H, W, D = image.shape
        oH, oW, oD = self.output_size

        pad_h = max(oH - H, 0)
        pad_w = max(oW - W, 0)
        pad_d = max(oD - D, 0)

        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            image = np.pad(image, (
                (0, 0),
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
                (pad_d // 2, pad_d - pad_d // 2)), mode='constant')
            label = np.pad(label, (
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
                (pad_d // 2, pad_d - pad_d // 2)), mode='constant')

            _, H, W, D = image.shape

        start_h = np.random.randint(0, H - oH + 1)
        start_w = np.random.randint(0, W - oW + 1)
        start_d = np.random.randint(0, D - oD + 1)

        image_cropped = image[:, start_h:start_h + oH, start_w:start_w + oW, start_d:start_d + oD]
        label_cropped = label[start_h:start_h + oH, start_w:start_w + oW, start_d:start_d + oD]

        return image_cropped, label_cropped


class ToTensor(object):
    def __call__(self, sample):
        image = sample[0].astype(np.float32)
        label = sample[1].astype(np.float32)
        return torch.from_numpy(image), torch.from_numpy(label)


if __name__ == '__main__':
    data_dir = '/content/drive/MyDrive/SSL/Dataset/PICAI_dataset'
    list_dir = '/content/drive/MyDrive/SSL/SSL_Project_code_Colab2/PICAI_SSL/Datasets/picai/data_split'

    labset = PICAIDataset(data_dir, list_dir, split='train_lab')
    unlabset = PICAIDataset(data_dir, list_dir, split='train_unlab')
    testset = PICAIDataset(data_dir, list_dir, split='test')

    lab_sample = labset[0]
    unlab_sample = unlabset[0]
    test_sample = testset[0]

    print(len(labset), lab_sample[0].shape, lab_sample[1].shape)
    print(len(unlabset), unlab_sample[0].shape, unlab_sample[1].shape)
    print(len(testset), test_sample[0].shape, test_sample[1].shape)
