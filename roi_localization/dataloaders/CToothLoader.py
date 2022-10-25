import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import nibabel as nib
import itertools
from torch.utils.data.sampler import Sampler


def RandomCropNumpy(x_data, y_data, out_shape):
    # 随机裁剪 266x266x200 -> 128x128x128
    d, h, w = x_data.shape
    crop_depth = np.random.randint(0, d - out_shape[0])
    crop_height = np.random.randint(0, h - out_shape[1])
    crop_width = np.random.randint(0, w - out_shape[2])
    print(f'random (d, w, h): ({crop_depth, crop_height, crop_width})')

    crop_x_data = x_data[crop_depth:crop_depth + out_shape[0],
                        crop_height:crop_height + out_shape[1],
                        crop_width:crop_width + out_shape[2]]
    crop_y_data = y_data[crop_depth:crop_depth + out_shape[0],
                        crop_height:crop_height + out_shape[1],
                        crop_width:crop_width + out_shape[2]]
    return crop_x_data, crop_y_data


class CToothImageLoader(Dataset):
    """ MY Dataset """
    def __init__(self, base_dir=None, split='train', data_path=None, num=None, transform=None):
        """

        :param base_dir: 这个没用到，不用管
        :param split: 这个是划分 训练 loader 和 测试 loader
        :param data_path:a dict include train_path, test_path, nii_path
        :param num: 不用管，我也不知道是干啥的，没用上
        :param transform: 数据处理方式，在外面用上了，也是这个CToothLoader里面的方法，在下面可以找到
        """
        self._base_dir = base_dir
        self.transform = transform
        self.data_path = data_path
        self.sample_list = []
        if split == 'train':
            # 读取02split_train中生成的训练集列表，下同
            # with open(r'E:\Jupter\ctooth\TrainData\train.list', 'r') as f:
            with open(self.data_path['train'], 'r') as f:
                self.sample_list = f.readlines()
        elif split == 'test':
            # with open(r'E:\Jupter\ctooth\TrainData\test.list', 'r') as f:
            with open(self.data_path['train'], 'r') as f:
                self.sample_list = f.readlines()
        self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None:
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        image_name = self.sample_list[idx]  # 获取第几个
        # train_image_dir = r'E:\Jupter\ctooth\TrainData\nii_train_h5'  # 用来训练的 .h5的路径
        train_image_dir = self.data_path['nii']  # 用来训练的 .h5的路径
        train_image_name = os.path.join(train_image_dir, image_name.split('.')[0]+'_norm.h5')
        # print(train_image_name)
        h5f = h5py.File(train_image_name, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        label[label > 0.5] = 1
        # print(image.shape, label.shape)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class CToothImageLoader2(Dataset):
    """ MY Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        """

        :param base_dir: 这个没用到，不用管
        :param split: 这个是划分 训练 loader 和 测试 loader
        :param num: 不用管，我也不知道是干啥的，没用上
        :param transform: 数据处理方式，在外面用上了，也是这个CToothLoader里面的方法，在下面可以找到
        """
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        if split == 'train':
            # 读取02split_train中生成的训练集列表，下同
            with open(r'E:\Jupter\ctooth\train3.list', 'r') as f:
                self.sample_list = f.readlines()
        elif split == 'test':
            with open(r'E:\Jupter\ctooth\test3.list', 'r') as f:
                self.sample_list = f.readlines()
        self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None:
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        image_name = self.sample_list[idx]  # 获取第几个
        train_image_dir = r'E:\Jupter\ctooth\nii_Train_3dim_norm'  # 用来训练的 .h5的路径
        train_image_name = os.path.join(train_image_dir, image_name.split('.')[0]+'_norm.h5')
        h5f = h5py.File(train_image_name, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        label[label > 0.5] = 1

        # 随机裁剪成 output_shape
        output_shape = [128, 128, 128]
        image_crop, label_crop = RandomCropNumpy(image, label, output_shape)
        sample = {'image': image_crop, 'label': label_crop}
        # sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class LAHeart(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        if split == 'train':
            with open('/public_bme/data/czm/NC_CBCT/h5_roi/file.list', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open('/public_bme/data/czm/NC_CBCT/h5_roi/test.list', 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(image_name, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        label[label > 0.5] = 1
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # print(f'image.shape: {(w, h, d)}')
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]),
                        -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image),
                    'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image),
                    'label': torch.from_numpy(sample['label'].astype(np.float32)).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    train_set = CToothImageLoader()
    print(len(train_set))

    data = train_set[0]
    image, label = data['image'], data['label']
    # print(image.shape, label.shape)
    daloader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=1)
    print(len(daloader))