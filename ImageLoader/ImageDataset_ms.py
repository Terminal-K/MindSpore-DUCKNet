import numpy as np
import os
import glob
import random
import cv2
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from ImageLoader.ImageLoader2D import load_data
from sklearn.model_selection import train_test_split

def get_whole_dataset(folder_path, img_size, seed_value, split=["train", "val", "test"]):
    def get_split_folder(x, y, split_folder):
        image_folder = os.path.join(split_folder, "images")
        mask_folder = os.path.join(split_folder, "masks")
        if not os.path.exists(image_folder): os.makedirs(image_folder)
        if not os.path.exists(mask_folder): os.makedirs(mask_folder)
        for i, (per_x, per_y) in enumerate(zip(x, y)):
            cv2.imwrite(os.path.join(image_folder, f"{i}".zfill(4) + ".jpg"), per_x * 255)
            cv2.imwrite(os.path.join(mask_folder, f"{i}".zfill(4) + ".jpg"), per_y * 255)

    # Loading the data
    X, Y = load_data(folder_path, img_size, img_size, -1, 'kvasir')
    # Splitting the data, seed for reproducibility
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=True, random_state=seed_value)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.111, shuffle=True, random_state=seed_value)

    split_dataset_x = [x_train, x_valid, x_test]
    split_dataset_y = [y_train, y_valid, y_test]

    split_folder = {}
    for per_split, split_x, split_y in zip(split, split_dataset_x, split_dataset_y):
        folder = os.path.join(folder_path, "split", per_split)
        split_folder[per_split] = folder
        if not os.path.exists(folder):
            os.makedirs(folder)
            get_split_folder(split_x, split_y, folder)

    return split_folder

class Data_Loader:
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'images/*.jpg'))
        self.label_path = glob.glob(os.path.join(data_path, 'masks/*.jpg'))

    def __getitem__(self, index):
        # 根据index读取图片
        image = cv2.imread(self.imgs_path[index])
        label = cv2.imread(self.label_path[index], cv2.IMREAD_GRAYSCALE)
        label = label.reshape((label.shape[0], label.shape[1], 1))
        label_ = np.where(label > 127, True, False)

        return image.astype(np.float32), label_.astype(np.float32)

    @property
    def column_names(self):
        column_names = ['image', 'label']
        return column_names

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


def create_dataset(data_dir, batch_size, augment, shuffle, seed_value):
    mc_dataset = Data_Loader(data_path=data_dir)
    dataset = ds.GeneratorDataset(mc_dataset, mc_dataset.column_names, shuffle=shuffle)

    if augment:
        transform_img = [
            vision.RandomHorizontalFlip(prob=0.5),
            vision.RandomVerticalFlip(prob=0.5),
            # vision.TrivialAugmentWide(),
            # vision.AutoAugment(policy=vision.AutoAugmentPolicy.IMAGENET),
            # vision.Normalize(mean=mean, std=std),
            # vision.Affine(scale=(0.5, 1.5), translate_percent=(-0.125, 0.125), rotate=(-180, 180), shear=(-22.5, 22), always_apply=True),
            # vision.Affine(scale=(0.5, 1.5), translate=(-0.125, 0.125), shear=(-22.5, 22)),
            vision.RandomAffine(scale=(0.5, 1.5), translate=(-0.125, 0.125), degrees=(-180, 180), shear=(-22.5, 22)),
            vision.HWC2CHW()
        ]
    else:
        transform_img = [
            # vision.Decode(),
            # vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]

    ms.set_seed(seed_value)
    dataset = dataset.map(input_columns='image', num_parallel_workers=1, operations=transform_img)
    ms.set_seed(seed_value)
    dataset = dataset.map(input_columns="label", num_parallel_workers=1, operations=transform_img)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size, num_parallel_workers=1)
    if augment == True and shuffle == True:
        print("训练集数据量：", len(mc_dataset))
    elif augment == False and shuffle == False:
        print("验证集数据量：", len(mc_dataset))
    else:
        pass
    return dataset
