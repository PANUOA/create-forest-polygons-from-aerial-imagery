import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random


class z15_Loader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'imgs/*.png'))
        self.label_path = glob.glob(os.path.join(data_path, 'masks/*.png'))

    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = self.label_path[index]
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        if label.max() > 1:
            label = label / 255

        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        return len(self.imgs_path)


if __name__ == "__main__":
    z15_dataset = z15_Loader("../data/train/")
    print("Size of z15: ", len(z15_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=z15_dataset,
                                               batch_size=2,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
