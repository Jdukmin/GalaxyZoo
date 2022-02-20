#시스템 모듈
import os
import copy
import numpy as np
import time

#Pytorch 모듈
import torch
from torch.utils.data import Dataset, DataLoader

from skimage import io, transform
from PIL import Image

#pandas
import pandas as pd

#Google Drive 마운트
#from google.colab import drive, files
#drive.mount('/content/drive')

##__getitem__함수가 동작하지 않음!

class MyCifarSet(Dataset):
    def __init__(self, data_path_list, classes, transform=None):
        # x : path_list
        self.path_list = data_path_list
        self.label = self.get_label(data_path_list)
        self.transform = transform
        # y : classes
        self.classes = classes

    def __len__(self):
        return len(self.path_list)

    def get_label(self, data_path_list):
        label_list = []
        for path in data_path_list:
            label_list.append(path.split('/')[-2])
        return label_list

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = io.imread(self.path_list[idx])

        target = self.classes.index(self.label[idx])
        if self.transform is not None:
            image = self.transform(image)
            target = self.transform(target)

        return image, target


class MyGalaxySet(Dataset):
    def __init__(self, data_path, transform=None):
        self.PATH_image = os.path.join(data_path, 'images_training_rev1')
        self.PATH_csv = os.path.join(data_path, 'training_solutions_rev1.csv')
        # self.LIST_image = glob(os.path.join(self.PATH_image, '*.jpg'))
        print(self.PATH_image)
        print(self.PATH_csv)

        # self.data = [] # 데이터를 정제해 담을 행렬 생성
        self.transform = transform  # 데이터 받아오기 이후 데이터 전처리

        self.df_mnist = pd.read_csv(self.PATH_csv, encoding='CP949')  # pandas를 이용한 csv 파일 읽기
        self.df_mnist.head()
        # image_name = os.path.join(self.PATH_image, self.df_mnist[idx, 0] + '.jpg')
        print("#####Initialize Dataset#####")
        print("CSV Shape : ", self.df_mnist.shape)

    def __len__(self):
        return len(self.df_mnist)

    def __getitem__(self, idx):
        image_name = str(self.df_mnist.iloc[idx, 0]) + ".jpg"
        landmarks = self.df_mnist.iloc[idx, 1:].values

        PATH_image_idx = os.path.join(self.PATH_image, image_name)
        # image = io.imread(PATH_image_idx)
        image = Image.open(PATH_image_idx)

        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float')

        if self.transform:
            image = self.transform(image)
            landmarks = torch.FloatTensor(landmarks)

        return image, landmarks