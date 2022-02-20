#시스템 모듈
import numpy as np
import os
import copy
import time

#Image Handling 모듈
import matplotlib.pyplot as plt
from glob import glob
from skimage import io, transform
from PIL import Image

#Pytorch 모듈
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split

from Neural import *
from Dataloader import *

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def Evaluator(testloaders, model, device) :
    if (device) :
        print("Check Device : ", device)
        model = model.to(device)
    model.eval()
    test_loss = 0.0

    with torch.no_grad() :
        for i, data in enumerate(testloaders, 0):
            inputs, labels = data
            if (device):
                inputs = inputs.to(device)
            outputs = model(inputs)

            outputs = outputs.to('cpu').data.numpy()
            labels = labels.numpy()

            test_loss += np.sum((outputs - labels)**2) / len(labels)

    print(np.sqrt(test_loss) / len(testloaders))

if __name__ == '__main__' :
    MODEL = "Custom"
    if MODEL == "Custom":
        transform = transforms.Compose([
            transforms.CenterCrop((212, 212)),
            transforms.Resize((45, 45)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 360)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop((214, 214)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 360)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    # Model 불러오기
    model = SanderDielemanNet()
    model_name = "cifar_net__2022-02-20_13-51-57.pth"
    PATH = os.path.join(os.getcwd(), 'results', model_name)
    model.load_state_dict(torch.load(PATH))

    RootDir = os.path.join(os.getcwd(), 'galaxy_zoo')
    print("Root Directory : ", RootDir)
    BatchSize = 256

    Transformed_dataset = MyGalaxySet(RootDir, transform=transform)
    trainset, testset = random_split(Transformed_dataset, [55000, 6578])
    testloader = DataLoader(
        testset,
        batch_size=BatchSize,
        shuffle=True,
        num_workers=2
    )


    #이미지 판별 테스트
    #저장된 파일을 불러올땐 파일이름을 다음에 입력하세요.

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))
    #https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/overview/evaluation
    #print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # Device 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    Evaluator(testloader, model, device)

    #print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
    #                              for j in range(4)))
