#시스템 모듈
import os
import copy
import numpy as np
import time

#Pytorch 모듈
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split

#Image Handling 모듈
import matplotlib.pyplot as plt
from glob import glob
from skimage import io, transform
from PIL import Image

#pandas
import pandas as pd

from Neural import *
from Dataloader import *

# 신경망에 모델을 학습시키는 함수를 정의하는 블럭
def train_model(model, dataloaders, criterion, optimizer, device=None, num_epochs=25, scheduler = None):
    since = time.time()
    if (device):
        print("Check Device : ", device)
    model.train()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0.0

        # Iterate over data.
        for i, data in enumerate(dataloaders, 0):
            inputs, labels = data
            if (device):
                inputs = inputs.to(device)
                labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            running_corrects += loss.item()

            if i % 32 == 31:  # print every 32 mini-batche
                print('[%d, %5d] Local loss: %.3f' %
                      (epoch + 1, i + 1, running_corrects / 32))
                running_corrects = 0.0
        print('Epoch Done. Loss at this epoch : {}'.format(running_loss / len(dataloaders)))
        if(scheduler) :
            scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model

#데이터셋 로드 테스트
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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

    # classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # csv file loaction
    RootDir = os.path.join(os.getcwd(), 'galaxy_zoo')
    print("Root Directory : ", RootDir)
    BatchSize = 256

    Transformed_dataset = MyGalaxySet(RootDir, transform=transform)
    trainset, testset = random_split(Transformed_dataset, [55000, 6578])
    trainloader = DataLoader(
        trainset,
        batch_size=BatchSize,
        shuffle=True,
        num_workers=2
    )

    # dataloaders = {'train': trainloader, 'val' : testloader}
    # image_datasets = {'train' : datasets.ImageFolder('./CIFAR-10-images-master/train/', trainloader), 'val' : datasets.ImageFolder('./CIFAR-10-images-master/test/', testloader)}
    # dataset_sizes = {'train' : len(image_datasets['train']), 'val' : len(image_datasets['val'])}

    for i in range(len(Transformed_dataset)):
        leninfo = Transformed_dataset.__len__()
        print(leninfo)

        image, landmarks = Transformed_dataset.__getitem__(i)
        print(i)
        print(image.size())
        print(landmarks.size())

        if i == 3:
            break
    model = SanderDielemanNet()
    # Device 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())

    # Parameter 설정
    criterion = nn.MSELoss()
    #optimizer_ft = optim.SGD(model.parameters(), lr=0.4, momentum=0.9)
    optimizer_ft = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.95)

    # 학습용 이미지를 무작위로 가져오기
    #dataiter = iter(trainloader)
    #images, landmarks = dataiter.next()

    # 이미지 보여주기
    #imshow(torchvision.utils.make_grid(images))
    # 정답(label) 출력
    # print(' '.join('%5s' % classes[labels[j]] for j in range(BatchSize)))

    # 모델 학습
    if (device):
        model = model.to(device)
    model_ft = train_model(model, trainloader, criterion, optimizer_ft, device=device, num_epochs=25, scheduler = scheduler)

    # Colab 안끊기게 하기
    # Colab은 90분동안 사용자 입력이 없을 시 자동으로 런타임 연결을 해제합니다.
    # 웹브라우저 콘솔에 60초마다 한번씩 클릭을 해줘서 회피하는 방법입니다.
    # 사용법 : F12 -> Console(콘솔) 클릭 -> 코드 복사-붙여넣기(앞에 주석은 떼주세요! Javascript코드입니다.)
    # function ClickConnect(){
    #    console.log("코랩 연결 끊김 방지");
    #    document.querySelector("colab-toolbar-button#connect").click()
    # }
    # setInterval(ClickConnect, 60 * 1000)

    # 학습결과를 파일으로 저장
    # 돌려놓고 갔다와도 학습결과가 파일으로 저장되어 있기 때문에 런타임이 끊겨도 이 코준블럭은 안돌려도 됩니다.
    # 저장위치 : Colab Notebooks 폴더 내에 'results'폴더를 만들고 실행하면 거기에 pth(파이토치 파일형식)파일형태로 저장이 됩니다.
    # 파일 생성시 시간이 기록됩니다.(UTC 기준)
    PATH = os.path.join(os.getcwd(), 'results', 'cifar_net__' + time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())) + '.pth')
    torch.save(model.state_dict(), PATH)