import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch

class PatchGenerator:

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, img):
        num_channels = img.size(0)  # img.size() : (채널, H, W)     한 장의 이미지에 대해 변형 중이므로 배치 차원 제외
        patches = img.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size).reshape(num_channels, -1, self.patch_size, self.patch_size)
        # unfold를 써서 img.size(1)에 대해 자르고 (H 방향 자름)    다시 img.size(2)에 대해 자름 (W 방향 자름)
        # 패치사이즈 16으로 가정, (채널, 16x16패치개수, 16x16패치모양) : 16x16짜리가 16x16개 있는 것, 따라서 가운데의 패치개수부분을 1열로 쭉 나열시키기 위해 -1
        
        patches = patches.permute(1,0,2,3)  # 패치개수차원을 앞으로 빼냄, (16x16개의, 3채널짜리, 16, 16 패치)
        num_patch = patches.size(0)

        return patches.reshape(num_patch,-1)    # 마지막으로 완전히 1열로 펴주는 작업, (16x16개, 채널x가로x세로), linear projection에 대한 input

class Flattened2Dpatches:

    def __init__(self, patch_size=16, dataname='imagenet', img_size=256, batch_size=64):
        self.patch_size = patch_size
        self.dataname = dataname
        self.img_size = img_size
        self.batch_size = batch_size

    def make_weights(self, labels, nclasses):
        labels = np.array(labels)
        weight_arr = np.zeros_like(labels)
        _, counts = np.unique(labels, return_counts=True)
        for cls in range(nclasses):
            weight_arr = np.where(labels == cls, 1/counts[cls], weight_arr) 
    
        return weight_arr 

    def patchdata(self):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        train_transform = transforms.Compose([transforms.Resize(self.img_size), transforms.RandomCrop(self.img_size, padding=2),
                                              transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std),
                                              PatchGenerator(self.patch_size)]) # patch로 잘라서 데이터로더를 만듦
        test_transform = transforms.Compose([transforms.Resize(self.img_size), transforms.ToTensor(),   # test 데이터는 augmentation 하지 않는다
                                             transforms.Normalize(mean, std), PatchGenerator(self.patch_size)])

        if self.dataname == 'cifar10': 
            trainset = torchvision.datasets.CIFAR10(root='ViT\data', train=True, download=True, transform=train_transform)
            testset = torchvision.datasets.CIFAR10(root='ViT\data', train=False, download=True, transform=test_transform)
            evens = list(range(0, len(testset), 2))
            odds = list(range(1, len(testset), 2))
            valset = torch.utils.data.Subset(testset, evens)    # testset의 짝수 index는 validation set으로 하고
            testset = torch.utils.data.Subset(testset, odds)    # 홀수 index는 test set으로 씀
          
        elif self.dataname == 'imagenet':
            pass

        weights = self.make_weights(trainset.targets, len(trainset.classes))  # 가중치 계산
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        trainloader = DataLoader(trainset, batch_size=self.batch_size, sampler=sampler) # sampler : 한 배치 당 클래스가 비슷한 개수를 sampling(뽑아) 후 배치 생성, stratify와 비슷한 역할
        valloader = DataLoader(valset, batch_size=self.batch_size, shuffle=False)
        testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)

        return trainloader, valloader, testloader

'''Weighted random sampling 동작방법
Weighted random sampling은 클래스 불균형 문제를 해결하기 위한 방법 중 하나입니다. 개별 이미지 한 장이 뽑힐 확률은 1/전체개수 입니다. 
따라서 이미지를 많이 가지고 있는 클래스가 뽑힐 확률이 더 높습니다. 
이를 보완하고자 더 적은 이미지를 갖는 클래스의 이미지가 뽑힐 확률을 높히도록 큰 가중치를 곱하고 
반대로 많은 이미지를 갖는 클래스의 이미지가 뽑힐 확률이 낮아지도록 작은 가중치를 곱하게 되어 클래스 당 확률을 동일하게 맞춰줍니다. 
이렇게 맞춰진 가중 확률을 기반으로 Sampler가 이미지를 확률적으로 골라서 배치를 만들게 됩니다.

따라서 데이터가 중복으로 뽑힐 수 있다.
매 배치마다 클래스 개수가 balanced한 배치 셋을 생성한다.'''


# 패치 분할 확인 용 코드 - 주의점: patchdata 함수의 픽셀 normalization부분 빼고 출력해야 원래 이미지모양대로 나옴. 안그러면 분포 바꿔 나와서 이상함.
def imshow(img):
    plt.figure(figsize=(100,100))
    plt.imshow(img.permute(1,2,0).numpy())  # torchTensor -> numpy로 바꿔야 imshow사용할 수 있음
    plt.savefig('pacth_example.png')

if __name__ == "__main__":
    print("Testing Flattened2Dpatches..")
    batch_size = 64
    patch_size = 8
    img_size = 32
    num_patches = int((img_size*img_size)/(patch_size*patch_size))
    d = Flattened2Dpatches(dataname='cifar10', img_size=img_size, patch_size=patch_size, batch_size=batch_size)
    trainloader, _, _ = d.patchdata()
    images, labels = iter(trainloader).next()   # 만들어진 데이터로더의 내부 데이터를 확인하고자 할 경우 iter(데이터로더).next()를 사용한다
    print(images.size(), labels.size())

    sample = images.reshape(batch_size, num_patches, -1, patch_size, patch_size)[0] # flattened 데이터이므로 다시 모양 잡아서 화면에 보여줌
    print("Sample image size: ", sample.size())
    imshow(torchvision.utils.make_grid(sample, nrow=int(img_size/patch_size)))
