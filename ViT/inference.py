from PIL import Image
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import model
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # 에러 대응

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vit = model.VisionTransformer(patch_vec_size=48, num_patches=64,
                                  latent_vec_dim=128, num_heads=8, mlp_hidden_dim=64,
                                  drop_rate=0., num_layers=12, num_classes=10).to(device)
vit.load_state_dict(torch.load('ViT/model.pth'))    

def imshow(img):
    plt.figure(figsize=(4,4))   
    plt.imshow(img.permute(1,2,0).numpy())
    plt.axis('off')
    plt.show()
    # plt.close()
    return img

def inv_normal(img):
    img  = img.reshape(64, -1, 4, 4)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    print(img.size())
    for i in range(3):
        img[:,i,:,:] = torch.abs(img[:,i,:,:]*std[i] + mean[i])
    return img   

img = Image.open('ViT/3.jpg').convert('RGB')
img = img.resize((32,32))

img_mean = np.mean(img)
img_var = np.var(img)
img = (np.array(img) - img_mean) / np.sqrt(img_var)
# standard normalize 안하면 사이즈 32x32로 줄이면서 이미지 엄청 뿌옇게 들어감

img = torch.from_numpy(img).float()
img = img.permute(2, 0, 1)
# img.size() # torch.Size([3, 32, 32])

num_channels = img.size(0)  
patches = img.unfold(1, 4, 4).unfold(2, 4, 4).reshape(num_channels, -1, 4, 4)
# unfold를 써서 img.size(1)에 대해 자르고 (H 방향 자름)    다시 img.size(2)에 대해 자름 (W 방향 자름)
# 패치사이즈 16으로 가정, (채널, 16x16패치개수, 16x16패치모양) : 16x16짜리가 16x16개 있는 것, 따라서 가운데의 패치개수부분을 1열로 쭉 나열시키기 위해 -1

patches = patches.permute(1,0,2,3)  # 패치개수차원을 앞으로 빼냄, (16x16개의, 3채널짜리, 16, 16 패치)
num_patch = patches.size(0)

image_patch = patches.reshape(1, num_patch,-1)
image_patch.size() # torch.Size([1, 64, 48])

sample = inv_normal(image_patch)
original_img = imshow(torchvision.utils.make_grid(sample, nrow=8, padding=0))
# torch.Size([64, 3, 4, 4])

print(original_img.size())
# torch.Size([3, 32, 32])

_ = imshow(torchvision.utils.make_grid(sample, nrow=8, padding=1, pad_value=1))

vit.eval()
output, attention = vit(image_patch.to(device))
result = torch.argmax(output, dim=-1)
print('예측결과:')
if result[0] == 0:
    print('비행기')
elif result[0] == 1:
    print('자동차')
elif result[0] == 2:
    print('새')
elif result[0] == 3:
    print('고양이')
elif result[0] == 4:
    print('사슴')
elif result[0] == 5:
    print('강아지')
elif result[0] == 6:
    print('개구리')    
elif result[0] == 7:
    print('말')
elif result[0] == 8:
    print('배')
elif result[0] == 9:
    print('트럭')
    
