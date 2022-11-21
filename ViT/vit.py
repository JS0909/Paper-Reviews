import patchdata
import model
import test

import torch
import torch.optim as optim
import torch.nn as nn
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vision Transformer')
    parser.add_argument('--img_size', default=32, type=int, help='image size')
    parser.add_argument('--patch_size', default=4, type=int, help='patch size')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--save_acc', default=50, type=int, help='val acc') # validation accuracy가 해당 수치% 보다 높을 때만 저장
    parser.add_argument('--epochs', default=501, type=int, help='training epoch')
    parser.add_argument('--lr', default=2e-3, type=float, help='learning rate')
    parser.add_argument('--drop_rate', default=.1, type=float, help='drop rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay') # L2 regulization의 패널티 값
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
    parser.add_argument('--latent_vec_dim', default=128, type=int, help='latent dimension') # linear projection 이후 나오는 차원 (model의 풀디멘션)
    parser.add_argument('--num_heads', default=8, type=int, help='number of heads') # 어텐션의 헤드 개수
    parser.add_argument('--num_layers', default=12, type=int, help='number of layers in transformer') # ViT 쌓는 레이어 수
    parser.add_argument('--dataname', default='cifar10', type=str, help='data name')
    parser.add_argument('--mode', default='train', type=str, help='train or evaluation')
    parser.add_argument('--pretrained', default=0, type=int, help='pretrained model')
    args = parser.parse_args()
    print(args)

    latent_vec_dim = args.latent_vec_dim
    mlp_hidden_dim = int(latent_vec_dim/2)  # 인코더 내 MLP의 노드 수 (풀디멘션의 반으로 임의 설정)
    num_patches = int((args.img_size * args.img_size) / (args.patch_size * args.patch_size)) # H*W / P^2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Image Patches
    d = patchdata.Flattened2Dpatches(dataname=args.dataname, img_size=args.img_size, patch_size=args.patch_size,
                                     batch_size=args.batch_size)
    trainloader, valloader, testloader = d.patchdata()
    image_patches, _ = iter(trainloader).next() # 패치 사이즈 넘겨주기 위해 사용했음
    
    # patch_vec_size = 패치H*패치W*채널
    
    # Model
    vit = model.VisionTransformer(patch_vec_size=image_patches.size(2), num_patches=image_patches.size(1),
                                  latent_vec_dim=latent_vec_dim, num_heads=args.num_heads, mlp_hidden_dim=mlp_hidden_dim,
                                  drop_rate=args.drop_rate, num_layers=args.num_layers, num_classes=args.num_classes).to(device)
    
    if args.pretrained == 1:
        vit.load_state_dict(torch.load('ViT/model.pth'))

    if args.mode == 'train':
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(vit.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        #optimizer = torch.optim.SGD(vit.parameters(), lr=args.lr, momentum=0.9)
        #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(trainloader), epochs=args.epochs)

        # Train
        n = len(trainloader) # 배치 묶음의 개수
        best_acc = args.save_acc # 정해준 accuracy보다 높을 때만 save하도록 하는 용도
        for epoch in range(args.epochs):
            running_loss = 0
            for img, labels in trainloader:
                optimizer.zero_grad()
                outputs, _ = vit(img.to(device))
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                #scheduler.step()

            train_loss = running_loss / n
            val_acc, val_loss = test.accuracy(valloader, vit)
            # if epoch % 5 == 0:
            print('[%d] train loss: %.3f, validation loss: %.3f, validation acc %.2f %%' % (epoch, train_loss, val_loss, val_acc))

            if val_acc > best_acc:
                best_acc = val_acc
                # print('[%d] train loss: %.3f, validation acc %.2f - Save the best model' % (epoch, train_loss, val_acc))
                torch.save(vit.state_dict(), 'ViT/model.pth') # accuracy 기준 best만 저장함

    else:
        test_acc, test_loss = test.accuracy(testloader, vit)
        print('test loss: %.3f, test acc %.2f %%' % (test_loss, test_acc))