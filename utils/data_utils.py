from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch


def get_loader(args):
    transform_train = transforms.Compose([
        #transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        #transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "mnist":
        train_set = datasets.MNIST(root='./data',
                                   train=True,
                                   download=True,
                                   transform=transforms.ToTensor())
        test_set = datasets.MNIST(root='./data',
                                  train=False,
                                  download=True,
                                  transform=transforms.ToTensor())
        chw = (1, 28, 28)

    elif args.dataset == "cifar10":
        train_set = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        test_set = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) 
        chw = (3, 32, 32)
    else:
        train_set = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        test_set = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test)
        chw = (3, 32, 32)

    train_loader = DataLoader(train_set,
                              shuffle=True,
                              batch_size=args.train_batch_size)
    test_loader = DataLoader(test_set,
                             shuffle=False,
                             batch_size=args.eval_batch_size)

    return train_loader, test_loader, chw


def patchify(images, patch_size):
    """Divides images into patches"""

    b_size, img_c, img_w, img_h = images.shape

    assert img_w == img_h, "This implementation only considers images with equal widths and heights"
    assert img_w % patch_size == 0, "Image size is not a multiple of patch size"

    n_patches_dim = int(img_w / patch_size)
    n_patches = pow(n_patches_dim, 2)

    patches = torch.zeros(b_size, n_patches, patch_size*patch_size*img_c)
    for i, image in enumerate(images):
        p = 0
        for h in range(n_patches_dim):
            for w in range(n_patches_dim):
                patches[i, p, :] = image[:, h*patch_size:(h+1)*patch_size, w*patch_size:(w+1)*patch_size].flatten()
                p+=1

    return patches
