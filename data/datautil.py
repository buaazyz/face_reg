import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import bcolz

class Dataset(data.Dataset):

    def __init__(self, root, data_list_file, phase='train', input_shape=(1, 112, 112)):
        self.phase = phase
        self.input_shape = input_shape

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        imgs = [os.path.join(root, img[:-1]) for img in imgs]
        self.imgs = np.random.permutation(imgs)

        # normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
        #                         std=[0.5, 0.5, 0.5])

        normalize = T.Normalize(mean=[0.5], std=[0.5])

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.RandomCrop(self.input_shape[1:]),
                # T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split(',')
        img_path = splits[0]
        data = Image.open(img_path)
        data = data.convert('RGB')
        data = self.transforms(data)
        label = np.int32(splits[1])
        return data.float(), label

    def __len__(self):
        return len(self.imgs)


def get_val_pair(path, name):
    # a = path / name
    # print(a)
    carray = bcolz.carray(rootdir=path, mode='r')
    issame = np.load(path / '{}_list.npy'.format(name))
    return carray, issame


def get_val_data(data_path):
    essex, essex_issame = get_val_pair(data_path, 'essex')
    return essex, essex_issame




if __name__ == '__main__':
    ds = Dataset(root='D:/PycharmProjects/myface/data/pic',
                      data_list_file='D:/PycharmProjects/myface/data/list_train.txt',
                      phase='train',
                      input_shape=(1, 112, 112))

    trainloader = data.DataLoader(ds, batch_size=32)
    temp = []
    for i in trainloader:
        # print(i[1])
        for k in i[1]:
            temp.append(int(k))
    print(set(temp))