import numpy as np
import torch, os, torch.nn as nn, glob
from torchvision import transforms
from PIL import Image


class dataset(torch.utils.data.Dataset):
    def __init__(self, opt, mode='train', seed=1):

        self.rng = np.random.RandomState(seed)
        self.img_path = '/export/data/mdorkenw/data/alaska2/'
        self.mode = mode
        self.jpeg_comp = np.load(self.img_path + 'JPEG_compression.npy')
        self.use_attention = opt.Network['attention_mask']
        self.input_domain = opt.Network['input_domain']
        self.img_size = opt.Network['image_size']

        if mode == 'train':
            self.length = int(opt.Training['train_size'])
            self.method = ['Cover', 'JUNIWARD', 'JMiPOD', 'UERD']
            self.data_path = glob.glob(self.img_path + "Cover/*.jpg")
            self.idx_dict = [i[-(4 + 5):-4] for i in self.data_path]
            self.offset = 0
        elif mode == 'evaluation':
            self.length = int(opt.Training['evaluation_size'])
            self.method = ['Cover', 'JUNIWARD', 'JMiPOD', 'UERD']
            self.data_path = glob.glob(self.img_path + "Cover/*.jpg")
            self.idx_dict = [i[-(4 + 5):-4] for i in self.data_path]
            self.offset = opt.Training['train_size']
        elif mode == 'test':
            self.length = int(opt.Training['test_size'])
            self.method = ['Test']
            self.data_path = glob.glob(self.img_path + "Test/*.jpg")
            self.idx_dict = [i[-(4 + 5):-4] for i in self.data_path]
            self.offset = 0
        else:
            raise NotImplementedError('Specified mode is not implemented')

        self.n_classes = opt.Network['n_classes']

        self.augment_train = transforms.Compose([
             transforms.Resize(self.img_size),
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.augment_test  = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.high_pass_filter = torch.nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        kernel = torch.from_numpy(np.array([[0, -1 / 4, 0], [-1 / 4, 1, -1 / 4], [0, -1 / 4, 0]]))
        self.high_pass_filter.weight = torch.nn.Parameter(kernel.unsqueeze(0).unsqueeze(0).float())
        # _ = self.high_pass_filter.cuda()

    def get_1hot_(self, label):
        hot1 = np.zeros(self.n_classes)
        hot1[label] = 1
        return hot1

    def load_and_augment_RGB(self, data_path):
        img = Image.open(data_path)
        if self.mode=='test' or self.mode == 'evaluation':
            return self.augment_train(img)
        else:
            return self.augment_test(img)

    def load_and_augment_DCT(self, data_path):
        dct = torch.from_numpy(1/(np.load(data_path)+1e-3)).float().permute(2, 0, 1)
        return dct/dct.max()

    def __getitem__(self, idx):
        if self.n_classes == 1 or self.mode == 'evaluation':
            mode = np.random.choice([0, 1, 2, 3], p=[0.5, 0.5/3, 0.5/3, 0.5/3])
        else:
            mode = np.random.randint(0, len(self.method))

        img_dir = self.img_path + self.method[mode] + "/" + self.idx_dict[idx + self.offset] + ".jpg"
        dct_dir = self.img_path + 'DCT/' + self.method[mode] + "/" + self.idx_dict[idx + self.offset][1:] + '_block.npy'

        if self.input_domain == 'RGB':
            input = self.load_and_augment_RGB(img_dir)
        elif self.input_domain == 'DCT':
            input = self.load_and_augment_DCT(dct_dir)

        if self.n_classes == 1:
            if mode > 1:
                mode = 1

        if self.n_classes == 12:
            mode = mode*3 + self.jpeg_comp[idx]

        label = self.get_1hot_(mode)

        mask = torch.zeros(1)
        if self.use_attention:
            mask = self.high_pass_filter(transforms.ToTensor()(Image.open(img_dir).convert('L')).unsqueeze(0))
            mask = torch.abs(mask).squeeze(0).detach()

        return {'input': input, 'label': label, 'mask': mask/mask.max()}

    def __len__(self):
        return self.length