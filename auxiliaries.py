import numpy as np, time, random, csv, glob
import torch, ast, pandas as pd, copy, itertools as it, os, torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
import itertools as it, copy
import matplotlib.pyplot as plt
from sklearn import metrics


class dataset(torch.utils.data.Dataset):
    def __init__(self, opt, mode='train', seed=1):

        self.rng = np.random.RandomState(seed)
        self.img_path = '/export/data/mdorkenw/data/alaska2/'
        self.mode = mode
        self.jpeg_comp = np.load(self.img_path + 'JPEG_compression.npy')

        if mode=='train':
            self.length = int(opt.Training['train_size'])
            self.method = ['Cover', 'JUNIWARD', 'JMiPOD', 'UERD']
            self.data_path = glob.glob(self.img_path + "Cover/*.jpg")
            self.idx_dict = [i[-(4 + 5):-4] for i in self.data_path]
            self.offset = 0
        elif mode=='evaluation':
            self.length = int(opt.Training['evaluation_size'])
            self.method = ['Cover', 'JUNIWARD', 'JMiPOD', 'UERD']
            self.data_path = glob.glob(self.img_path + "Cover/*.jpg")
            self.idx_dict = [i[-(4 + 5):-4] for i in self.data_path]
            self.offset = opt.Training['train_size']
        else:
            self.length = int(opt.Training['test_size'])
            self.method = ['Test']
            self.data_path = glob.glob(self.img_path + "Test/*.jpg")
            self.idx_dict = [i[-(4 + 5):-4] for i in self.data_path]
            self.offset = 0

        self.n_classes = opt.Network['n_classes']

        self.augment_train = transforms.Compose([
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomHorizontalFlip(p=0.5),
             # transforms.RandomCrop(224),
             transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.augment_test  = transforms.Compose([
            # transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_1hot_(self, label):
        hot1 = np.zeros(self.n_classes)
        hot1[label] = 1
        return hot1

    def load_and_augment_(self, data_path):
        if self.mode=='test' or self.mode=='evaluation':
            return self.augment_train(Image.open(data_path))
        else:
            return self.augment_test(Image.open(data_path))

    def __getitem__(self, idx):
        if self.n_classes == 1 or self.mode == 'evaluation':
            mode = np.random.choice([0, 1, 2, 3], p=[0.5, 0.5/3, 0.5/3, 0.5/3])
        else:
            mode = np.random.randint(0, len(self.method))
        image = self.load_and_augment_(self.img_path + self.method[mode] + "/" + self.idx_dict[idx + self.offset] + ".jpg")
        # label = self.get_1hot_(mode)
        compression = self.jpeg_comp[idx]
        if self.n_classes == 4:
            label = torch.tensor([mode])
        elif self.n_classes == 12:
            label = torch.tensor([mode*3 + compression])

        if self.n_classes == 1:
            label[label > 1] = 1
        return {'image': image, 'label': label}

    def __len__(self):
        return self.length


### Function to extract setup info from text file ###
def extract_setup_info(config_file):
    baseline_setup = pd.read_csv(config_file, sep='\t', header=None)
    baseline_setup = [x for x in baseline_setup[0] if '=' not inct x]
    sub_setups = [x.split('#')[-1] for x in np.array(baseline_setup) if '#' in x]
    vals = [x for x in np.array(baseline_setup)]
    set_idxs = [i for i, x in enumerate(np.array(baseline_setup)) if '#' in x] + [len(vals)]

    settings = {}
    for i in range(len(set_idxs) - 1):
        settings[sub_setups[i]] = [[y.replace(" ", "") for y in x.split(':')] for x in
                                   vals[set_idxs[i] + 1:set_idxs[i + 1]]]

    # d_opt = vars(opt)
    d_opt = {}
    for key in settings.keys():
        d_opt[key] = {subkey: ast.literal_eval(x) for subkey, x in settings[key]}

    opt = argparse.Namespace(**d_opt)
    if d_opt['Paths']['network_variation_setup_file'] == '':
        return [opt]

    variation_setup = pd.read_csv(d_opt['Paths']['network_variation_setup_file'], sep='\t', header=None)
    variation_setup = [x for x in variation_setup[0] if '=' not in x]
    sub_setups = [x.split('#')[-1] for x in np.array(variation_setup) if '#' in x]
    vals = [x for x in np.array(variation_setup)]
    set_idxs = [i for i, x in enumerate(np.array(variation_setup)) if '#' in x] + [len(vals)]
    settings = {}
    for i in range(len(set_idxs) - 1):
        settings[sub_setups[i]] = []
        for x in vals[set_idxs[i] + 1:set_idxs[i + 1]]:
            y = x.split(':')
            settings[sub_setups[i]].append([[y[0].replace(" ", "")], ast.literal_eval(y[1].replace(" ", ""))])
        # settings

    all_c = []
    for key in settings.keys():
        sub_c = []
        for s_i in range(len(settings[key])):
            sub_c.append([[key] + list(x) for x in list(it.product(*settings[key][s_i]))])
        all_c.extend(sub_c)

    setup_collection = []
    training_options = list(it.product(*all_c))
    for variation in training_options:
        base_opt = copy.deepcopy(opt)
        base_d_opt = vars(base_opt)
        # WHY??? you never use base_d_opt again
        for i, sub_variation in enumerate(variation):
            base_d_opt[sub_variation[0]][sub_variation[1]] = sub_variation[2]
            base_d_opt['iter_idx'] = i
        setup_collection.append(base_opt)

        return setup_collection


def gimme_save_string(opt):
    varx = vars(opt)
    base_str = ''
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key],dict):
            for sub_key, sub_item in varx[key].items():
                base_str += '\n\t'+str(sub_key)+': '+str(sub_item)
        else:
            base_str += '\n\t'+str(varx[key])
        base_str+='\n\n'
    return base_str


class CSVlogger():
    def __init__(self, logname, header_names):
        self.header_names = header_names
        self.logname      = logname
        with open(logname,"a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(header_names)

    def write(self, inputs):
        with open(self.logname,"a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(inputs)


def progress_plotter(x, train_loss, train_metric, labels, savename='result.svg'):
    plt.style.use('ggplot')
    f, ax = plt.subplots(1)
    ax.plot(x, train_loss, 'b--', label=labels[0])

    axt = ax.twinx()
    axt.plot(x, train_metric, 'b', label=labels[1])

    ax.legend(loc=0)
    axt.legend(loc=2)

    f.suptitle('Loss and Evaluation Metric Progression')
    f.set_size_inches(15, 10)
    f.savefig(savename)
    plt.close()


def summary_plots(loss_dic_train, loss_dic_test, epoch, save_path):
    progress_plotter(np.arange(0, len(loss_dic_train["Loss"])), loss_dic_train["Loss"], loss_dic_test["AUC"],
                     ["Loss Train", "AUC Test"], save_path + '/Loss_AUC.png')


class loss_tracking():
    def __init__(self, names):
        super(loss_tracking, self).__init__()
        self.loss_dic = names
        self.hist = {x: np.array([]) for x in self.loss_dic}
        self.keys = [*self.hist]

    def reset(self):
        self.dic = {x: np.array([]) for x in self.loss_dic}

    def append(self, losses):
        assert (len(self.keys)-1 == len(losses))
        for idx in range(len(losses)):
            self.dic[self.keys[idx]] = np.append(self.dic[self.keys[idx]], losses[idx])

    def append_auc(self, auc):
        self.dic['AUC'] = np.append(self.dic['AUC'], auc)

    def get_iteration_mean(self):
        mean = []
        for idx in range(len(self.keys)-1):
            if len(self.dic[self.keys[idx]]) < 100:
                mean.append(np.mean(self.dic[self.keys[idx]]))
            else:
                mean.append(np.mean(self.dic[self.keys[idx]][-100:]))
        return mean

    def get_mean(self):
        self.mean = []
        for idx in range(len(self.keys)):
            self.mean.append(np.mean(self.dic[self.keys[idx]]))
        self.history()

    def history(self):
        for idx in range(len(self.keys)):
            self.hist[self.keys[idx]] = np.append(self.hist[self.keys[idx]], self.mean[idx])

    def get_current_mean(self):
        return self.mean

    def get_hist(self):
        return self.hist


class Base_Loss(nn.Module):
    def __init__(self, dic):
        super(Base_Loss, self).__init__()
        self.n_classes = dic['n_classes']
        if self.n_classes > 1:
            self.loss    = nn.CrossEntropyLoss()
        else:
            self.loss    = nn.BCEWithLogitsLoss()

    def forward(self, inp, target):
        if self.n_classes > 1:
            return self.loss(inp, target.long().reshape(-1))
        else:
            return self.loss(inp.reshape(-1), target.reshape(-1))


def auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)

    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)

        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min  # normalize such that curve starts at y=0
        score = metrics.auc(x, y)
        submetric = score * weight
        competition_metric += submetric

    return competition_metric / normalization


## Alterantive loss fom kaggle
class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.05):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)

            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)

