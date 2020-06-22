### Libraries
import torch.nn as nn, torch
from utils.normalization_layer import Norm2D as Norm
from efficientnet_pytorch import EfficientNet


class Basic_Block(nn.Module):
    def __init__(self, n_in, n_out, pars, depth=2):
        super(Basic_Block, self).__init__()
        self.activate = nn.LeakyReLU(0.2, inplace=True)

        ## Down convolution
        self.down = [nn.Conv2d(n_in, n_out, 3, 2, 1), Norm(n_out, pars)]
        self.down = nn.Sequential(*self.down)

        self.layer = []
        for _ in range(depth-1):
            self.layer.append(nn.Conv2d(n_out, n_out, 3, 1, 1))
            self.layer.append(Norm(n_out, pars))
            self.layer.append(self.activate)

        self.layer.extend([nn.Conv2d(n_out, n_out, 3, 1, 1)])
        self.layer.append(Norm(n_out, pars))
        self.layer = nn.Sequential(*self.layer)

    def forward(self, x):
        x = self.down(x)
        return self.activate(x + self.layer(x))


class Network(nn.Module):
    def __init__(self, dic):
        super(Network, self).__init__()

        self.dic = dic

        self.network = []
        in_channels = 3
        for out_channels in dic['channels']:
            self.network.append(Basic_Block(in_channels, out_channels, dic, depth=dic['depth']))
            in_channels = out_channels
        self.network = nn.Sequential(*self.network)

        self.out_size = int((dic["image_size"] / 2**len(dic["channels"]))**2 * out_channels)
        self.fc = nn.Sequential(nn.Linear(self.out_size, self.out_size//2),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(self.out_size//2, self.out_size//4),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(self.out_size//4, dic["n_classes"]))

    def forward(self, x):
        rep = self.network(x)
        pred = self.fc(rep.view(x.size(0), self.out_size))
        return pred


class Net(nn.Module):
    def __init__(self, dic):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        # self.model = EfficientNet.from_name('efficientnet-b0')
        # 1280 is the number of neurons in last layer. is diff for diff. architecture
        # self.dense_output = nn.Linear(2560, dic["n_classes"])
        self.dense_output = nn.Linear(1280, 512)
        self.output = nn.Linear(512, dic["n_classes"])
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.model.extract_features(x)
        feat = nn.functional.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)
        return self.output(self.act(self.dense_output(feat)))


if __name__ == '__main__':
    import os, argparse
    import auxiliaries as aux

    os.environ["CUDA_VISIBLE_DEVICES"] = '9'

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--config", type=str,
                        default='/export/home/mdorkenw/code/ALASKA2/network_base_setup.txt',
                        help="Define config file")
    args = parser.parse_args()
    opt = aux.extract_setup_info(args.config)[0]
    model = Network(opt.Network).cuda()
    print("Number of parameters in generator", sum(p.numel() for p in model.parameters()))
    dummy = torch.ones((2, 3, 512, 512)).cuda()
    breakpoint()
    print(model(dummy).shape)
