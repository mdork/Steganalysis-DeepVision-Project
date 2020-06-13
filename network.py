
### Libraries
import torch.nn as nn, torch





################################################################################################
### NOTE: Divide and put into channels? ########################################################
################################################################################################
class ConvLayer(nn.Module):
    def __init__(self, n_in, n_out, pars):
        super(ConvLayer, self).__init__()
        self.layer  = [nn.Conv2d(n_in, n_out, 3, 1, 1)]
        if pars['use_BN']: self.layer.append(nn.BatchNorm2d(n_out))
        self.layer.extend([nn.LeakyReLU(0.2), nn.Dropout2d(pars['dropout'])])
        self.layer = nn.Sequential(*self.layer)

    def forward(self, x):
        return self.layer(x)


class ConvBlock(nn.Module):
    def __init__(self, idx, pars):
        super(ConvBlock, self).__init__()
        self.reps      = pars['setup'][idx]
        self.in_filters, self.out_filters  = pars['start_filter']*2**(idx), pars['start_filter']*2**(idx+1)
        self.elements    = [ConvLayer(self.in_filters, self.out_filters, pars) if i==0 else ConvLayer(self.out_filters, self.out_filters, pars) for i in range(self.reps)]
        self.elements    = nn.ModuleList(self.elements)

    def forward(self, x):
        for layer in self.elements:
            x = layer(x)
        return x


class Extract_Scaffold(nn.Module):
    def __init__(self, opt):
        """
        Passable arguments:
            channels     : Number of input channels - default 1 for green channel
            start_filter : Number of feature maps to start with. Per pooling get updated by 2**(pooling depth)
            n_classes    : Number of target classes for multilabel problem.
            setup        : Structure of conv-blocks divided by pooling layers
        """
        super(Extract_Scaffold, self).__init__()
        self.pars = opt.Network

        self.layer_in              = nn.Conv2d(self.pars['channels'], self.pars['start_filter'], 3, 1, 1)
        self.feature_extract       = nn.ModuleList([ConvBlock(idx, self.pars) for idx in range(len(self.pars['setup']))])
        self.layer_out             = nn.Conv2d(self.feature_extract[-1].out_filters, self.pars['n_classes'], 3, 1, 1)

        self.pool    = torch.nn.functional.avg_pool2d #use conv pool alternatively
        self.out_act = nn.Sigmoid()


    def forward(self, x):
        x = self.layer_in(x)
        for i,layer in enumerate(self.feature_extract):
            x = layer(x)
            if i<len(self.feature_extract)-1:
                x = self.pool(x, (2,2))

        x = self.layer_out(x)
        #Use global average pooling instead of fully connected layers
        x = torch.nn.functional.avg_pool2d(x, x.size()[-2:])
        x = self.out_act(x)

        bs,ch = x.size()[:2]
        return x.view(bs,ch,-1)[:,:,0]
