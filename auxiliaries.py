import numpy as np, time, random, csv
import torch, ast, pandas as pd, copy, itertools as it, os, torch.nn as nn
from torchvision import transforms
from PIL import Image
import itertools as it, copy
import matplotlib.pyplot as plt

"""============================================"""
### Dataset for Dataloader
class dataset(torch.utils.data.Dataset):
    def __init__(self, opt, mode='train', seed=1):
        self.rng = np.random.RandomState(seed)

        main_path, assign_file = opt.Paths['train_data'], opt.Paths['assign_file']
        tv_split, perc_data    = opt.Training['train_val_split'], opt.Training['perc_data']

        self.assign_file = pd.read_csv(assign_file, header=0)
        self.data_paths  = [[main_path+'/'+x+'_'+y+'.png' for y in ['red', 'yellow', 'blue','green']] for x in self.assign_file['Id']]
        self.labels      = [ast.literal_eval(x.replace(' ',',')) for x in self.assign_file['Target']]
        self.labels      = [[x] if not isinstance(x,tuple) else list(x) for x in self.labels]

        random.seed(seed)
        random.shuffle(self.data_paths)
        random.seed(seed)
        random.shuffle(self.labels)

        perc_data  = int(len(self.labels)*perc_data)
        tv_split   = int(perc_data*tv_split)

        self.data_paths  = self.data_paths[:perc_data]
        self.labels      = self.labels[:perc_data]

        if mode=='train':
            self.data_paths  = self.data_paths[:tv_split]
            self.labels      = self.labels[:tv_split]
        else:
            self.data_paths  = self.data_paths[tv_split:]
            self.labels      = self.labels[tv_split:]


        self.max_label   = opt.Network['n_classes']
        img_mean, img_std = 0,1

        # self.augment     = transforms.Compose([transforms.Rescale(256),
        #                                        transforms.RandomAffine(degrees=180, translate=0.15, scale=(0.8,1.2), fillcolor=0),
        #                                        transforms.ColorJitter(),
        #                                        transforms.ToTensor(),
        #                                        transforms.Normalize(img_mean, img_std)
        #                                        ])

        self.augment        = transforms.Compose([transforms.ToTensor()])
        self.cost_per_class = np.ones(self.max_label)

        self.n_files = len(self.labels)

    def get_1hot_(self, labels):
        hot1 = np.zeros(self.max_label)
        hot1[labels] = 1
        return hot1

    def load_and_augment_(self, data_paths):
        return torch.cat([self.augment(Image.open(data_path)) for data_path in data_paths], dim=0)

    def __getitem__(self, idx):
        image = self.load_and_augment_(self.data_paths[idx])
        label = self.get_1hot_(self.labels[idx])
        return {'Image':image, 'label':label}

    def __len__(self):
        return self.n_files



"""======================================================="""
### Function to extract setup info from text file ###
def extract_setup_info(opt):
    baseline_setup = pd.read_table(opt.Paths['network_base_setup_file'], header=None)
    baseline_setup = [x for x in baseline_setup[0] if '=' not in x]
    sub_setups     = [x.split('#')[-1] for x in np.array(baseline_setup) if '#' in x]
    vals           = [x for x in np.array(baseline_setup)]
    set_idxs       = [i for i,x in enumerate(np.array(baseline_setup)) if '#' in x]+[len(vals)]
    settings = {}
    for i in range(len(set_idxs)-1):
        settings[sub_setups[i]] = [[y.replace(" ","") for y in x.split(':')] for x in vals[set_idxs[i]+1:set_idxs[i+1]]]

    d_opt = vars(opt)
    for key in settings.keys():
        d_opt[key] = {subkey:ast.literal_eval(x) for subkey,x in settings[key]}

    if opt.Paths['network_variation_setup_file'] == '':
        return [opt]


    variation_setup = pd.read_table(opt.Paths['network_variation_setup_file'], header=None)
    variation_setup = [x for x in variation_setup[0] if '=' not in x]
    sub_setups      = [x.split('#')[-1] for x in np.array(variation_setup) if '#' in x]
    vals            = [x for x in np.array(variation_setup)]
    set_idxs        = [i for i,x in enumerate(np.array(variation_setup)) if '#' in x]+[len(vals)]
    settings = {}
    for i in range(len(set_idxs)-1):
        settings[sub_setups[i]] = []
        for x in vals[set_idxs[i]+1:set_idxs[i+1]]:
            y = x.split(':')
            settings[sub_setups[i]].append([[y[0].replace(" ","")], ast.literal_eval(y[1].replace(" ",""))])
        settings

    all_c = []
    for key in settings.keys():
        sub_c = []
        for s_i in range(len(settings[key])):
            sub_c.append([[key]+list(x) for x in list(it.product(*settings[key][s_i]))])
        all_c.extend(sub_c)


    setup_collection = []
    training_options = list(it.product(*all_c))
    for variation in training_options:
        base_opt   = copy.deepcopy(opt)
        base_d_opt = vars(base_opt)
        for i,sub_variation in enumerate(variation):
            base_d_opt[sub_variation[0]][sub_variation[1]] = sub_variation[2]
            base_d_opt['iter_idx'] = i
        setup_collection.append(base_opt)

    return setup_collection


"""==================================================="""
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


"""==================================================="""
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


"""===================================================="""
def progress_plotter(x, train_loss, train_metric, val_loss, val_metric, savename='result.svg', title='No title'):
    plt.style.use('ggplot')
    f,ax = plt.subplots(1)
    ax.plot(x, train_loss,'b--',label='Training Loss')
    ax.plot(x, val_loss,'r--',label='Validation Loss')

    axt = ax.twinx()
    axt.plot(x, train_metric, 'b', label='Training Dice')
    axt.plot(x, val_metric, 'r', label='Validation Dice')

    ax.set_title(title)
    ax.legend(loc=0)
    axt.legend(loc=2)

    f.suptitle('Loss and Evaluation Metric Progression')
    f.set_size_inches(15,10)
    f.savefig(savename)
    plt.close()



"""===================================================="""
class Base_Loss(nn.Module):
    def __init__(self, weights=None):
        super(Base_Loss, self).__init__()
        self.weights = weights.type(torch.cuda.FloatTensor) if weights is not None else None
        self.loss    = nn.BCELoss(weight=self.weights)
        # self.loss    = nn.CrossEntropyLoss(weight=self.weights, size_average=False, reduce=False)

    def forward(self, inp, target):
        return self.loss(inp, target)


"""==================================================="""
def gimme_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


"""==================================================="""
def save_graph(network_output, savepath, savename):
    from graphviz import Digraph
    def make_dot(var, savename, params=None):
        """ Produces Graphviz representation of PyTorch autograd graph
        Blue nodes are the Variables that require grad, orange are Tensors
        saved for backward in torch.autograd.Function
        Args:
            var: output Variable
            params: dict of (name, Variable) to add names to node that
                require grad (TODO: make optional)
        """
        if params is not None:
            assert all(isinstance(p, Variable) for p in params.values())
            param_map = {id(v): k for k, v in params.items()}

        node_attr = dict(style='filled',shape='box',align='left',fontsize='12',ranksep='0.1',height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
        seen = set()

        def size_to_str(size):
            return '('+(', ').join(['%d' % v for v in size])+')'

        def add_nodes(var):
            if var not in seen:
                if torch.is_tensor(var):
                    dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
                elif hasattr(var, 'variable'):
                    u = var.variable
                    name = param_map[id(u)] if params is not None else ''
                    node_name = '%s\n %s' % (name, size_to_str(u.size()))
                    dot.node(str(id(var)), node_name, fillcolor='lightblue')
                else:
                    dot.node(str(id(var)), str(type(var).__name__))
                seen.add(var)
                if hasattr(var, 'next_functions'):
                    for u in var.next_functions:
                        if u[0] is not None:
                            dot.edge(str(id(u[0])), str(id(var)))
                            add_nodes(u[0])
                if hasattr(var, 'saved_tensors'):
                    for t in var.saved_tensors:
                        dot.edge(str(id(t)), str(id(var)))
                        add_nodes(t)

        add_nodes(var.grad_fn)
        print("Saving...")
        dot.save(savename)
        return dot

    if not os.path.exists(savepath+"/Network_Graphs"):
        os.makedirs(savepath+"/Network_Graphs")
    viz_graph = make_dot(network_output, savepath+"/Network_Graphs"+"/"+savename)
    print("Creating pdf...")
    viz_graph.view()
