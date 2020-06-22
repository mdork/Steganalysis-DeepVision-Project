import numpy as np, os, sys, pandas as pd, ast, time, gc
import torch, torch.nn as nn, pickle as pkl, random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm, trange
import network as net
import auxiliaries as aux
import argparse
from distutils.dir_util import copy_tree
from datetime import datetime

seed = 42
print(f'setting everything to seed {seed}')
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def trainer(network, epoch, data_loader, loss_track, optimizer, loss_func):

    _ = network.train()
    loss_track.reset()
    logits_collect = []; labels_collect = []
    data_iter = tqdm(data_loader, position=2)
    inp_string = 'Epoch {} || Loss: --- | Acc: ---'.format(epoch)
    data_iter.set_description(inp_string)

    for image_idx, file_dict in enumerate(data_iter):

        optimizer.zero_grad()
        image = file_dict["image"].type(torch.FloatTensor).cuda()
        label = file_dict["label"].type(torch.FloatTensor).cuda()

        logits = network(image)
        loss   = loss_func(logits, label)

        loss.backward()
        optimizer.step()

        prediction = np.argmax(logits.cpu().data.numpy(), axis=1)
        acc = (np.round(prediction==label.cpu().data.numpy())).mean()

        logits = logits.detach().cpu()
        logits_collect.append(logits)
        labels_collect.append(label.cpu().numpy())

        loss_dic = [loss.item(), acc]
        loss_track.append(loss_dic)

        if image_idx%20==0:
            loss_mean, acc_mean, *_ = loss_track.get_iteration_mean()
            inp_string = 'Epoch {} || Loss: {} | Acc: {}'.format(epoch, np.round(loss_mean, 2),
                                                                 np.round(acc_mean, 3))
            data_iter.set_description(inp_string)

    logits = torch.cat(logits_collect, dim=0)
    label = np.concatenate(labels_collect, axis=0)
    if logits.shape[1] ==4:
        logits = nn.Softmax(dim=1)(logits).numpy()
        pred  = np.sum(logits[:, 1:], axis=1)
    elif logits.shape[1] == 12:
        logits = nn.Softmax(dim=1)(logits).numpy()
        pred = np.sum(logits[:, 3:], axis=1)
    else:
        pred = logits.numpy().reshape(-1)

    label[label>1] = 1
    label = label.reshape(-1)
    auc = aux.auc(label.astype(int), pred)
    loss_track.append_auc(auc)

    ### Empty GPU cache
    torch.cuda.empty_cache()
    loss_track.get_mean()


def validator(network, epoch, data_loader, loss_track, loss_func, scheduler):

    _ = network.eval()
    loss_track.reset()

    logits_collect = []; labels_collect = []
    data_iter = tqdm(data_loader, position=2)
    inp_string = 'Epoch {} || Loss: --- | Acc: ---'.format(epoch)
    data_iter.set_description(inp_string)

    with torch.no_grad():
        for image_idx, file_dict in enumerate(data_iter):

            image = file_dict["image"].type(torch.FloatTensor).cuda()
            label = file_dict["label"].type(torch.FloatTensor).cuda()

            logits   = network(image)
            loss     = loss_func(logits, label)

            prediction = np.argmax(logits.cpu().data.numpy(), axis=1)
            acc = (np.round(prediction==label.cpu().data.numpy())).mean()

            logits_collect.append(logits.detach().cpu())
            labels_collect.append(label.cpu().numpy())

            loss_dic = [loss.item(), acc]

            loss_track.append(loss_dic)
            if image_idx%20==0:
                inp_string = 'Epoch {} || Loss: {} | Acc: {}'.format(epoch, np.round(loss.item(), 2), acc)
                data_iter.set_description(inp_string)

    logits = torch.cat(logits_collect, dim=0)
    label = np.concatenate(labels_collect, axis=0)

    if logits.shape[1] ==4:
        logits = nn.Softmax(dim=1)(logits).numpy()
        pred  = np.sum(logits[:, 1:], axis=1)
    elif logits.shape[1] == 12:
        logits = nn.Softmax(dim=1)(logits).numpy()
        pred = np.sum(logits[:, 3:], axis=1)
    else:
        pred = logits.numpy().reshape(-1)

    label[label>1] = 1
    label = label.reshape(-1)
    auc = aux.auc(label.astype(int), pred)
    loss_track.append_auc(auc)

    ### Empty GPU cache
    torch.cuda.empty_cache()
    loss_track.get_mean()
    scheduler.step(loss_track.get_current_mean()[0])


def main(opt):
    """============================================"""
    ### Load Network
    network = net.Net(opt.Network).cuda()
    print("Number of parameters in model", sum(p.numel() for p in network.parameters()))

    ###### Define Optimizer ######
    loss_func   = aux.Base_Loss(opt.Network)
    optimizer   = torch.optim.AdamW(network.parameters(), lr=opt.Training['lr'], weight_decay=opt.Training['weight_decay'])
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, min_lr=1e-7)

    ###### Create Dataloaders ######
    train_dataset     = aux.dataset(opt, mode='train')
    train_data_loader = torch.utils.data.DataLoader(train_dataset, num_workers=opt.Training['workers'],
                                                    batch_size=opt.Training['bs'], shuffle=True)
    val_dataset       = aux.dataset(opt, mode='evaluation')
    val_data_loader   = torch.utils.data.DataLoader(val_dataset, num_workers=opt.Training['workers'],
                                                    batch_size=opt.Training['bs'], shuffle=False)

    ###### Set Logging Files ######
    dt = datetime.now()
    dt = '{}-{}-{}-{}-{}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
    opt.Training['name'] = 'Model' + '_Date-' + dt  # +str(opt.iter_idx)+
    if opt.Training['savename'] != "":
        opt.Training['name'] += '_' + opt.Training['savename']

    save_path = opt.Paths['save_path'] + "/" + opt.Training['name']

    # Make the saving directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        count = 1
        while os.path.exists(save_path):
            count += 1
            svn = opt.Training['name'] + "_" + str(count)
            save_path = opt.Paths['save_path'] + "/" + svn
        opt.Training['name'] = svn
        os.makedirs(save_path)
    opt.Paths['save_path'] = save_path

    def make_folder(name):
        if not os.path.exists(name):
            os.makedirs(name)

    # Make summary plots, images, segmentation and videos folder
    save_summary = save_path + '/summary_plots'
    make_folder(save_path + '/summary_plots')

    ### Copy Code !!
    copy_tree('./', save_path + '/code/')
    save_str = aux.gimme_save_string(opt)

    ### Save rudimentary info parameters to text-file and pkl.
    with open(opt.Paths['save_path'] + '/Parameter_Info.txt', 'w') as f:
        f.write(save_str)
    pkl.dump(opt, open(opt.Paths['save_path'] + "/hypa.pkl", "wb"))

    logging_keys = ["Loss", "ACC", "AUC"]

    loss_track_train = aux.loss_tracking(logging_keys)
    loss_track_test = aux.loss_tracking(logging_keys)

    ### Setting up CSV writers
    full_log_train = aux.CSVlogger(save_path + "/log_per_epoch_train.csv", ["Epoch", "Time", "LR"] + logging_keys)
    full_log_test = aux.CSVlogger(save_path + "/log_per_epoch_test.csv", ["Epoch", "Time", "LR"] + logging_keys)

    epoch_iterator = tqdm(range(0, opt.Training['n_epochs']), ascii=True, position=1)
    best_val_acc   = 0

    for epoch in epoch_iterator:
        epoch_time = time.time()

        ###### Training ########
        epoch_iterator.set_description("Training with lr={}".format(np.round([group['lr'] for group in optimizer.param_groups][0], 6)))
        trainer(network, epoch, train_data_loader, loss_track_train, optimizer, loss_func)

        ###### Validation #########
        epoch_iterator.set_description('Validating...')
        validator(network, epoch, val_data_loader, loss_track_test, loss_func, scheduler)

        ###### SAVE CHECKPOINTS ########
        save_dict = {'epoch': epoch+1, 'state_dict': network.state_dict(),
                     'optim_state_dict': optimizer.state_dict()}

        # Best Validation Score
        current_auc = loss_track_test.get_current_mean()[-1]
        if current_auc > best_val_acc:
            torch.save(save_dict, opt.Paths['save_path'] + '/checkpoint_best_val.pth.tar')
            best_val_acc = current_auc

        ###### Logging Epoch Data ######
        full_log_train.write([epoch, time.time() - epoch_time, [group['lr'] for group in optimizer.param_groups][0], *loss_track_train.get_current_mean()])
        full_log_test.write([epoch, time.time() - epoch_time, [group['lr'] for group in optimizer.param_groups][0], *loss_track_test.get_current_mean()])

        ###### Generating Summary Plots #######
        # aux.summary_plots(loss_track_train.get_hist(), loss_track_test.get_hist(), epoch, save_summary)
        _ = gc.collect()


### Start Training ###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--config", type=str, default='/export/home/mdorkenw/code/ALASKA2/network_base_setup.txt',
                        help="Define config file")
    args = parser.parse_args()
    training_setups = aux.extract_setup_info(args.config)

    # find all the gpus we want to use now (currently not necessary, only if we want to run different
    # training setting on different gpus at the same time)
    gpus = []
    for tr in training_setups:
        for GPU in tr.Training['GPU']:
            gpus.append(str(GPU))

    gpus = ",".join(gpus)

    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    for training_setup in tqdm(training_setups, desc='Training Setups... ', position=0, ascii=True):
        main(training_setup)
