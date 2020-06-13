import numpy as np, os, sys, pandas as pd, ast, imp, time, gc
import torch, torch.nn as nn, pickle as pkl
from PIL import Image
from torchvision import transforms
from tqdm import tqdm, trange
import network as net
import auxiliaries as aux
import argparse
from datetime import datetime


"""=================================================================================="""
"""=================================================================================="""
def trainer(network, epoch, data_loader, Metrics, optimizer, loss_func):

    _ = network.train()

    start_time = time.time()

    epoch_coll_acc, epoch_coll_loss  = [],[]

    data_iter = tqdm(data_loader, position=2)
    inp_string = 'Epoch {} || Loss: --- | Acc: ---'.format(epoch)
    data_iter.set_description(inp_string)

    for image_idx, file_dict in enumerate(data_iter):

        iter_start_time = time.time()

        input_image     = torch.autograd.Variable(file_dict["Image"]).type(torch.FloatTensor).cuda()
        target          = torch.autograd.Variable(file_dict["label"]).type(torch.FloatTensor).cuda()

        #--- Run Training ---
        prediction = network(input_image)

        ### BASE LOSS
        feed_dict = {'inp':prediction, 'target':target}
        loss      = loss_func(**feed_dict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        acc = (np.round(prediction.cpu().data.numpy())==target.cpu().data.numpy()).reshape(-1)

        #--- Get Scores ---
        epoch_coll_acc.extend(list(acc))
        epoch_coll_loss.append(loss.cpu().data.numpy()[0])

        if image_idx%opt.Training['verbose_idx']==0 and image_idx:
            inp_string = 'Epoch {} || Loss: {} | Acc: {}'.format(epoch,
                                                                 np.round(np.mean(epoch_coll_loss),4),
                                                                 np.round(np.sum(epoch_coll_acc)/len(epoch_coll_acc),4))
            data_iter.set_description(inp_string)

    ### Empty GPU cache
    torch.cuda.empty_cache()

    Metrics['Train Loss'].append(np.round(np.mean(epoch_coll_loss),4))
    Metrics['Train Acc'].append(np.round(np.sum(epoch_coll_acc)/len(epoch_coll_acc),4))

"""=================================================================================="""
"""=================================================================================="""
def validator(network, epoch, data_loader, Metrics, loss_func):

    _ = network.eval()

    start_time = time.time()

    epoch_coll_acc, epoch_coll_loss  = [],[]

    data_iter = tqdm(data_loader, position=2)
    inp_string = 'Epoch {} || Loss: --- | Acc: ---'.format(epoch)
    data_iter.set_description(inp_string)

    for image_idx, file_dict in enumerate(data_iter):

        iter_start_time = time.time()

        input_image     = torch.autograd.Variable(file_dict["Image"]).type(torch.FloatTensor).cuda()
        target          = torch.autograd.Variable(file_dict["label"]).type(torch.FloatTensor).cuda()

        #--- Run Training ---
        #NOTE: Test Triplet, Margin and BCE loss
        prediction = network(input_image)

        ### BASE LOSS
        feed_dict = {'inp':prediction, 'target':target}
        loss      = loss_func(**feed_dict)


        acc = (np.round(prediction.cpu().data.numpy())==target.cpu().data.numpy()).reshape(-1)

        #--- Get Scores ---
        epoch_coll_acc.extend(list(acc))
        epoch_coll_loss.append(loss.cpu().data.numpy()[0])

        if image_idx%opt.Training['verbose_idx']==0 and image_idx:
            inp_string = 'Epoch {} || Loss: {} | Acc: {}'.format(epoch,
                                                                 np.round(np.mean(epoch_coll_loss),4),
                                                                 np.round(np.sum(epoch_coll_acc)/len(epoch_coll_acc),4))
            data_iter.set_description(inp_string)

    ### Empty GPU cache
    torch.cuda.empty_cache()

    Metrics['Val Loss'].append(np.round(np.mean(epoch_coll_loss),4))
    Metrics['Val Acc'].append(np.round(np.sum(epoch_coll_acc)/len(epoch_coll_acc),4))


"""=================================================================================="""
"""=================================================================================="""
def main(opt):
    """============================================"""
    ### Load Network
    imp.reload(net)
    network = net.Extract_Scaffold(opt)
    network.n_params = aux.gimme_params(network)
    _ = network.cuda()

    ### Set Optimizer
    # Base_Loss   = aux.Base_Loss(weights=torch.autograd.Variable(torch.ones(opt.Network['n_classes'])))
    loss_func   = aux.Base_Loss()
    optimizer   = torch.optim.Adam(network.parameters(), lr=opt.Training['lr'], weight_decay=opt.Training['weight_decay'])
    scheduler   = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.Training['step_size'], gamma=opt.Training['gamma'])

    """============================================"""
    ### Set Dataloader
    train_dataset     = aux.dataset(opt, mode='train')
    train_data_loader = torch.utils.data.DataLoader(train_dataset, num_workers = opt.Training['kernels'],
                                                    batch_size = opt.Training['bs'], shuffle=True)
    val_dataset       = aux.dataset(opt, mode='val')
    val_data_loader   = torch.utils.data.DataLoader(val_dataset,   num_workers = opt.Training['kernels'],
                                                    batch_size = opt.Training['bs'], shuffle=False)


    """============================================"""
    ### Set Logging Files ###
    dt = datetime.now()
    dt = '{}-{}-{}-{}-{}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
    opt.Training['name'] = 'Extract-network_Iter-'+str(opt.iter_idx)+'_Date-'+dt
    if opt.Training['savename']!="":
        opt.Training['name'] += '_'+opt.Training['savename']

    save_path = opt.Paths['save_path']+"/"+opt.Training['name']

    #Make the saving directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        count = 1
        while os.path.exists(save_path):
            count        += 1
            svn          = opt.name+"_"+str(count)
            save_path    = opt.save_path+"/"+svn
        opt.name = svn
        os.makedirs(save_path)
    opt.Paths['save_path'] = save_path

    #Generate save string
    save_str = aux.gimme_save_string(opt)

    ### Save rudimentary info parameters to text-file and pkl.
    with open(opt.Paths['save_path']+'/Parameter_Info.txt','w') as f:
        f.write(save_str)
    pkl.dump(opt,open(opt.Paths['save_path']+"/hypa.pkl","wb"))


    """============================================"""
    logging_keys    = ["Train Acc", "Train Loss", "Val Acc", "Val Loss"]
    Metrics         = {key:[] for key in logging_keys}
    Timer           = {key:[] for key in ["Train","Val"]}

    ### Setting up CSV writers
    full_log  = aux.CSVlogger(save_path+"/log_per_epoch.csv", ["Epoch", "Time", "Training Loss", "Training Acc", "Validation Acc", "Learning Rate"])


    """============================================="""
    epoch_iterator = tqdm(range(opt.Training['n_epochs']),position=1)
    best_val_acc   = 0

    for epoch in epoch_iterator:
        epoch_time = time.time()

        scheduler.step()

        ###### Training ########
        epoch_iterator.set_description("Training with lr={}".format(np.round(scheduler.get_lr(),8)))
        trainer(network, epoch, train_data_loader, Metrics, optimizer, loss_func)

        ###### Validation #########
        epoch_iterator.set_description('Validating...')
        validator(network, epoch, val_data_loader, Metrics, loss_func)


        ###### SAVE CHECKPOINTS ########
        save_dict = {'epoch': epoch+1, 'state_dict':network.state_dict(),
                     'optim_state_dict':optimizer.state_dict()}
        # Best Validation Score
        if Metrics['Val Acc'][-1]>best_val_acc:
            torch.save(save_dict, opt.Paths['save_path']+'/checkpoint_best_val.pth.tar')
            best_val_acc = Metrics['Val Acc'][-1]

        # After Epoch
        torch.save(save_dict, opt.Paths['save_path']+'/checkpoint.pth.tar')

        ###### Logging Epoch Data ######
        full_log.write([epoch, time.time()-epoch_time, np.mean(Metrics["Train Loss"][epoch]),np.mean(Metrics["Train Acc"][epoch]), np.mean(Metrics["Val Acc"][epoch]), scheduler.get_lr()[0]])


        ###### Generating Summary Plots #######
        sum_title = 'Max Train Acc: {0:2.3f} | Max Val Acc: {1:2.3f}'.format(np.max([np.mean(Metrics["Train Acc"][ep]) for ep in range(epoch+1)]),
                                                                             np.max([np.mean(Metrics["Val Acc"][ep]) for ep in range(epoch+1)]))
        aux.progress_plotter(np.arange(epoch+1), \
                             [np.mean(Metrics["Train Loss"][ep]) for ep in range(epoch+1)],[np.mean(Metrics["Train Acc"][ep]) for ep in range(epoch+1)],
                             [np.mean(Metrics["Val Loss"][ep]) for ep in range(epoch+1)], [np.mean(Metrics["Val Acc"][ep]) for ep in range(epoch+1)], save_path+'/training_results.png', sum_title)


        _ = gc.collect()



### Start Training ###
if __name__ == '__main__':
    """============================================"""
    ### GET TRAINING SETUPs ###
    #Read network and training setup from text file.
    opt = argparse.Namespace()
    opt.Paths = {}

    ### DATA PATHS ###
    opt.Paths['train_data']                   = '/home/karsten_dl/Dropbox/Data_Dump/HPA/train'
    opt.Paths['assign_file']                  = '/home/karsten_dl/Dropbox/Data_Dump/HPA/train.csv'
    opt.Paths['save_path']                    = '/home/karsten_dl/Dropbox/Data_Dump/HPA/Logs'
    opt.Paths['network_base_setup_file']      = '/home/karsten_dl/Dropbox/Projects/current_projects/hpa-kaggle/Train/network_base_setup.txt'
    opt.Paths['network_variation_setup_file'] = '/home/karsten_dl/Dropbox/Projects/current_projects/hpa-kaggle/Train/network_variation_setup.txt'

    training_setups = aux.extract_setup_info(opt)
    for training_setup in tqdm(training_setups, desc='Training Setups... ', position=0):
        main(training_setup)
