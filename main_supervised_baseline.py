# encoding=utf-8
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from models.backbones import *
from models.loss import *
from models.scatterWave import *
from models.WaveletNet import *
from models.ModernTCN import *
from trainer import *
import torch
import torch.nn as nn
import argparse
from datetime import datetime
import pickle
import numpy as np
import os
import logging
import sys
from data_preprocess.data_preprocess_utils import normalize
from scipy import signal
from copy import deepcopy
import fitlog
from utils import tsne, mds, _logger
from new_augmentations import vanilla_mixup_sup, cutmix_sup
# fitlog.debug()


parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--cuda', default=0, type=int, help='cuda device ID, 0/1')
# hyperparameter
parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=60, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--lr_cls', type=float, default=1e-3, help='learning rate for linear classifier')
parser.add_argument('--VAE', action='store_true')
parser.add_argument('--VanillaMixup', action='store_true')
parser.add_argument('--BinaryMix', action='store_true')
parser.add_argument('--Cutmix', action='store_true')
parser.add_argument('--Magmix', action='store_true')
parser.add_argument('--phase_shift', action='store_true')
parser.add_argument('--MSE', action='store_true')
parser.add_argument('--robust_check', action='store_true')
parser.add_argument('--controller', action='store_true')
parser.add_argument('--random_aug', action='store_true')
#
parser.add_argument('--blur', action='store_true')
parser.add_argument('--aps', action='store_true')
# dataset
parser.add_argument('--dataset', type=str, default='ucihar', choices=['physio', 'ucihar', 'hhar', 'usc', 'ieee_small','ieee_big', 'dalia', 'chapman', 'clemson', 'sleep'], help='name of dataset')
parser.add_argument('--n_feature', type=int, default=77, help='name of feature dimension')
parser.add_argument('--len_sw', type=int, default=30, help='length of sliding window')
parser.add_argument('--n_class', type=int, default=18, help='number of class')
parser.add_argument('--cases', type=str, default='subject_val', choices=['random', 'subject', 'subject_large', 'cross_device', 'joint_device'], help='name of scenarios')
parser.add_argument('--split_ratio', type=float, default=0.2, help='split ratio of test/val: train(0.64), val(0.16), test(0.2)')
parser.add_argument('--target_domain', type=str, default='0', help='the target domain, [0 to 29] for ucihar, '
                                                                   '[1,2,3,5,6,9,11,13,14,15,16,17,19,20,21,22,23,24,25,29] for shar, '
                                                                   '[a-i] for hhar')

# backbone model
parser.add_argument('--backbone', type=str, default='DCL', choices=['FCN', 'FCN_b', 'DCL', 'LSTM', 'Transformer', 'resnet', 'WaveletNet', 'ModernTCN'], help='name of framework')
# model paramters
parser.add_argument('--block', type=int, default=3, help='number of groups')
parser.add_argument('--stride', type=int, default=2, help='stride')

# log
parser.add_argument('--logdir', type=str, default='log/', help='log directory')

# AE & CNN_AE
parser.add_argument('--lambda1', type=float, default=1.0, help='weight for reconstruction loss when backbone in [AE, CNN_AE]')


# python main_supervised_baseline.py --dataset 'ieee_small' --backbone 'resnet' --lr 5e-4 --n_epoch 999 --cuda 0 --phase_shift

# hhar
parser.add_argument('--device', type=str, default='Phones', choices=['Phones', 'Watch'], help='data of which device to use (random case); data of which device to be used as training data (cross-device case, data from the other device as test data)')


############### Parser done ################


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    if args.VAE:
        loss = (lam * nn.CrossEntropyLoss(reduction='none')(pred, y_a) + (1 - lam) * nn.CrossEntropyLoss(reduction='none')(pred, y_b)).sum()/y_a.size(0)
    else:
        loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    return loss

def adjust_learning_rate(optimizer,  epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = BASE_lr * (0.5 ** (epoch // 30))
    lr = 0.003 * (0.95)**epoch
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args, train_loaders, val_loader, model, DEVICE, criterion, save_dir='results/', model_c=None):
    
    if model_c is not None: 
        optimizer_model_c = optim.Adam(model_c.parameters(), lr=5e-4)
        optimizer_model = optim.Adam(list(model.parameters()) + list(model_c.parameters()), lr=args.lr)
    else:
        parameters = model.parameters()
        optimizer_model = torch.optim.Adam(parameters, args.lr, weight_decay=args.weight_decay)

    min_val_loss, counter = 1e8, 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_model, mode='min', patience=15, factor=0.5, min_lr=1e-7, verbose=False)
    for epoch in range(args.n_epoch):
        #logger.debug(f'\nEpoch : {epoch}')
        train_loss, n_batches, total, correct = 0, 0, 0, 0
        if args.backbone == 'WaveletNet':
            wave_loss = WaveletLoss(weight_loss=1.)
        model.train()
        if model_c is not None: model_c.train()
        for loader_idx, train_loader in enumerate(train_loaders):
            for idx, (sample, target, domain) in enumerate(train_loader):
                n_batches += 1

                if args.controller: 
                    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
                    # sample_framed = constant_phase_shift(sample, args, DEVICE)
                    ref_frame = model_c(torch.abs(fftsamples).to(DEVICE).float())
                    # ref_frame = model_c(sample_framed.to(DEVICE).float())
                    loss_c = torch.std(ref_frame)
                    sample_c = frame_transform(sample, fftsamples, ref_frame, args, DEVICE)

                if args.phase_shift:
                    sample = constant_phase_shift(sample, args, DEVICE)
                elif args.controller:
                    sample = sample_c
                
                if args.random_aug: sample, all_shifts = random_time_shift(sample)

                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()


                if args.backbone[-2:] == 'AE':
                    out, x_decoded = model(sample)
                else:
                    out, _ = model(sample)

                if args.MSE == False:
                    loss = criterion(out, target)
                else:
                    loss = criterion(out.squeeze(), target.float())

                # loss = criterion(out, target) if args.MSE == False else criterion(out.squeeze(), target.float())

                if args.backbone == 'WaveletNet':
                    loss = loss + wave_loss(model)

                train_loss += loss.item()
                optimizer_model.zero_grad()

                if args.controller: 
                    optimizer_model_c.zero_grad()
                    loss_c.backward(retain_graph=True)

                loss.backward()
                optimizer_model.step()

                if args.controller: 
                    optimizer_model_c.step()

        if val_loader is None:
            best_model = deepcopy(model.state_dict())
            model_dir = save_dir + args.model_name + '.pt'
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, model_dir)
        else:
            with torch.no_grad():
                model.eval()
                if args.controller: model_c.eval()
                val_loss, n_batches, total, correct = 0, 0, 0, 0
                for idx, (sample, target, domain) in enumerate(val_loader):
                    n_batches += 1

                    if args.controller: 
                        fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
                        # sample_framed = constant_phase_shift(sample, args, DEVICE)
                        ref_frame = model_c(torch.abs(fftsamples).to(DEVICE).float())
                        # ref_frame = model_c(sample_framed.to(DEVICE).float())
                        sample_c = frame_transform(sample, fftsamples, ref_frame, args, DEVICE)

                    if args.phase_shift:
                        sample = constant_phase_shift(sample, args, DEVICE)
                    elif args.controller:
                        sample = sample_c

                    sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()

                    if args.backbone[-2:] == 'AE':
                        out, x_decoded = model(sample)
                    else:
                        out, _ = model(sample)

                    if args.MSE == False:
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out.squeeze(), target.float())

                    # loss = criterion(out, target) if args.MSE == False else criterion(out.squeeze(), target.float())

                    if args.backbone[-2:] == 'AE': 
                        loss += nn.MSELoss()(sample, x_decoded) * args.lambda1
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(out.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum()

                if val_loss <= min_val_loss:
                    min_val_loss = val_loss
                    best_model = deepcopy(model.state_dict())

                    model_dir = save_dir + args.model_name + '.pt' if args.phase_shift == False else save_dir + args.model_name + '_phase_shift.pt'
                    torch.save(model.state_dict(), model_dir)
                    if args.controller:
                        model_dir = save_dir + args.model_name + '_controller.pt'
                        torch.save(model_c.state_dict(), model_dir)
                else:
                    counter += 1
                    if counter > 90: 
                        return best_model
                scheduler.step(val_loss)
    return best_model

def test(test_loader, model, DEVICE, criterion, plot=False, model_c=None):
    if args.adversary_robust: atk = PGD(model, eps=args.eps, alpha=args.eps/5, steps=10, device=DEVICE)
    # with torch.no_grad():
    model.eval()
    if model_c is not None: 
        model_c.eval()
        assigned_phase = np.array([])
    total_loss, final_const, n_batches, total, correct = 0, 0, 0, 0, 0
    feats, prds, trgs, cnst = None, None, None, None
    otp, confusion_matrix = np.array([]), torch.zeros(args.n_class, args.n_class)
    for idx, (sample, target, domain) in enumerate(test_loader):
        n_batches += 1
        B = sample.shape[0]
        if args.robust_check: # Robustness to time shift
            # continous_shift_evaluate(sample, model, DEVICE, target)
            sample_shifted, all_shifts = random_time_shift(sample)
            sample_shifted_2, _ = random_time_shift(sample)
            sample = torch.cat((sample, sample_shifted_2), 0)
            target = torch.cat((target, target), 0)
        
        if args.controller: 
            fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
            # sample_framed = constant_phase_shift(sample, args, DEVICE)
            ref_frame = model_c(torch.abs(fftsamples).to(DEVICE).float())
            # ref_frame = model_c(sample_framed.to(DEVICE).float())
            sample_c = frame_transform(sample, fftsamples, ref_frame, args, DEVICE)
            assigned_phase = np.concatenate((assigned_phase, ref_frame.detach().cpu().numpy())) if assigned_phase.size != 0 else ref_frame.detach().cpu().numpy()

        if args.phase_shift:
            sample = constant_phase_shift(sample, args, DEVICE)
        elif args.controller:
            sample = sample_c

        sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
        # import pdb;pdb.set_trace();
        out, _ = model(sample)
        out = out.detach()
        if args.MSE == False:
            loss = criterion(out, target)
        else:
            loss = criterion(out.squeeze(), target.float())
        # loss = criterion(out, target) if args.MSE == False else criterion(out.squeeze(), target.float())

        total_loss += loss.item()
        _, predicted = torch.max(out.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum()
        otp = np.vstack((otp, out.data.cpu().numpy())) if otp.size != 0 else out.data.cpu().numpy()

        if prds is None:
            prds = predicted
            trgs = target
            const = predicted[0:B] - predicted[B:] if args.robust_check else None
        else:
            prds = torch.cat((prds, predicted))
            trgs = torch.cat((trgs, target))
            const = torch.cat((const, predicted[0:B] - predicted[B:])) if args.robust_check else None

    acc_test = float(correct) * 100.0 / total
    maF = f1_score(trgs.cpu().numpy(), prds.cpu().numpy(), average='weighted') * 100
    # import pdb;pdb.set_trace();
    correlation = f1_score(trgs.cpu().numpy(), prds.cpu().numpy(), average='macro') * 100
    if args.dataset == 'ieee_small' or args.dataset =='ieee_big' or args.dataset == 'dalia':
        acc_test = np.sqrt(torch.mean(((trgs-prds)**2).float()).cpu())
        maF = torch.mean((torch.abs(trgs-prds)).float()).cpu()
        correlation = np.corrcoef(trgs.cpu(), prds.cpu())[0,1]
        if np.isnan(correlation): correlation = 0
        if args.robust_check: 
            final_const = 100-100*sum((const != 0)).item()/(const.shape[0])
            print(f'Consistency : {100-100*sum((const != 0)).item()/(const.shape[0]):.2f}')
    elif args.dataset == 'chapman' or args.dataset == 'physio':
        otp1 =  softmax(otp,axis=1)
        maF = roc_auc_score(trgs.cpu(), otp1, multi_class='ovo')
        correlation = f1_score(trgs.cpu().numpy(), prds.cpu().numpy(), average='macro') * 100
        if args.robust_check:
            final_const = 100-100*sum((const != 0)).item()/(const.shape[0])
            print(f'Consistency : {100-100*sum((const != 0)).item()/(const.shape[0]):.2f}')
    elif args.dataset == 'clemson': 
        trgs, prds = trgs + 29, prds + 29
        acc_test = 100*torch.mean(torch.abs((trgs-prds)/trgs)).cpu()
        maF = torch.mean((torch.abs(trgs-prds)).float()).cpu()
        correlation = 1
        if args.robust_check: 
            final_const = 100-100*sum((const != 0)).item()/(const.shape[0])
            print(f'Consistency : {100-100*sum((const != 0)).item()/(const.shape[0]):.2f}')
    elif args.dataset == 'sleep':
        # maF = f1_score(trgs.cpu().numpy(), prds.cpu().numpy(), average='macro') * 100
        correlation = cohen_kappa_score(trgs.cpu().numpy(), prds.cpu().numpy())
        if args.robust_check:
            final_const = 100-100*sum((const != 0)).item()/(const.shape[0])
            print(f'Consistency : {100-100*sum((const != 0)).item()/(const.shape[0]):.2f}')
    else:
        if args.robust_check: 
            final_const = 100-100*sum((const != 0)).item()/(const.shape[0])
            print(f'Consistency : {100-100*sum((const != 0)).item()/(const.shape[0]):.2f}')
    if plot == True:
        tsne(feats, trgs, domain=None, save_dir=plot_dir_name + args.model_name + '_tsne.png')
        mds(feats, trgs, domain=None, save_dir=plot_dir_name + args.model_name + 'mds.png')
        sns_plot = sns.heatmap(confusion_matrix, cmap='Blues', annot=True)
        sns_plot.get_figure().savefig(plot_dir_name + args.model_name + '_confmatrix.png')
    return acc_test, maF, correlation, final_const

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_sup(args):
    train_loaders, val_loader, test_loader = setup_dataloaders(args)
    
    if args.MSE: args.n_class = 1

    if args.backbone == 'FCN':
        if args.blur:
            model = FCN_blur(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)
        elif args.aps:
            model = FCN_aps(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)
        else:
            model = FCN(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)
    elif args.backbone == 'FCN_b':
        model = FCN_big(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)
    elif args.backbone == 'DCL':
        model = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=False)
    elif args.backbone == 'LSTM':
        model = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=False)
    elif args.backbone == 'AE':
        model = AE(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, outdim=128, backbone=False)
    elif args.backbone == 'CNN_AE':
        model = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, backbone=False)
    elif args.backbone == 'Transformer':
        model = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=False)
    elif args.backbone == 'WaveletNet':
        model = WaveletNet(args=args)  
    elif args.backbone == 'ModernTCN':
        model = ModernTCN(args=args, class_num=args.n_class, seq_len=args.len_sw)
    elif args.backbone == 'resnet':
        if args.blur:
            model = ResNet1D_blur(in_channels=args.n_feature, base_filters=32, kernel_size=5, stride=args.stride, groups=1, n_block=args.block, n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4, backbone=False)
        elif args.aps:
            model = ResNet1D_aps(in_channels=args.n_feature, base_filters=32, kernel_size=5, stride=args.stride, groups=1, n_block=args.block, n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4, backbone=False)
        else:
            model = ResNet1D(in_channels=args.n_feature, base_filters=32, kernel_size=5, stride=args.stride, groups=1, n_block=args.block, n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4, backbone=False)
    else:
        NotImplementedError

    if args.controller: 
        model_c = FCN_controller(n_channels=args.n_feature, args=args)
        model_c = model_c.to(DEVICE)
    else: model_c = None

    model = model.to(DEVICE)
    if args.target_domain == '17' or args.target_domain == 'a' or args.target_domain == '10' or args.target_domain == '0': 
        print('Number of parameters: ', sum(p.numel() for p in model.parameters()))
    # import pdb;pdb.set_trace();
    args.model_name = args.backbone + '_'+args.dataset + '_cuda' + str(args.cuda) + '_bs' + str(args.batch_size) + '_sw' + str(args.len_sw)

    save_dir = 'results/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # log
    if os.path.isdir(args.logdir) == False:
        os.makedirs(args.logdir)

    criterion = nn.CrossEntropyLoss() if not args.MSE and not args.dataset == 'ptb' else torch.nn.BCEWithLogitsLoss()

    best_model = train(args, train_loaders, val_loader, model, DEVICE, criterion, model_c=model_c)

    if args.backbone == 'FCN':
        if args.blur:
            model_test = FCN_blur(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)
        elif args.aps:
            model_test = FCN_aps(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)
        else:
            model_test = FCN(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)
    if args.backbone == 'FCN_b':
        model_test = FCN_big(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)
    elif args.backbone == 'DCL':
        model_test = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=False)
    elif args.backbone == 'LSTM':
        model_test = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=False)
    elif args.backbone == 'AE':
        model_test = AE(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, outdim=128, backbone=False)
    elif args.backbone == 'CNN_AE':
        model_test = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, backbone=False)
    elif args.backbone == 'Transformer':
        model_test = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=False)
    elif args.backbone == 'WaveletNet':
        model_test = WaveletNet(args=args)    
    elif args.backbone == 'ModernTCN':
        model_test = ModernTCN(args=args, class_num=args.n_class, seq_len=args.len_sw)
    elif args.backbone == 'resnet':
        if args.blur:
            model_test = ResNet1D_blur(in_channels=args.n_feature, base_filters=32, kernel_size=5, stride=args.stride, groups=1, n_block=args.block, n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4, backbone=False)
        elif args.aps:
            model_test = ResNet1D_aps(in_channels=args.n_feature, base_filters=32, kernel_size=5, stride=args.stride, groups=1, n_block=args.block, n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4, backbone=False)
        else:
            model_test = ResNet1D(in_channels=args.n_feature, base_filters=32, kernel_size=5, stride=args.stride, groups=1, n_block=args.block, n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4, backbone=False)
        # model_test = ResNet1D(in_channels=args.n_feature, base_filters=32, kernel_size=6, stride=2, groups=1, n_block=2, n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4)
    else:
        NotImplementedError

    if args.controller:
        model_c = FCN_controller(n_channels=args.n_feature, args=args)
        model_dir = save_dir + args.model_name + '_controller.pt'
        model_c.load_state_dict(torch.load(model_dir))
        model_c = model_c.to(DEVICE)

    model_dir = save_dir + args.model_name + '.pt' if args.phase_shift == False else save_dir + args.model_name + '_phase_shift.pt'
    model_test.load_state_dict(torch.load(model_dir))
    model_test = model_test.to(DEVICE)

    if args.controller:
        acc, mf1, correlation, const = test(test_loader, model_test, DEVICE, criterion, plot=False, model_c=model_c)
    else:    
        acc, mf1, correlation, const = test(test_loader, model_test, DEVICE, criterion, plot=False)

    return acc, mf1, correlation, const
    #training_end = datetime.now()
    #training_time = training_end - training_start
    #logger.debug(f"Training time is : {training_time}")


def set_domains(args):
    args = parser.parse_args()
    if args.dataset == 'usc':
        domain = [10, 11, 12, 13]        
    elif args.dataset == 'ucihar':
        domain = [0, 1, 2, 3, 4]
    elif args.dataset == 'ieee_small':
        domain = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    elif args.dataset == 'ieee_big':
        domain = [17, 18, 19, 20, 21]
    elif args.dataset == 'dalia':
        domain = [0, 1, 2, 3, 4]
    elif args.dataset == 'hhar':
        domain = ['a', 'b', 'c', 'd']
    elif args.dataset == 'clemson':
        domain = [i for i in range(0, 10)]
    elif args.dataset == 'chapman' or args.dataset == 'physio' or args.dataset == 'sleep':
        domain = [0]
    return domain

if __name__ == '__main__':
    args = parser.parse_args()
    domain = set_domains(args)
    all_metrics = []
    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print('device:', DEVICE, 'dataset:', args.dataset)
    for i in range(3):
        set_seed(i*10+1)
        print(f'Training for seed {i}')
        seed_metric = []
        for k in domain:
            setattr(args, 'target_domain', str(k))
            setattr(args, 'save', args.dataset + str(k))
            setattr(args, 'cases', 'subject_val')
            # if args.dataset == 'hhar':
            #     setattr(args, 'cases', 'subject')
            # else:
            #     setattr(args, 'cases', 'subject_large')
            mif,maf,mac, const = train_sup(args)
            seed_metric.append([mif,maf,mac,const])
        seed_metric = np.array(seed_metric)
        all_metrics.append([np.mean(seed_metric[:,0]), np.mean(seed_metric[:,1]), np.mean(seed_metric[:,2]), np.mean(seed_metric[:,3])])
    values = np.array(all_metrics)
    mean = np.mean(values,0)
    std = np.std(values,0)
    print('M1: {:.3f}, M2: {:.4f}, M3: {:.4f}'.format(mean[0], mean[1], mean[2]))
    print('Std1: {:.3f}, Std2: {:.4f}, Std3: {:.4f}'.format(std[0], std[1], std[2]))
    if args.robust_check: print('Mean consistency: {:.4f}, Std consistency: {:.4f}'.format(mean[3], std[3]))