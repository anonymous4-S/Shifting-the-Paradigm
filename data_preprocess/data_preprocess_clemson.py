import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import random
import scipy.io
import pickle as cp
from data_preprocess.base_loader import base_loader


def load_domain_data(domain_idx):
    str_folder = 'data/'
    data_all = scipy.io.loadmat(str_folder + 'clemson.mat')
    data_all = data_all['whole_data']
    domain_idx = int(domain_idx)
    X = data_all[domain_idx,0].transpose()
    y = np.squeeze(data_all[domain_idx,1])
    return X, y

class data_loader_clemson(base_loader):
    def __init__(self, samples, labels, args):
        super(data_loader_clemson, self).__init__(samples, labels, args)

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        sample = np.reshape(sample, (sample.shape[0], 1, 1))
        target = target - 29
        return np.squeeze(np.transpose(sample, (1, 0, 2)),0), target, sample


def prep_domains_clemson_subject_large(args):
    source_domain_list = [str(i) for i in range(0, 30)]
    target_domain_list = [str(i) for i in range(args.target_domain*3, args.target_domain*3+3)]
    source_domain_list = [x for x in source_domain_list if x not in target_domain_list]
    # source domain data prep
    xtrain, xbpms = np.array([]), np.array([])
    for source_domain in source_domain_list:
        #print('source_domain:', source_domain)
        x, y = load_domain_data(source_domain)
        x = x.reshape((-1, x.shape[-1]))

        xtrain = np.concatenate((xtrain, x), axis=0) if xtrain.size else x
        xbpms = np.concatenate((xbpms, y), axis=0) if xbpms.size else y
    
    if args.augs:
        xtrain, xbpms, lin_ratio = aug_data(xtrain, xbpms, args)
    else: lin_ratio = np.ones((xtrain.shape[0], 1))

    data_set = data_loader_clemson(xtrain, xbpms, lin_ratio, args)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    # source domain data prep
    xtest, ybpms = np.array([]), np.array([])
    for target_domain in target_domain_list:
        x, y = load_domain_data(target_domain)
        x = x.reshape((-1, x.shape[-1]))

        xtest = np.concatenate((xtest, x), axis=0) if xtest.size else x
        ybpms = np.concatenate((ybpms, y), axis=0) if ybpms.size else y
    
    data_set = data_loader_clemson(xtest, ybpms, np.ones((xtest.shape[0], 1)), args)
    target_loader = DataLoader(data_set, batch_size=512, shuffle=False)  # For testing keep the batch size as 512

    return source_loader, None, target_loader

def prep_domains_clemson_subject(args):
    source_domain_list = [str(i) for i in range(0, 30)]
    target_domain_list = [str(i) for i in range(args.target_domain*3, args.target_domain*3+3)]
    source_domain_list = [x for x in source_domain_list if x not in target_domain_list]

    # source domain data prep
    xtrain, xbpms = np.array([]), np.array([])
    for source_domain in source_domain_list:
        #print('source_domain:', source_domain)
        x, y = load_domain_data(source_domain)
        x = x.reshape((-1, x.shape[-1]))

        xtrain = np.concatenate((xtrain, x), axis=0) if xtrain.size else x
        xbpms = np.concatenate((xbpms, y), axis=0) if xbpms.size else y
    
    if args.augs:
        xtrain, xbpms, lin_ratio = aug_data(xtrain, xbpms, args)
    else: lin_ratio = np.ones((xtrain.shape[0], 1))

    # Identify unique classes and their counts
    unique_classes, class_counts = np.unique(xbpms.round(), return_counts=True)

    # Determine the minimum number of samples available for any class
    min_samples_per_class = min(class_counts)

    # Create an empty list to store the balanced signals
    balanced_signals = []
    balanced_classes = []
    # Randomly sample the same number of samples from each class
    for class_label in unique_classes:
        class_indices = np.where(xbpms.round() == class_label)[0]
        random_samples = random.sample(list(class_indices), min_samples_per_class)
        balanced_signals.extend(xtrain[random_samples])
        balanced_classes.extend(xbpms[random_samples])

    # Check if additional samples are needed to balance the dataset
    additional_samples_needed = 60 - len(balanced_signals)

    # Add more random samples to balance the dataset if needed
    while additional_samples_needed > 0:
        # Randomly select a class and add a random sample from it
        random_class_label = random.choice(unique_classes)
        class_indices = np.where(xbpms == random_class_label)[0]
        random_sample_index = random.choice(class_indices)
        balanced_signals.append(xtrain[random_sample_index])
        balanced_classes.append(xbpms[random_sample_index])
        additional_samples_needed -= 1

    # Create a balanced signals array
    balanced_signals_array = np.array(balanced_signals)
    balanced_classes = np.array(balanced_classes)

    data_set = data_loader_clemson(balanced_signals_array, balanced_classes, lin_ratio, args)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    # target domain data prep
    xtest, ybpms = np.array([]), np.array([])
    for target_domain in target_domain_list:
        x, y = load_domain_data(target_domain)
        x = x.reshape((-1, x.shape[-1]))

        xtest = np.concatenate((xtest, x), axis=0) if xtest.size else x
        ybpms = np.concatenate((ybpms, y), axis=0) if ybpms.size else y

    data_set = data_loader_clemson(xtest, ybpms, np.ones((xtest.shape[0], 1)), args)
    target_loader = DataLoader(data_set, batch_size=512, shuffle=False)  # For testing keep the batch size as 512

    return source_loader, None, target_loader


def prep_domains_clemson_subject_val(args):
    source_domain_list = [i for i in range(0, 30)]
    target_domain_list = [i for i in range(int(args.target_domain)*3, int(args.target_domain)*3+3)]
    source_domain_list = [x for x in source_domain_list if x not in target_domain_list]

    val_domain_list = random.sample(source_domain_list, 3) # 1-fold for validation

    source_domain_list = [x for x in source_domain_list if x not in val_domain_list]

    source_domain_list = random.sample(source_domain_list, 6) # 2-fold for training

    # source domain data prep
    xtrain, xbpms = np.array([]), np.array([])
    for source_domain in source_domain_list:
        #print('source_domain:', source_domain)
        x, y = load_domain_data(source_domain)
        x = x.reshape((-1, x.shape[-1]))

        xtrain = np.concatenate((xtrain, x), axis=0) if xtrain.size else x
        xbpms = np.concatenate((xbpms, y), axis=0) if xbpms.size else y

    data_set = data_loader_clemson(xtrain, xbpms, args)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)
    source_loaders = [source_loader]

    # val domain data prep
    xval, ybpms = np.array([]), np.array([])
    for val_domain in val_domain_list:
        x, y = load_domain_data(val_domain)
        x = x.reshape((-1, x.shape[-1]))

        xval = np.concatenate((xval, x), axis=0) if xval.size else x
        ybpms = np.concatenate((ybpms, y), axis=0) if ybpms.size else y

    data_set = data_loader_clemson(xval, ybpms, args)
    val_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    # target domain data prep
    xtest, ybpms = np.array([]), np.array([])
    for target_domain in target_domain_list:
        x, y = load_domain_data(target_domain)
        x = x.reshape((-1, x.shape[-1]))

        xtest = np.concatenate((xtest, x), axis=0) if xtest.size else x
        ybpms = np.concatenate((ybpms, y), axis=0) if ybpms.size else y

    data_set = data_loader_clemson(xtest, ybpms, args)
    target_loader = DataLoader(data_set, batch_size=512, shuffle=False)  # For testing keep the batch size as 512

    return source_loaders, val_loader, target_loader


def prep_clemson(args):
    if args.cases == 'subject_large' or args.cases == 'subject_large_ssl_fn':
        return prep_domains_clemson_subject_large(args)
    elif args.cases == 'subject':
        return prep_domains_clemson_subject(args)
    elif args.cases == 'subject_val':
        return prep_domains_clemson_subject_val(args)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'

