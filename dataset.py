import pandas as pd
import os 
import logging
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
from sklearn.metrics import roc_auc_score
import numpy as np

class WSIDataset(Dataset):
    def __init__(self, args, wsi_labels, infold_cases):
        self.args = args
        self.wsi_labels = wsi_labels

        self.infold_features, self.infold_labels = [], []
        for case_id, slide_id, label in wsi_labels:
            if case_id in infold_cases:
                if args.feature_extractor in ['ResNet50_ImageNet', 'ViTB32_ImageNet']:
                    if self.args.dataset in ['Camelyon16', 'Camelyon17', 'TCGA-NSCLC', 'TCGA-BRCA']:
                        fea_path = os.path.join(args.feature_dir, args.feature_extractor, slide_id+'.pt')
                        if os.path.exists(fea_path):
                            self.infold_features.append(fea_path)
                            self.infold_labels.append(label)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
                    
    def __len__(self):
        return len(self.infold_features)
        
    def __getitem__(self, index):
        path = self.infold_features[index]
        label = self.infold_labels[index]        
        fea = torch.load(path)
        return fea, label, path


def read_wsi_label(args):
    data = pd.read_csv(args.label_csv)

    wsi_labels = []
    for i in range(len(data)):
        case_id, slide_id, label = data.loc[i, "case_id"], data.loc[i, "slide_id"], data.loc[i, "label"]

        if args.dataset in ['Camelyon16']:
            assert label in ['tumor_tissue', 'normal_tissue']
            label = 0 if label == 'normal_tissue' else 1
        elif args.dataset == 'TCGA-NSCLC':
            assert label in ['TCGA-LUSC', 'TCGA-LUAD']
            label = 0 if label == 'TCGA-LUSC' else 1
        elif args.dataset == 'TCGA-BRCA':
            assert label in ['Infiltrating Ductal Carcinoma', 'Infiltrating Lobular Carcinoma']
            label = 0 if label == 'Infiltrating Ductal Carcinoma' else 1
        else:
            raise NotImplementedError

        wsi_labels.append([case_id, slide_id, label])   

    return wsi_labels

def read_in_fold_cases(fold_csv):
    data = pd.read_csv(fold_csv)
    train_cases, valid_cases, test_cases = data.loc[:, 'train'].dropna(axis=0, how='any').to_list(), data.loc[:, 'val'].dropna(axis=0, how='any').to_list(), data.loc[:, 'test'].dropna(axis=0, how='any').to_list()
    return train_cases, valid_cases, test_cases

def make_weights_for_balanced_classes_split(data_set):
    N = float(len(data_set))           

    classes = {}
    for label in data_set.infold_labels:
        if label not in classes:
            classes[label] = 1
        else:
            classes[label] += 1
                                                                                                
    weight = [0] * int(N)                                           
    for idx in range(len(data_set)):   
        y = data_set.infold_labels[idx]                       
        weight[idx] = N / classes[y]    
        
    return torch.DoubleTensor(weight)

def init_data_wsi(args):
    wsi_labels = read_wsi_label(args)

    split_dir = os.path.join(args.split_dir, '{}-fold-{}%-label/').format(args.fold, int(args.label_frac * 100))
    train_cases, valid_cases, test_cases = read_in_fold_cases(os.path.join(split_dir + 'splits_{}.csv'.format(args.k)))

    train_set = WSIDataset(args, wsi_labels, train_cases)
    valid_set = WSIDataset(args, wsi_labels, valid_cases)
    test_set = WSIDataset(args, wsi_labels, test_cases)
    
    logging.info("Case/WSI number for trainset in fold-{} = {}/{}".format(args.k, len(train_cases), len(train_set)))
    logging.info("Case/WSI number for validset in fold-{} = {}/{}".format(args.k, len(valid_cases), len(valid_set)))
    logging.info("Case/WSI number for testset in fold-{} = {}/{}".format(args.k, len(test_cases), len(test_set)))

    weights = make_weights_for_balanced_classes_split(train_set)

    if args.dataset in ['Camelyon16', 'TCGA-NSCLC', 'TCGA-BRCA']:
        train_loader = DataLoader(train_set, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights), replacement=True))
        valid_loader = DataLoader(valid_set, batch_size=1, sampler = SequentialSampler(valid_set))
        test_loader = DataLoader(test_set, batch_size=1, sampler = SequentialSampler(test_set))

        return train_loader, valid_loader, test_loader
    else:
            raise NotImplementedError