import argparse
import torch
import csv
import random
import os
import numpy as np
import logging
from torch.utils.data import Dataset
import os 
import numpy as np
import torch 
from tqdm import tqdm



def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_loggers(stdout_txt):
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(stdout_txt)
    formatter = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel('INFO')

def parse_args_and_save():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--phase", type=str, default='train')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=10)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--feature_dim", type=int, default=1024)
    parser.add_argument("--task", type=str, default="pretrain")
    parser.add_argument("--label_frac", type=float, default=1.0)
    parser.add_argument("--dataset", type=str, default="Camelyon16")
    parser.add_argument("--feature_extractor", type=str, default="ResNet50_ImageNet")
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--main_model", type=str, default="ABMIL")
    parser.add_argument("--aux_model", type=str, default='Inst-Max')
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--patch_ratio", type=float, default=1.0)
    parser.add_argument("--ema_alpha", type=float, default=0.9)
    parser.add_argument("--pretrain_path", type=str, default='')


    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu_id))


    assert args.task in ['pretrain', 'dpbaug']

    args.feature_dir = f'/zhang_yb/02.data/02.processed_data/{args.dataset}/20x/feats/'
    args.label_csv = f'/zhang_yb/02.data/02.processed_data/{args.dataset}_label.csv'
    args.split_dir = f'/zhang_yb/02.data/02.processed_data/{args.dataset}/splits/'

    assert args.dataset in ['Camelyon16', 'Camelyon17', 'TCGA-NSCLC', 'TCGA-BRCA']
    args.n_classes, args.k_sample = 2, 32
    args.subtyping = False if args.dataset in ['Camelyon16', 'Camelyon17'] else True

    assert args.feature_extractor in ['ResNet50_ImageNet', 'ViTB32_ImageNet']
    args.feature_dim = 1024 if args.feature_extractor == 'ResNet50_ImageNet' else 768

    if args.task == 'pretrain':
        exp_dir = os.path.join('experiments/stage-{}/{}-20x-{}fold/{}/{}+{}/label_frac={}/patch_ratio={}'.
            format(args.task, args.dataset, args.fold, args.feature_extractor, args.main_model, args.aux_model, args.label_frac, args.patch_ratio)
        )
    else:
        exp_dir = os.path.join('experiments/stage-{}/{}-20x-{}fold/{}/{}+{}/label_frac={}/patch_ratio={}'.
            format(args.task, args.dataset, args.fold, args.feature_extractor, args.main_model, args.aux_model, args.label_frac, args.patch_ratio)
        )
  

    args.exp_dir = exp_dir
    args.log_dir = os.path.join(args.exp_dir, 'logs')

    if os.path.exists(args.exp_dir):
        # if args.k == 0 and args.phase == 'train':
        #     os.system('rm -rvf {}'.format(args.exp_dir))
        pass
    else:    
        os.makedirs(args.exp_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    set_loggers(os.path.join(args.log_dir, "{}-stdout-fold{}.txt".format(args.phase, args.k)))


    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp dir = {}".format(args.exp_dir))
    logging.info("Writing log file to {}".format(os.path.join(args.exp_dir, 'logs')))

    return args
   
def main(args):
    csv_name = 'valid_metrics.csv' if args.phase == 'train' else f'{args.phase}_metrics.csv'
        
    with open(os.path.join(args.log_dir, csv_name), 'a') as f:
        writer = csv.writer(f)

        if args.k == 0:
            if args.phase == 'train':
                writer.writerow(['fold', 'valid_loss', 'valid_auc', 'valid_acc', 'valid_precision', 'valid_recall', 'valid_f1'])
            else:
                writer.writerow(['fold', 'test_loss', 'test_auc', 'test_acc', 'test_precision', 'test_recall', 'test_f1'])

        logging.info("\n{}start fold {} {}".format(''.join(['*'] * 50), args.k, ''.join(['*'] * 50)))

        args.ckpt_dir = os.path.join(args.exp_dir, 'ckpts/fold-{}'.format(args.k))
        if not os.path.exists(args.ckpt_dir):   
            os.makedirs(args.ckpt_dir)
                
        if args.task == 'pretrain':
            from solver import PretrainMIL as MIL
        else:
            from solver import DPBAugMIL as MIL

        MIL_runner = MIL(args)
        if args.phase == 'train':
            loss, auc, acc, precision, recall, f1 = MIL_runner.train()
        elif args.phase == 'test':
            loss, auc, acc, precision, recall, f1 = MIL_runner.test()
       
        writer.writerow(['{}'.format(args.k)] + [round(loss, 4), round(auc, 4), round(acc, 4), round(precision, 4), round(recall, 4), round(f1, 4)])

if __name__ == '__main__':
    args = parse_args_and_save()
    seed_torch(args.seed)
    main(args)
