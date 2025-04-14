import os
import argparse
import time 

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str)
parser.add_argument("--gpu_id", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--fold", type=int, default=10)
parser.add_argument("--label_frac", type=float, default=1.00)
parser.add_argument("--feature_extractor", type=str)
parser.add_argument("--dataset", type=str, default='Camelyon16')
parser.add_argument("--main_model", type=str, default='ABMIL')
parser.add_argument("--aux_model", type=str, default='Inst-Max')
parser.add_argument("--patch_ratio", type=float, default=1.0)
parser.add_argument("--pretrain_path", type=str, default='None')
args = parser.parse_args()

#######################################################
# train and val
#######################################################
for k in range(1):
    train_cmd = 'python main.py --task {} --gpu_id {} --phase train --dataset {} --feature_extractor {} --main_model {} --aux_model {} --fold {} --label_frac {} --patch_ratio {} --lr {} --k {} --pretrain_path {}'.\
        format(args.task, args.gpu_id, args.dataset, args.feature_extractor, args.main_model, args.aux_model, args.fold, args.label_frac, args.patch_ratio, args.lr, k, args.pretrain_path)
    os.system(train_cmd)

    test_cmd = 'python main.py --task {} --gpu_id {} --phase test --dataset {} --feature_extractor {} --main_model {} --aux_model {} --fold {} --label_frac {} --patch_ratio {} --lr {} --k {} --pretrain_path {}'.\
        format(args.task, args.gpu_id, args.dataset, args.feature_extractor, args.main_model, args.aux_model, args.fold, args.label_frac, args.patch_ratio, args.lr, k, args.pretrain_path)
    os.system(test_cmd)