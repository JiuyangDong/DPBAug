import torch
import pandas as pd
import os
import itertools
import logging
import json
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F 
from copy import deepcopy
import random
from dataset import init_data_wsi
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import math


class PretrainMIL:
    def __init__(self, args):
        self.args = args

        self.train_loader, self.valid_loader, self.test_loader = init_data_wsi(args)

        self.model = self.init_model(args.main_model, args.aux_model)
        self.model_ema = deepcopy(self.model).to(self.args.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-5)
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        
        # self.ema_alpha, self.counter, self.patience, self.stop_epoch, self.best_loss, self.flag = 0.9, 0, 20, 50, np.Inf, 1
        self.ema_alpha, self.counter, self.patience, self.stop_epoch, self.best_loss, self.flag = 0.9, 0, 20, 0, np.Inf, 1
        self.ckpt_name = os.path.join(self.args.ckpt_dir, 'pretrain_best_epoch.pth')
        self.best_valid_metrics = None 
    
    def init_model(self, main_model_type, aux_model_type):
        assert aux_model_type == 'Inst-Max'

        if main_model_type == 'ABMIL':
            from models.ABMIL_InstMax import ABMIL_InstMax
            model = ABMIL_InstMax(I=self.args.feature_dim, L=512, D=256, dropout=True, n_classes=self.args.n_classes, args=self.args).to(self.args.device)
        elif main_model_type == 'CLAM-MB':
            from models.CLAMMB_InstMax import CLAMMB_InstMax, SmoothTop1SVM
            model = CLAMMB_InstMax(I=self.args.feature_dim, dropout=True, n_classes=self.args.n_classes, subtyping=self.args.subtyping,
                instance_loss_fn = SmoothTop1SVM(n_classes = 2).cuda(self.args.device), k_sample=self.args.k_sample, args=self.args).to(self.args.device)
        elif main_model_type == 'MambaMIL':
            from models.MambaMIL_InstMax import MambaMIL_InstMax
            model = MambaMIL_InstMax(in_dim=self.args.feature_dim, n_classes=self.args.n_classes, dropout=0, act='gelu', layer=2, rate=5, type = 'SRMamba', args=self.args).to(self.args.device)
        
        return model

    def train(self):
        step = 0
        for epoch in range(1, self.args.n_epochs + 1):
            avg_train_loss = 0
            self.model.train()
            self.model_ema.eval()
            for i, (fea, label, _) in enumerate(tqdm(self.train_loader)):
                step += 1
                fea, label = fea.to(self.args.device), label.to(self.args.device)
                self.optimizer.zero_grad()
                loss = self.train_inference(fea, label, self.args.main_model, self.args.aux_model, self.model) 
                loss.backward()
                self.optimizer.step()
                avg_train_loss += loss.item()   

            self.ema_update(self.model, self.model_ema)
            avg_train_loss /= (i + 1)

            logging.info("In step {} (epoch {}), average train loss = {:.4f}".format(step, epoch, avg_train_loss))
            
            if epoch >= self.stop_epoch:
                self.valid(epoch)
                if self.flag == -1:
                    break
        return self.best_valid_metrics

    def valid(self, epoch):
        avg_loss = 0
        self.model_ema.eval()
        labels, probs = [], []
        for i, (fea, label, _) in enumerate(tqdm(self.valid_loader)):
            fea, label = fea.to(self.args.device), label.to(self.args.device)
            with torch.no_grad():
                loss, y_prob = self.test_inference(fea, label, self.args.main_model, self.args.aux_model, self.model_ema)
            labels.append(label.data.cpu().numpy())
            probs.append(y_prob.data.cpu().numpy())
            avg_loss += loss.item()
        avg_loss /= (i + 1)

        labels, probs = np.concatenate(labels, 0), np.concatenate(probs, 0)
        auc = self.cal_AUC(probs, labels, self.args.n_classes)
        acc, _, precision, recall, f1 = self.cal_ACC(probs, labels, self.args.n_classes)

        logging.info("loss = {:.4f}, auc = {:.4f}, acc = {:.4f}, precision = {:.4f}, recall = {:.4f}, f1 = {:.4f}".\
            format(avg_loss, auc, acc, precision, recall, f1))

        if epoch >= self.stop_epoch:
            if avg_loss < self.best_loss:
                self.counter = 0
                logging.info(f'Validation loss decreased ({self.best_loss:.4f} --> {avg_loss:.4f}).  Saving model ...')
                torch.save(self.model_ema.state_dict(), self.ckpt_name)
                self.best_loss = avg_loss
                self.best_valid_metrics = [avg_loss, auc, acc, precision, recall, f1]
            else:
                self.counter += 1
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.flag = -1

    def test(self):
        avg_loss = 0
        self.model_ema.load_state_dict(torch.load(self.ckpt_name))
        self.model_ema.eval()

        labels, probs = [], []
        for i, (fea, label, _) in enumerate(tqdm(self.test_loader)):
            fea, label = fea.to(self.args.device), label.to(self.args.device)
            with torch.no_grad():
                loss, y_prob = self.test_inference(fea, label, self.args.main_model, self.args.aux_model, self.model_ema)
            labels.append(label.data.cpu().numpy())
            probs.append(y_prob.data.cpu().numpy())
            avg_loss += loss.item()
        avg_loss /= (i + 1)

        labels, probs = np.concatenate(labels, 0), np.concatenate(probs, 0)

        auc = self.cal_AUC(probs, labels, self.args.n_classes)
        acc, _, precision, recall, f1 = self.cal_ACC(probs, labels, self.args.n_classes)

        logging.info("loss = {:.4f}, auc = {:.4f}, acc = {:.4f}, precision = {:.4f}, recall = {:.4f}, f1 = {:.4f}".\
            format(avg_loss, auc, acc, precision, recall, f1))

        return avg_loss, auc, acc, precision, recall, f1

    def train_inference(self, feas, label, main_model_type, aux_model_type, model):
        if main_model_type in ['ABMIL', 'MambaMIL'] and aux_model_type == 'Inst-Max':
            bag_logits, _ = model(feas, phase='train')
            B = bag_logits.shape[0]
            loss = self.loss(bag_logits, label.repeat(B))
        elif main_model_type == 'CLAM-MB' and aux_model_type == 'Inst-Max':
            bag_logits, total_inst_loss = model(feas, phase='train', label=label)
            B = bag_logits.shape[0]
            loss = self.loss(bag_logits, label.repeat(B)) * 0.7 + total_inst_loss.mean() * 0.3
        else:
            raise NotImplementedError

        return loss 

    def test_inference(self, fea, label, main_model_type, aux_model_type, model_ema):
        if (main_model_type == 'ABMIL' and aux_model_type == 'Inst-Max') or \
            (main_model_type == 'CLAM-MB' and aux_model_type == 'Inst-Max') or \
                (main_model_type == 'MambaMIL' and aux_model_type == 'Inst-Max'):
            bag_logits = model_ema(fea, phase='test')[0]
            B = bag_logits.shape[0]
            loss = self.loss(bag_logits, label.repeat(B))
            y_prob = sum([F.softmax(bag_logits[i: i + 1, :], dim=1) for i in range(B)]) / B

            return loss, y_prob

    def cal_AUC(self, probs, labels, nclasses):
        return roc_auc_score(labels, probs[:, 1])
    
    def cal_ACC(self, probs, labels, nclasses):
        log = [{"count": 0, "correct": 0} for i in range(nclasses)]
        pred_hat = np.argmax(probs, 1)
        labels = labels.astype(np.int32)

        acc_score = accuracy_score(labels, pred_hat)
        precision = precision_score(labels, pred_hat, average='binary')
        recall = recall_score(labels, pred_hat, average='binary')
        f1 = f1_score(labels, pred_hat, average='binary')

        return acc_score, log, precision, recall, f1

    def ema_update(self, model, model_ema):
        model_state_dict = model.state_dict()
        model_ema_state_dict = model_ema.state_dict()

        for name in model_ema_state_dict: 
            assert name in model_state_dict
            model_ema_state_dict[name].data = model_ema_state_dict[name].data * self.ema_alpha + model_state_dict[name].data * (1 - self.ema_alpha)

        model_ema.load_state_dict(model_ema_state_dict)


class DPBAugMIL(PretrainMIL):
    def __init__(self, args):
        self.args = args

        self.train_loader, self.valid_loader, self.test_loader = init_data_wsi(args)

        self.model, self.model_ema = self.load_pretrain(args)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-5)      
        self.train_loss, self.loss = torch.nn.CrossEntropyLoss(reduction='none'), torch.nn.CrossEntropyLoss(reduction='mean')
        
        self.ema_alpha, self.counter, self.patience, self.stop_epoch, self.best_loss, self.best_auc_acc, self.flag = 0.9, 0, 20, 0, np.Inf, -np.Inf, 1
        self.ckpt_name = os.path.join(self.args.ckpt_dir, 'dpbaug_best_epoch.pth')
        self.best_valid_metrics = None 

        # self.num_bags, self.cross_num, self.k_length, self.prun_ratio, self.warm_up_epoch, self.max_length = 1, 3, 32, 0.5, 20, 4
        self.num_bags, self.cross_num, self.k_length, self.prun_ratio, self.warm_up_epoch, self.max_length = 1, 3, 32, 0.5, 3, 4
        self.key_instance_pool_class0, self.key_instance_pool_class1, self.psebag_fea_memory, self.psebag_score_memory = {}, {}, {}, {}

    def load_pretrain(self, args):
        self.model = self.init_model(args.main_model, args.aux_model)
        if args.phase == 'train':
            pretrain_path = os.path.join(self.args.pretrain_path, 'fold-{}/pretrain_best_epoch.pth'.format(self.args.k)).replace('', '')
            print(self.model.load_state_dict(torch.load(pretrain_path, map_location=f'cuda:{args.gpu_id}')))
        self.model = self.model.to(self.args.device)
        self.model_ema = deepcopy(self.model).to(self.args.device)
        print(self.model)

        return self.model, self.model_ema

    def create_input(self, fea, label, path, epoch):
        B, N, I = fea.shape
        
        pseudo_bags = []
        inst_preds = self.model_ema.inst_classifier(self.model_ema.in_layer(fea))                                  # B x N x C
        inst_loss = torch.nn.CrossEntropyLoss(reduction='none')(inst_preds.squeeze(0), label.repeat(inst_preds.shape[1])) # N,  (0, +nan) 
        key_ids = torch.topk(inst_loss, self.k_length, dim=0, largest=True)[1].squeeze().tolist()
        ord_ids = list(set(list(range(N))).difference(set(key_ids)))

        key_instances = fea[:, key_ids, :].cpu()
        if label == 0:
            self.key_instance_pool_class0[path[0]] = key_instances
        else:
            self.key_instance_pool_class1[path[0]] = key_instances

        for _ in range(self.num_bags):
            cross_wsi_paths, cross_key_instances = [], []
            for _ in range(self.cross_num):
                if label == 0:
                    cross_wsi_path = random.choice(list(self.key_instance_pool_class0.keys()))
                    cross_key_instance = self.key_instance_pool_class0[cross_wsi_path]   
                else:
                    cross_wsi_path = random.choice(list(self.key_instance_pool_class1.keys()))
                    cross_key_instance = self.key_instance_pool_class1[cross_wsi_path]
                cross_wsi_paths.append(cross_wsi_path)
                cross_key_instances.append(cross_key_instance)
            w_scaler = 2    # dual branch
            for _ in range(w_scaler):
                ord_length = max(min(len(ord_ids), len(key_ids)), int(N * self.args.patch_ratio) - (1 + self.cross_num) * len(key_ids))
                ord_instances = fea[:, random.sample(ord_ids, ord_length), :].cpu()
                psebag = torch.cat([key_instances, ord_instances] + [ck for ck in cross_key_instances], 1)
                pseudo_bags.append(psebag)
        # output: [b11, b12, b21, b22, b31, b32, ...]

        if path[0] in self.psebag_fea_memory:
            # update
            if len(self.psebag_fea_memory[path[0]]) < self.max_length * w_scaler:
                for i in range(self.num_bags * w_scaler):
                    self.psebag_fea_memory[path[0]].append(pseudo_bags[i])
                for j in range(self.num_bags):
                    self.psebag_score_memory[path[0]].append(torch.zeros(1).cpu())
            # prune
            else:
                scores = torch.cat(self.psebag_score_memory[path[0]], 0)
                max_score_indexs = torch.topk(scores, int(self.max_length * (1 - self.prun_ratio)), 0)[1].tolist()
                tmp = []
                for j in max_score_indexs:
                    tmp.append(self.psebag_fea_memory[path[0]][2 * j])
                    tmp.append(self.psebag_fea_memory[path[0]][2 * j + 1])
                
                self.psebag_fea_memory[path[0]] = tmp
                del tmp
                self.psebag_score_memory[path[0]] = [self.psebag_score_memory[path[0]][i] for i in max_score_indexs]

                for i in range(self.num_bags * w_scaler):
                    self.psebag_fea_memory[path[0]].append(pseudo_bags[i])
                for j in range(self.num_bags):
                    self.psebag_score_memory[path[0]].append(torch.zeros(1).cpu())
                    
        else:
            self.psebag_fea_memory[path[0]] = [pb.cpu() for pb in pseudo_bags]
            self.psebag_score_memory[path[0]] = [torch.zeros(1).cpu() for _ in range(self.num_bags)]

        if epoch < self.warm_up_epoch:
            pseudo_bags = torch.cat(pseudo_bags, 0).to(self.args.device) # [11 21 ... n1] / [11, 12, 21, 22, ..., n1, n2]
            return [pseudo_bags[::2], pseudo_bags[1::2]]
        else:
            pruned_pseudo_bags = torch.cat(self.psebag_fea_memory[path[0]], 0).to(self.args.device)
            return [pruned_pseudo_bags[::2], pruned_pseudo_bags[1::2]]
    
    def train(self):
        self.valid(0)
        logging.info('Beginning WarmUp ~~~~~~~~~~~~~~~~~~~~~')
        for epoch in range(1, self.args.n_epochs + 1):
            if epoch == self.warm_up_epoch + 1:
                logging.info('Beginning Pruning ~~~~~~~~~~~~~~~~~~~~~')

            avg_train_loss = 0
            self.model.train()   
            self.model_ema.eval()     

            c = 0
            for i, (fea, label, path) in enumerate(tqdm(self.train_loader)):
                fea, label = fea.to(self.args.device), label.to(self.args.device)
                self.optimizer.zero_grad()
                with torch.no_grad():
                    pseudo_bags = self.create_input(fea, label, path, epoch)   

                loss, time_avg_loss = self.train_inference(pseudo_bags, label, self.args.main_model, self.args.aux_model, self.model)                   
                    
                score = torch.tensor(self.psebag_score_memory[path[0]]) * self.ema_alpha + time_avg_loss * (1 - self.ema_alpha)
                self.psebag_score_memory[path[0]] = list(score.split(1))
                        
                c += 1               
                avg_train_loss += loss.item()    
                loss.backward()
                self.optimizer.step()

            avg_train_loss /= (i + 1)
            self.ema_update(self.model, self.model_ema)
                        
            logging.info("In epoch {}, kept/all wsi numbers = {}/{} average train loss = {:.4f} self.flag={}".format(epoch, c, len(self.train_loader), avg_train_loss, self.flag))
            
            self.valid(epoch)

            if self.flag == -1:
                break

        return self.best_valid_metrics

    def valid(self, epoch):
        avg_loss = 0

        self.model_ema.eval()
        labels, probs = [], []

        for i, (fea, label, path) in enumerate(tqdm(self.valid_loader)):
            fea, label = fea.to(self.args.device), label.to(self.args.device)
            with torch.no_grad():
                loss, y_prob = self.test_inference(fea, label, self.args.main_model, self.args.aux_model, self.model_ema)
                
            labels.append(label.data.cpu().numpy()) 
            probs.append(y_prob.data.cpu().numpy())
            avg_loss += loss.item()
        avg_loss /= (i + 1)

        labels, probs = np.concatenate(labels, 0), np.concatenate(probs, 0)

        auc = self.cal_AUC(probs, labels, self.args.n_classes)
        acc, _, precision, recall, f1 = self.cal_ACC(probs, labels, self.args.n_classes)

        logging.info("loss = {:.4f}, auc = {:.4f}, acc = {:.4f}, precision = {:.4f}, recall = {:.4f}, f1 = {:.4f}".\
            format(avg_loss, auc, acc, precision, recall, f1))
        
        if epoch == 0 or epoch >= self.stop_epoch:
            auc_acc = (auc + acc) / 2.
            if auc_acc > self.best_auc_acc or (auc_acc == self.best_auc_acc and avg_loss < self.best_loss):
                self.counter = 0
                logging.info(f'Validation auc_acc increased ({self.best_auc_acc:.4f} --> {auc_acc:.4f}).  Saving model ...')
                torch.save(self.model_ema.state_dict(), self.ckpt_name)
                self.best_auc_acc = auc_acc
                self.best_loss = avg_loss
                self.best_valid_metrics = [avg_loss, auc, acc, precision, recall, f1]
            else:
                if epoch > self.warm_up_epoch:
                    self.counter += 1
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.flag = -1

    def train_inference(self, feas, label, main_model_type, aux_model_type, model):
        if main_model_type in ['ABMIL', 'MambaMIL'] and aux_model_type == 'Inst-Max':
            bag_logits, _ = model(feas, phase='train')
            B = bag_logits.shape[0]
            bag_loss = self.train_loss(bag_logits, label.repeat(B))
            embed_bag_loss = bag_loss[:B//2]
            inst_bag_loss = bag_loss[B//2:]
            loss = bag_loss.mean()
        
        elif main_model_type == 'CLAM-MB' and aux_model_type == 'Inst-Max':
            bag_logits, total_inst_loss = model(feas, phase='train', label=label)
            B = bag_logits.shape[0]
            bag_loss = self.train_loss(bag_logits, label.repeat(B))
            embed_bag_loss = bag_loss[:B//2]
            inst_bag_loss = bag_loss[B//2:]
            inst_loss = total_inst_loss
            loss = 0.5 * (0.7 * embed_bag_loss.mean() + 0.3 * inst_loss.mean()) + 0.5 * inst_bag_loss.mean()
        else:
            raise NotImplementedError

        time_avg_loss = (embed_bag_loss.data.cpu() + inst_bag_loss.data.cpu()) / 2.

        return loss, time_avg_loss.detach()
