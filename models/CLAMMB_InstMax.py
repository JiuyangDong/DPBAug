import torch.nn as nn
import torch 
import torch.nn.functional as F 
import random 
import numpy as np
import random
from topk.svm import SmoothTop1SVM

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		 
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)

class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class CLAM_SB(nn.Module):
    def __init__(self, I=1024, gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, args=None):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [I, 512, 256], "big": [I, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        inst_length = min(self.k_sample, A.shape[1])
        top_p_ids = torch.topk(A, inst_length)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, inst_length, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(inst_length, device)
        n_targets = self.create_negative_targets(inst_length, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        inst_length = min(self.k_sample, A.shape[1])
        top_p_ids = torch.topk(A, inst_length)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(inst_length, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False):
        assert len(h.shape) == 3         # B x N x I  
        B = h.shape[0]

        device = h.device
        A, h = self.attention_net(h)     # B x N x 1, B x N x L     the h here differs from the input in terms of the dimension 
        A = torch.permute(A, (0, 2, 1))  # B x 1 x N
        A = F.softmax(A, dim=-1)         # B x 1 x N

        if instance_eval:
            total_inst_loss = [0.0 for _ in range(B)]
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: # in-the-class:
                    tmp_all_preds, tmp_all_targets = [], []
                    for b in range(B):
                        instance_loss, preds, targets = self.inst_eval(A[b, :, :], h[b], classifier)
                        total_inst_loss[b] += instance_loss
                else: #out-of-the-class
                    if self.subtyping:
                        for b in range(B):
                            instance_loss, preds, targets = self.inst_eval_out(A[b, :, :], h[b], classifier)
                            total_inst_loss[b] += instance_loss
                    else:
                        continue

            total_inst_loss = torch.stack(total_inst_loss, 0)
            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.bmm(A, h)             # B x 1 x L
        logits = self.classifiers(M)    # B x 1 x C
        

        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss}
        else:
            results_dict = {}

        if return_features:
            results_dict.update({'features': M})

        return logits.squeeze(1), A, results_dict, M

class CLAMMB_InstMax(CLAM_SB):
    def __init__(self, I=1024, gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, args=None):
        nn.Module.__init__(self)

        self.args = args

        self.size_dict = {"small": [I, 512, 256], "big": [I, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        self.in_layer = nn.Sequential(*fc)


        if gate:
            self.attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            self.attention_ne = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        if self.args.patch_ratio > 0:
            self.k_sample = int(self.k_sample * self.args.patch_ratio)
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)

        # max-pooling classifier
        self.inst_classifier = nn.Sequential(
            nn.Linear(size[1], n_classes)
        )

    def forward_embed(self, h, label=None, instance_eval=False):
        A = self.attention_net(h)[0]
        B = h.shape[0]
        device = h.device
        A = torch.permute(A, (0, 2, 1))  # B x C x N
        A = F.softmax(A, dim=-1)         # B x C x N
        
        if instance_eval:
            total_inst_loss = [0.0 for _ in range(B)]
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    for b in range(B):
                        instance_loss, preds, targets = self.inst_eval(A[b, i, :], h[b], classifier)
                        total_inst_loss[b] += instance_loss
                else: #out-of-the-class
                    if self.subtyping:
                        for b in range(B):
                            instance_loss, preds, targets = self.inst_eval_out(A[b, i, :], h[b], classifier)
                            total_inst_loss[b] += instance_loss
                    else:
                        continue
                        
            total_inst_loss = torch.stack(total_inst_loss, 0)
            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
        else:
            total_inst_loss = None

        M = torch.bmm(A, h)                                              # B x C x L
        logits = torch.empty(B, self.n_classes).float().to(device)       # B x C

        for c in range(self.n_classes):
            logits[:, c] = self.classifiers[c](M[:, c, :]).squeeze(1)
        
        return logits, total_inst_loss

                
    def forward_inst(self, x):
        inst_preds = self.inst_classifier(x)                                                # B x N x C
        bag_logit = torch.max(inst_preds, 1, keepdim=True)[0]                               # B x 1 x C
        return bag_logit.squeeze(1)


    def forward(self, x, phase='train', label=None):
        if self.args.task == 'pretrain':
            x = self.in_layer(x)

            if phase == 'train'  and self.args.patch_ratio < 1.0:
                B, N, _ = x.shape
                embed_indexs = torch.randperm(N)[:int(N * self.args.patch_ratio)]
                inst_indexs = torch.randperm(N)[:int(N * self.args.patch_ratio)]
                embed_x, inst_x = x[:, embed_indexs, :], x[:, inst_indexs, :]           
            else:
                embed_x, inst_x = x, x
        else:
            if isinstance(x, list):
                embed_x, inst_x = self.in_layer(x[0]), self.in_layer(x[1])
            else:
                embed_x, inst_x = self.in_layer(x), self.in_layer(x)

        total_inst_loss = None
        embed_bag_logit, total_inst_loss = self.forward_embed(embed_x, label=label, instance_eval=(label!=None))                       # B x C
        inst_bag_logit = self.forward_inst(inst_x)                                                                                     # B x C

        return torch.cat([embed_bag_logit, inst_bag_logit], 0), total_inst_loss