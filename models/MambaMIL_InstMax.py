"""
MambaMIL
"""
import torch
import torch.nn as nn
from models.mamba.mamba_ssm import SRMamba
from models.mamba.mamba_ssm import BiMamba
from models.mamba.mamba_ssm import Mamba
import torch.nn.functional as F
import random

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class MambaMIL_InstMax(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act, survival = False, layer=2, rate=10, type="SRMamba", args=None):
        super(MambaMIL_InstMax, self).__init__()
        self.args = args

        self.in_layer = [nn.Linear(in_dim, 512)]
        if act.lower() == 'relu':
            self.in_layer += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.in_layer += [nn.GELU()]
        if dropout:
            self.in_layer += [nn.Dropout(dropout)]

        self.in_layer = nn.Sequential(*self.in_layer)
        self.norm = nn.LayerNorm(512)
        self.layers = nn.ModuleList()
        self.survival = survival

        if type == "SRMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        SRMamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        elif type == "Mamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        Mamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        elif type == "BiMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        BiMamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        else:
            raise NotImplementedError("Mamba [{}] is not implemented".format(type))

        self.n_classes = n_classes
        self.classifier = nn.Linear(512, self.n_classes)
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.rate = rate
        self.type = type

        self.apply(initialize_weights)

        self.inst_classifier = nn.Sequential(
            nn.Linear(512, n_classes)
        )

    def forward_embed(self, h):
        if self.type == "SRMamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)
                h = layer[1](h, rate=self.rate)
                h = h + h_
        elif self.type == "Mamba" or self.type == "BiMamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)
                h = layer[1](h)
                h = h + h_

        h = self.norm(h)
        A = self.attention(h) # [B, n, K]
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1) # [B, K, n]
        h = torch.bmm(A, h) # [B, K, 512]
        h = h.squeeze(0)

        logits = self.classifier(h)  # [B, n_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        A_raw = None
        results_dict = None
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None
        return logits.squeeze(1)
    
    def forward_inst(self, x):
        inst_preds = self.inst_classifier(x)                                                # B x N x C
        bag_logit = torch.max(inst_preds, 1, keepdim=True)[0]                               # B x 1 x C
        return bag_logit.squeeze(1)

    def forward(self, x, phase='train'):                          # B x N x I
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

            

        embed_bag_logit = self.forward_embed(embed_x)                      # B x C
        inst_bag_logit = self.forward_inst(inst_x)                         # B x C
        
        return torch.cat([embed_bag_logit, inst_bag_logit], 0), None
        
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers  = self.layers.to(device)
        
        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)