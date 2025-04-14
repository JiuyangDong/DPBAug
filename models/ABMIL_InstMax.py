import torch.nn as nn
import torch 
import torch.nn.functional as F 
import random 

class ABMIL_InstMax(nn.Module):
    def __init__(self, I=1024, L = 512, D = 256, dropout = False, n_classes = 1, args=None):
        super(ABMIL_InstMax, self).__init__()
        self.args = args

        self.in_layer = nn.Sequential(
            nn.Linear(I, L),
            nn.ReLU(), 
            nn.Dropout(0.25)
        )

        self.attention = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Linear(D, 1)
        )
        
        self.inst_classifier = nn.Sequential(
            nn.Linear(L, n_classes)
        )

        self.bag_classifier = nn.Sequential(
            nn.Linear(L, n_classes)
        )

    def forward_embed(self, x):
        A = self.attention(x)                                                               # B x N x 1
        A = torch.permute(A, (0, 2, 1))                                                     # B x 1 x N
        A_soft = F.softmax(A, dim=-1)                                                       # B x 1 x N
        M = torch.bmm(A_soft, x)                                                            # B x 1 x L
        bag_logit = self.bag_classifier(M)                                                  # B x 1 x C
        return bag_logit.squeeze(1)

    def forward_inst(self, x):
        inst_preds = self.inst_classifier(x)                                                # B x N x C
        bag_logit = torch.max(inst_preds, 1, keepdim=True)[0]                               # B x 1 x C
        return bag_logit.squeeze(1)

    def forward(self, x, phase='train'):                          # B x N x I
        if self.args.task == 'pretrain':
            x = self.in_layer(x)

            if phase == 'train' and self.args.patch_ratio < 1.0:
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