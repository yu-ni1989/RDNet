import torch
from torch import nn
import torch.nn.functional as F
from .attention import PAM_Module, CAM_Module

class Attention_PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, out_dim, bins):
        super(Attention_PPM, self).__init__()
        self.apools = []
        self.p_attens = []
        self.c_attens = []
        self.beg_convs1 = []
        self.beg_convs2 = []
        self.end_convs1 = []
        self.end_convs2 = []

        for bin in bins:
            self.apools.append(nn.AdaptiveAvgPool2d(bin))
            self.beg_convs1.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
            self.beg_convs2.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
            self.p_attens.append(PAM_Module(reduction_dim))
            self.c_attens.append(CAM_Module(reduction_dim))
            self.end_convs1.append(nn.Sequential(
                nn.Conv2d(reduction_dim, reduction_dim, kernel_size=3, padding=1, groups=reduction_dim, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
            self.end_convs2.append(nn.Sequential(
                nn.Conv2d(reduction_dim, reduction_dim, kernel_size=3, padding=1, groups=reduction_dim, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))

        self.f_convs = nn.Sequential(
            nn.Conv2d(reduction_dim * len(bins) * 2 + in_dim, out_dim, kernel_size=1, padding=0, bias=False),
            # kernel size reduced
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        self.apools = nn.ModuleList(self.apools)
        self.beg_convs1 = nn.ModuleList(self.beg_convs1)
        self.beg_convs2 = nn.ModuleList(self.beg_convs2)
        self.p_attens = nn.ModuleList(self.p_attens)
        self.c_attens = nn.ModuleList(self.c_attens)
        self.end_convs1 = nn.ModuleList(self.end_convs1)
        self.end_convs2 = nn.ModuleList(self.end_convs2)


    def forward(self, x):
        x_size = x.size()
        x_as = [x]
        for i in range(len(self.apools)):
            x_f = self.apools[i](x)

            x_a = self.beg_convs1[i](x_f)
            x_a = self.p_attens[i](x_a)
            x_a = self.end_convs1[i](x_a)

            x_b = self.beg_convs2[i](x_f)
            x_b = self.c_attens[i](x_b)
            x_b = self.end_convs2[i](x_b)

            x_as.append(F.interpolate(x_a, x_size[2:], mode='bilinear', align_corners=True))
            x_as.append(F.interpolate(x_b, x_size[2:], mode='bilinear', align_corners=True))

        x_as = torch.cat(x_as, dim=1)
        return self.f_convs(x_as)



