###########################################################################################
###########################################################################################

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from .resnet import resnet18
from .rdm import RDM
from .rdm import IS
from .aapp import Attention_PPM
from .selected_loss import SelectedLoss


class RDNet(nn.Module):
    def __init__(self, layers=18, bins=(5, 8, 11), dropout=0.1, classes=2, ignore_label=255, use_ppm=True, pretrained=True):
        super(RDNet, self).__init__()
        assert layers in [18, 50, 101, 152]
        assert classes > 1
        self.use_ppm = use_ppm
        self.classes = classes
        self.ignore_label = ignore_label
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.criterion_f = SelectedLoss(0.7, self.ignore_label, classes=self.classes, y=None)

        self.resnet = resnet18(pretrained=pretrained)
        self.layer0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4

        self.cs_dim = 512
        self.rdm_dim1 = 64
        self.rdm_dim2 = 128

        ### RDM
        self.rdm_res1 = RDM(self.cs_dim, self.rdm_dim1, self.rdm_dim1)
        self.rdm_res2 = RDM(self.cs_dim, self.rdm_dim2, self.rdm_dim2)

        ### IS
        self.IS = IS(self.rdm_dim1*2 + self.rdm_dim2*2, self.rdm_dim1*2 + self.rdm_dim2*2)

        ### CS
        self.res_dim = 512
        if use_ppm:
            self.ppm = Attention_PPM(self.res_dim, int(self.res_dim//(len(bins)*4)), self.cs_dim, bins)

        ### Prediction
        self.cls = nn.Sequential(
            nn.Conv2d(self.cs_dim + self.rdm_dim1*2 + self.rdm_dim2*2, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(128, classes, kernel_size=1)
        )
        if self.training:
            self.aux_512_a = nn.Sequential(
                nn.Conv2d(self.cs_dim, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(64, classes, kernel_size=1)
            )
            # self.aux_512_b = nn.Sequential(
            #     nn.Conv2d(self.res_dim, 64, kernel_size=3, padding=1, bias=False),
            #     nn.BatchNorm2d(64),
            #     nn.ReLU(inplace=True),
            #     nn.Dropout2d(p=dropout),
            #     nn.Conv2d(64, classes, kernel_size=1)
            # )


    def MainStream(self, x):
        x = self.layer0(x)
        x_res1 = self.layer1(x)
        x_res2 = self.layer2(x_res1)
        x_res3 = self.layer3(x_res2)
        x_res = self.layer4(x_res3)

        if self.use_ppm:
            x = self.ppm(x_res)

        return x_res1, x_res2, x_res3, x_res, x


    def MultiSeq(self, x_low_sem, x_high):
        x1_size = x_low_sem.size()
        xs = [x_low_sem]
        xs.append(F.interpolate(x_high, x1_size[2:], mode='bilinear', align_corners=True))

        xs = torch.cat(xs, 1)
        return xs

    def forward(self, x, y=None, pred=False):
        x_size = x.size()
        h = x_size[2]
        w = x_size[3]

        x_res1, x_res2, x_res3, x_res, x_high = self.MainStream(x)

        x_res1 = F.interpolate(x_res1, x_res2.shape[-2:], mode='bilinear', align_corners=True)
        x_diff1 = self.rdm_res1(x_res1, x_high)
        x_diff2 = self.rdm_res2(x_res2, x_high)

        x_low_sem = self.IS(torch.cat((x_diff1, x_diff2), dim=1))
        del x_res1, x_res2, x_res3, x_diff1, x_diff2

        x = self.MultiSeq(x_low_sem, x_high)
        del x_low_sem

        x = self.cls(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if pred:
            return x
        if self.training:
            self.criterion_f = SelectedLoss(0.7, self.ignore_label, classes=self.classes, y=None)

            aux_512_a = self.aux_512_a(x_high)
            aux_512_a = F.interpolate(aux_512_a, size=(h, w), mode='bilinear', align_corners=True)
            # aux_512_b = self.aux_512_b(x_res)
            # aux_512_b = F.interpolate(aux_512_b, size=(h, w), mode='bilinear', align_corners=True)

            main_loss = self.criterion_f(x, y)
            aux_loss = self.criterion(aux_512_a, y)
            # aux_loss2 = self.criterion(aux_512_b, y)
            # aux_loss = aux_loss1 + aux_loss2
            # aux_loss = aux_loss1

            return x.max(1)[1], main_loss, aux_loss
        else:
            return x.max(1)[1]
            # return x






