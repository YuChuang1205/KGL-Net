#!/usr/bin/python3
# coding = gbk
"""
@Author : yuchuang
@Time :
@desc: KGL-Net
"""

from .module import *

##输入是64×64   KGL-Net
class feature_KGL(nn.Module):

    def __init__(self, is_bias=True, is_bias_FRN=True, dim_desc=128, drop_rate=0.2):
        super(feature_KGL, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate

        self.layer1_com1 = nn.Sequential(
            FRN(1, is_bias=is_bias_FRN),
            TLU(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer2_com1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
            ECA(in_channel=32, gamma=2, b=1),
        )

        self.layer3_com1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )

        self.layer4_com1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
            ECA(in_channel=64, gamma=2, b=1),
        )

        self.layer5_com1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer6_com1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
            ECA(in_channel=128, gamma=2, b=1),
        )

        self.layer7_com1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(256, is_bias=is_bias_FRN),
            TLU(256),
        )

        self.layer8_com1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=is_bias),
            FRN(256, is_bias=is_bias_FRN),
            TLU(256),
            ECA(in_channel=256, gamma=2, b=1),
        )

        self.layer9_com1 = nn.Sequential(
            # nn.Dropout(self.drop_rate),
            nn.Conv2d(256, self.dim_desc, kernel_size=8, bias=is_bias),
            nn.BatchNorm2d(self.dim_desc)
        )

        ####################

        self.layer1_com2 = nn.Sequential(
            FRN(1, is_bias=is_bias_FRN),
            TLU(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer2_com2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
            ECA(in_channel=32, gamma=2, b=1),
        )

        self.layer3_com2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )

        self.layer4_com2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
            ECA(in_channel=64, gamma=2, b=1),
        )

        # self.layer5_com2 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
        #     FRN(128, is_bias=is_bias_FRN),
        #     TLU(128),
        # )
        #
        # self.layer6_com2 = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
        #     FRN(128, is_bias=is_bias_FRN),
        #     TLU(128),
        #     ECA(in_channel=128, gamma=2, b=1),
        # )
        #
        # self.layer7_com2 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=is_bias),
        #     FRN(256, is_bias=is_bias_FRN),
        #     TLU(256),
        # )
        #
        # self.layer8_com2 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=is_bias),
        #     FRN(256, is_bias=is_bias_FRN),
        #     TLU(256),
        #     ECA(in_channel=256, gamma=2, b=1),
        # )
        #
        # self.layer9_com2 = nn.Sequential(
        #     # nn.Dropout(self.drop_rate),
        #     nn.Conv2d(256, self.dim_desc, kernel_size=8, bias=is_bias),
        #     nn.BatchNorm2d(self.dim_desc)
        # )
        #
        # ####################
        #
        # self.layer1_dif1 = nn.Sequential(
        #     FRN(1, is_bias=is_bias_FRN),
        #     TLU(1),
        #     nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
        #     FRN(32, is_bias=is_bias_FRN),
        #     TLU(32),
        # )
        #
        # self.layer2_dif1 = nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
        #     FRN(32, is_bias=is_bias_FRN),
        #     TLU(32),
        #     ECA(in_channel=32, gamma=2, b=1),
        # )
        #
        # self.layer3_dif1 = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
        #     FRN(64, is_bias=is_bias_FRN),
        #     TLU(64),
        # )
        #
        # self.layer4_dif1 = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
        #     FRN(64, is_bias=is_bias_FRN),
        #     TLU(64),
        #     ECA(in_channel=64, gamma=2, b=1),
        # )

        self.layer5_dif1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer6_dif1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
            ECA(in_channel=128, gamma=2, b=1),
        )

        self.layer7_dif1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(256, is_bias=is_bias_FRN),
            TLU(256),
        )

        self.layer8_dif1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=is_bias),
            FRN(256, is_bias=is_bias_FRN),
            TLU(256),
            ECA(in_channel=256, gamma=2, b=1),
        )

        ####################

        # self.layer1_dif2 = nn.Sequential(
        #     FRN(1, is_bias=is_bias_FRN),
        #     TLU(1),
        #     nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
        #     FRN(32, is_bias=is_bias_FRN),
        #     TLU(32),
        # )
        #
        # self.layer2_dif2 = nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
        #     FRN(32, is_bias=is_bias_FRN),
        #     TLU(32),
        #     ECA(in_channel=32, gamma=2, b=1),
        # )
        #
        # self.layer3_dif2 = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
        #     FRN(64, is_bias=is_bias_FRN),
        #     TLU(64),
        # )
        #
        # self.layer4_dif2 = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
        #     FRN(64, is_bias=is_bias_FRN),
        #     TLU(64),
        #     ECA(in_channel=64, gamma=2, b=1),
        # )

        self.layer5_dif2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer6_dif2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
            ECA(in_channel=128, gamma=2, b=1),
        )

        self.layer7_dif2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(256, is_bias=is_bias_FRN),
            TLU(256),
        )

        self.layer8_dif2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=is_bias),
            FRN(256, is_bias=is_bias_FRN),
            TLU(256),
            ECA(in_channel=256, gamma=2, b=1),
        )


        ###-------------
        self.layer1_com1.apply(init_weights2)
        self.layer2_com1.apply(init_weights2)
        self.layer3_com1.apply(init_weights2)
        self.layer4_com1.apply(init_weights2)
        self.layer5_com1.apply(init_weights2)
        self.layer6_com1.apply(init_weights2)
        self.layer7_com1.apply(init_weights2)
        self.layer8_com1.apply(init_weights2)
        self.layer9_com1.apply(init_weights2)

        self.layer1_com2.apply(init_weights2)
        self.layer2_com2.apply(init_weights2)
        self.layer3_com2.apply(init_weights2)
        self.layer4_com2.apply(init_weights2)
        # self.layer5_com2.apply(init_weights2)
        # self.layer6_com2.apply(init_weights2)
        # self.layer7_com2.apply(init_weights2)
        # self.layer8_com2.apply(init_weights2)
        # self.layer9_com2.apply(init_weights2)
        #
        # ###-------------
        # self.layer1_dif1.apply(init_weights2)
        # self.layer2_dif1.apply(init_weights2)
        # self.layer3_dif1.apply(init_weights2)
        # self.layer4_dif1.apply(init_weights2)
        self.layer5_dif1.apply(init_weights2)
        self.layer6_dif1.apply(init_weights2)
        self.layer7_dif1.apply(init_weights2)
        self.layer8_dif1.apply(init_weights2)

        # self.layer1_dif2.apply(init_weights2)
        # self.layer2_dif2.apply(init_weights2)
        # self.layer3_dif2.apply(init_weights2)
        # self.layer4_dif2.apply(init_weights2)
        self.layer5_dif2.apply(init_weights2)
        self.layer6_dif2.apply(init_weights2)
        self.layer7_dif2.apply(init_weights2)
        self.layer8_dif2.apply(init_weights2)


    def forward(self, x1, x2, mode='eval'):
        out_com_1 = x1
        out_com_2 = x2
        out_dif_1 = x1
        out_dif_2 = x2

        for layer_com_1 in [self.layer1_com1, self.layer2_com1, self.layer3_com1, self.layer4_com1, self.layer5_com1, self.layer6_com1, self.layer7_com1, self.layer8_com1]:
            out_com_1 = layer_com_1(out_com_1)

        for layer_com_2 in [self.layer1_com2, self.layer2_com2, self.layer3_com2, self.layer4_com2, self.layer5_com1, self.layer6_com1, self.layer7_com1, self.layer8_com1]:
            out_com_2 = layer_com_2(out_com_2)

        for layer_dif_1 in [self.layer1_com1, self.layer2_com1, self.layer3_com1, self.layer4_com1, self.layer5_dif1, self.layer6_dif1, self.layer7_dif1, self.layer8_dif1]:
            out_dif_1 = layer_dif_1(out_dif_1)

        for layer_dif_2 in [self.layer1_com2, self.layer2_com2, self.layer3_com2, self.layer4_com2, self.layer5_dif2, self.layer6_dif2, self.layer7_dif2, self.layer8_dif2]:
            out_dif_2 = layer_dif_2(out_dif_2)


        desc_raw_1 = self.layer9_com1(out_com_1).squeeze()
        desc_1 = desc_l2norm(desc_raw_1)

        desc_raw_2 = self.layer9_com1(out_com_2).squeeze()
        desc_2 = desc_l2norm(desc_raw_2)

        if mode == 'train':
            return desc_1, desc_raw_1, out_com_1, out_dif_1, desc_2, desc_raw_2, out_com_2, out_dif_2
        elif mode == 'eval':
            return desc_1, out_com_1, out_dif_1, desc_2, out_com_2, out_dif_2




class feature_KGL_metric(nn.Module):


    def __init__(self, is_bias=True, is_bias_FRN=True, drop_rate=0.2):
        super(feature_KGL_metric, self).__init__()
        self.drop_rate = drop_rate

        self.dif_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=is_bias),
            FRN(256, is_bias=is_bias_FRN),
            TLU(256),
            nn.AdaptiveAvgPool2d(output_size=1),
        )

        self.metric_net = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        ###-------------
        self.dif_conv.apply(init_weights2)
        self.metric_net.apply(init_weights2)


    def forward(self, x1, x2):
        diff_feature = torch.abs(x1 - x2)
        diff_feature = self.dif_conv(diff_feature).squeeze()
        diff_feature = self.metric_net(diff_feature).squeeze()
        return diff_feature