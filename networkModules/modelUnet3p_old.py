# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from networkModules.conv_modules_unet3p import unetConv2, BlurPool2D, MaxBlurPool2d, SEBlock
from networkModules.init_weights import init_weights
import sys
'''
    UNet 3+
'''
class UNet_3Plus(nn.Module):

    def __init__(self, config, in_channels=1, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_3Plus, self).__init__()
        self.is_deconv = is_deconv
        if config["input_img_type"] == "rgb":
            in_channels = 3    
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.kernel_size = config["kernel_size"]
        self.ch = config["channel"]
        n_classes = config["num_classes"]
        self.dropout = nn.Dropout2d(p=config["dropout"])
        self.useMaxBPool = config["use_maxblurpool"]

        self.samGuided = config["SAM_Guided"]
        self.advSamGuided = config["advancedSAM_Guided"]
        self.reduction_ratio = 16

        self.dropoutFlag = False

        # self.ch, original paper uses channel size of 64, while we use 16
        # uses relu activation by default
        filters = [self.ch, self.ch * 2, self.ch * 4, self.ch * 8, self.ch * 16]
        #filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, ks=self.kernel_size)
        if self.useMaxBPool:
            self.maxpool1 = MaxBlurPool2d(kernel_size=2)
        else:
            self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, ks=self.kernel_size)
        if self.useMaxBPool:
            self.maxpool2 = MaxBlurPool2d(kernel_size=2)
        else:
            self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm, ks=self.kernel_size)
        if self.useMaxBPool:
            self.maxpool3 = MaxBlurPool2d(kernel_size=2)    
        else:
            self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm, ks=self.kernel_size)
        if self.useMaxBPool:
            self.maxpool4 = MaxBlurPool2d(kernel_size=2)
        else:
            self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm, ks=self.kernel_size)

        self.samMaxPool = MaxBlurPool2d(kernel_size=2)


        #deconv for SAM for h1 and h2
        self.deconv_for_h1 = nn.ConvTranspose2d(
            in_channels=256,  
            out_channels=16,  
            kernel_size=8,
            stride=4,
            padding=2
            )
        
        self.deconv_for_h2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1
            )
        
        # conv to reduce feature channels
        self.conv_for_h3 =  nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)

        self.conv_for_h4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)


        # for h5 and SAM fusion
        self.unet_channels_h5 = filters[4]
        self.fused_channels_h5 = self.unet_channels_h5 + 256
        self.squeeze_excite_h5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.fused_channels_h5, self.fused_channels_h5 // self.reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fused_channels_h5 // self.reduction_ratio, self.unet_channels_h5, kernel_size=1),  # Output channels for gating
            nn.Sigmoid()  # Gating mechanism
        )
        self.cross_attention_h5 = nn.MultiheadAttention(embed_dim=self.unet_channels_h5, num_heads=1, batch_first=True)

        # for h4 and SAM fusion
        self.unet_channels_h4 = filters[3]
        self.fused_channels_h4 = self.unet_channels_h4 + 128
        self.squeeze_excite_h4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.fused_channels_h4, self.fused_channels_h4 // self.reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fused_channels_h4 // self.reduction_ratio, self.unet_channels_h4, kernel_size=1),  # Output channels for gating
            nn.Sigmoid()  # Gating mechanism
        )
        # for h3 and SAM fusion
        self.unet_channels_h3 = filters[2]
        self.fused_channels_h3 = self.unet_channels_h3 + 64
        self.squeeze_excite_h3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.fused_channels_h3, self.fused_channels_h3 // self.reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fused_channels_h3 // self.reduction_ratio, self.unet_channels_h3, kernel_size=1),  # Output channels for gating
            nn.Sigmoid()  # Gating mechanism
        )
        # for h2 and SAM fusion
        self.unet_channels_h2 = filters[1]
        self.fused_channels_h2 = self.unet_channels_h2 + 32
        self.squeeze_excite_h2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.fused_channels_h2, self.fused_channels_h2 // self.reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fused_channels_h2 // self.reduction_ratio, self.unet_channels_h2, kernel_size=1),  # Output channels for gating
            nn.Sigmoid()  # Gating mechanism
        )
        # for h1 and SAM fusion
        self.unet_channels_h1 = filters[0]
        self.fused_channels_h1 = self.unet_channels_h1 + 16
        self.squeeze_excite_h1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.fused_channels_h1, self.fused_channels_h1 // self.reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fused_channels_h1 // self.reduction_ratio, self.unet_channels_h1, kernel_size=1),  # Output channels for gating
            nn.Sigmoid()  # Gating mechanism
        )

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, self.kernel_size, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, self.kernel_size, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, self.kernel_size, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, self.kernel_size, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, self.kernel_size, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, self.kernel_size, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, self.kernel_size, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, self.kernel_size, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, self.kernel_size, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, self.kernel_size, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, self.kernel_size, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, self.kernel_size, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, self.kernel_size, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, self.kernel_size, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, self.kernel_size, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, self.kernel_size, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, self.kernel_size, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, self.kernel_size, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, self.kernel_size, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, self.kernel_size, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, self.kernel_size, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, self.kernel_size, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, self.kernel_size, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, self.kernel_size, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, self.kernel_size, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def setdropoutFlag(self, flag):
        self.dropoutFlag = flag

    def guidedForward(self, input, SAM_Enc):
            
        ## -------------Encoder-------------
        h1 = self.conv1(input)  # h1->32*256*256

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->64*128*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->128*64*64

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->256*16*16


        

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->512*16*16

        #return h1, h2, h3, h4, hd5

        #print("h1", h1.shape)
        #print("h2", h2.shape)
        #print("h3", h3.shape)
        #print("h4", h4.shape)
        #print("hd5", hd5.shape)
        #print("SAM_Enc", SAM_Enc.shape)

        samcd = self.samMaxPool(SAM_Enc)
        samcd = self.samMaxPool(samcd)
        #print("SAM_cd", samcd.shape)
        # concat shapes of 1, 256, 16, 16 with 1, 256, 64, 64
        combined = torch.cat((hd5, samcd), 1)
        #print("Combined=",combined.shape)

        gatingWeights = self.squeeze_excite_h5(combined)
        #print("GatingWeights=",gatingWeights.shape)

        gatedFeatures = gatingWeights * hd5
        #print("GatedFeatures=",gatedFeatures.shape)

        batch_size, channels, height, width = gatedFeatures.shape
        gated_unet_features_flat = gatedFeatures.view(batch_size, channels, -1).permute(0, 2, 1)
        unet_features_flat = gatedFeatures.view(batch_size, channels, -1).permute(0, 2, 1)

        # Apply cross-attention
        attention_output, _ = self.cross_attention_h5(query=gated_unet_features_flat, key=unet_features_flat, value=unet_features_flat)
        attention_output = attention_output.permute(0, 2, 1).view(batch_size, channels, height, width)

        #print("AttentionOutput=",attention_output.shape)

        hd5 = attention_output
        

        # dropout
        hd5 = self.dropout(hd5)

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        return torch.sigmoid(d1)
    

    def advGuidedForward(self, input, SAM_Enc):

        #SAM_Enc -> 256, 64, 64
        
        #print("SAM_Enc=",SAM_Enc.shape)
        ## -------------Encoder-------------
        h1 = self.conv1(input)  # h1->32*256*256
        upsampled_sam_h1 = self.deconv_for_h1(SAM_Enc)
        #print("upsampled_sam_h1=",upsampled_sam_h1.shape)
        combined = torch.cat((h1, upsampled_sam_h1), 1)
        gatingWeights = self.squeeze_excite_h1(combined)
        h1 = gatingWeights * h1
        



        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->64*128*128

        upsampled_sam_h2 = self.deconv_for_h2(SAM_Enc)
        #print("upsampled_sam_h2=",upsampled_sam_h2.shape)
        combined = torch.cat((h2, upsampled_sam_h2), 1)
        gatingWeights = self.squeeze_excite_h2(combined)
        h2 = gatingWeights * h2


        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->128*64*64

        samh3 = self.conv_for_h3(SAM_Enc)
        #print("samh3=",samh3.shape)
        combined = torch.cat((h3, samh3), 1)
        gatingWeights = self.squeeze_excite_h3(combined)
        h3 = gatingWeights * h3


        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->256*32*32

        samh4 = self.conv_for_h4(SAM_Enc)
        samh4 = self.samMaxPool(samh4)
        combined = torch.cat((h4, samh4), 1)
        gatingWeights = self.squeeze_excite_h4(combined)
        h4 = gatingWeights * h4

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->512*16*16

        #return h1, h2, h3, h4, hd5

        '''print("h1", h1.shape)
        print("h2", h2.shape)
        print("h3", h3.shape)
        print("h4", h4.shape)
        print("hd5", hd5.shape)
        print("SAM_Enc", SAM_Enc.shape)'''

        samcd = self.samMaxPool(SAM_Enc)
        samcd = self.samMaxPool(samcd)
        #print("SAM_cd", samcd.shape)
        # concat shapes of 1, 256, 16, 16 with 1, 256, 64, 64
        combined = torch.cat((hd5, samcd), 1)
        #print("Combined=",combined.shape)

        gatingWeights = self.squeeze_excite_h5(combined)
        #print("GatingWeights=",gatingWeights.shape)

        gatedFeatures = gatingWeights * hd5
        #print("GatedFeatures=",gatedFeatures.shape)

        batch_size, channels, height, width = gatedFeatures.shape
        gated_unet_features_flat = gatedFeatures.view(batch_size, channels, -1).permute(0, 2, 1)
        unet_features_flat = gatedFeatures.view(batch_size, channels, -1).permute(0, 2, 1)

        # Apply cross-attention
        attention_output, _ = self.cross_attention_h5(query=gated_unet_features_flat, key=unet_features_flat, value=unet_features_flat)
        attention_output = attention_output.permute(0, 2, 1).view(batch_size, channels, height, width)

        #print("AttentionOutput=",attention_output.shape)



        #sys.exit(0)
        hd5 = attention_output




        

        # dropout
        hd5 = self.dropout(hd5)

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        return torch.sigmoid(d1)

    def forward(self, inputs):

        input, SAM_Enc = inputs
        #SAM_Enc = SAM_Enc.squeeze(0)

        #print("SAM_Enc=",SAM_Enc.shape)
        #print("Input=",input.shape)
        if self.advSamGuided:
            return self.advGuidedForward(input, SAM_Enc)
        else:
            return self.guidedForward(input, SAM_Enc)

        