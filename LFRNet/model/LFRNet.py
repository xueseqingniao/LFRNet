 # coding=utf-8
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.conv import Conv2d
#from model.resattention import res_cbam
import torchvision.models as models
#from model.res2fg import res2net
import torch.nn.functional as F
import math
# Low level
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)#dim=1是因为传入数据是多加了一维unsqueeze(0)
        x=max_out
        x = self.conv1(x)
        return self.sigmoid(x)
class GC(nn.Module):
    def __init__(self,in_channels):
        super(GC,self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_channels,in_channels),requires_grad=True)
        self.reset_para()
    def reset_para(self):
        stdv=1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
    def forward(self,x):
        batch_size = x.size(0)
        channel = x.size(1)
        #print(channel)
        g_x = x.view(batch_size, channel, -1)#[bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)#[b,wh,c]
        theta_x = x.view(batch_size, channel, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = x.view(batch_size, channel, -1)
        
        f = torch.matmul(theta_x, phi_x)

        adj = F.softmax(f, dim=-1)
        #print(g_x.size())
        #print(self.weight.size())
        support = torch.matmul(g_x,self.weight)
        y = torch.matmul(adj,support)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, channel, *x.size()[2:])
        return y


class GA(nn.Module):
    def __init__(self,in_channels):
        super(GA,self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Conv2d(2*in_channels,in_channels,1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1   = nn.Conv2d(in_channels, in_channels // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_channels // 16, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x1,x2):
        x = self.conv(torch.cat((x1,self.upsample(x2)),1))
        avg_out = self.sigmoid(self.fc2(self.relu1(self.fc1(self.avgpool(x)))))
        out = x*avg_out+x
        return out

#mid level
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class SC(nn.Module):# Pyramid Feature Module
    def __init__(self,in_channels):
        super(SC,self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(in_channels, in_channels, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=7, dilation=7)
        )
        self.ba0 = nn.BatchNorm2d(in_channels)
        self.relu0 = nn.ReLU()

        self.ba1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()

        self.ba2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU()

        self.ba3 = nn.BatchNorm2d(in_channels)
        self.relu3 = nn.ReLU()

        self.conv = nn.Conv2d(4*in_channels,4*in_channels,kernel_size=3,padding=1)
        self.ba = nn.BatchNorm2d(4*in_channels)
        self.relu = nn.ReLU()

    def forward(self,x):
        x0,x1,x2,x3 = torch.split(x,x.size()[1]//4,dim=1)
        y0 = self.relu0(self.ba0(self.branch0(x0)))
        y1 = self.relu1(self.ba1(self.branch1(x1)))
        y2 = self.relu2(self.ba2(self.branch2(x2)))
        y3 = self.relu3(self.ba3(self.branch0(x3)))

        y = torch.cat((y0,y1,y2,y3),1)
        y = self.relu(self.ba(self.conv(y)))
        return y
class MutualAttention(nn.Module):
    def __init__(self,in_channels):
        super(MutualAttention,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.fc1   = nn.Conv2d(2*in_channels, in_channels // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_channels // 16, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x1,x2):
        x2 = self.upsample(x2)
        x_c = torch.cat((x1,x2),1)
        avg_out = self.sigmoid(self.fc2(self.relu1(self.fc1(self.avgpool(x_c)))))
        y1 = x1*avg_out
        y2 = x2*avg_out
        y = y1+y2
        return y

#high level
class Edge(nn.Module):
    def __init__(self,in_channels):
        super(Edge,self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)
        self.sigmoid = nn.Sigmoid()
        self.conv4 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)

        self.ba1 = nn.BatchNorm2d(in_channels)
        self.ba2 = nn.BatchNorm2d(in_channels)
        self.ba3 = nn.BatchNorm2d(in_channels)
        self.ba4 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
    def forward(self,x1,x2):
        x2 = self.upsample(x2)
        x1 = self.relu(self.ba1(self.conv1(x1)))
        x2 = self.relu(self.ba2(self.conv2(x2)))
        x_c = self.relu(self.ba3(self.conv3(torch.cat((x1,x2),1))))
        x_max,_ =torch.max(x_c,dim=1,keepdim=True)
        x_spatial = self.sigmoid(x_max)
        x3 = x_c*x_spatial+x_c
        y = self.relu(self.ba4(self.conv4(x3)))
        return y

#Fuse
class FF(nn.Module):
    def __init__(self,in_channels):
        super(FF,self).__init__()
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1   = nn.Conv2d(2*in_channels, in_channels // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_channels // 8, 2*in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv3 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)
    def forward(self,x1,x2,e):
        x2 = self.upsample(x2)
        x1 = self.conv1(x1+x1*e)
        x2 = self.conv2(x2+x2*e)
        #x_m = x1*x2
        #x1 = x1+x_m
        #x2 = x2+x_m
        x_c = torch.cat((x1,x2),1)
        #avg_out = self.sigmoid(self.fc2(self.relu1(self.fc1(self.avgpool(x_c)))))
        #y = x_c*avg_out
        y = self.conv3(x_c)

        return y
class CBR(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(CBR,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.Ba = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.relu(self.Ba(self.conv(x)))

        return out

class EG(nn.Module):
    def __init__(self,in_channels):
        super(EG,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.ba = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        #self.conv3 = nn.Conv2d(in_channels,1,1)
    def forward(self,x):
        out = self.relu(self.ba(self.conv2(self.conv1(x))))
        return out
class LFRNet(nn.Module):#输入三通道
    def __init__(self, in_channels):
        super(LFRNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        
        # ************************* Encoder ***************************
        # input conv3*3,64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#计算得到112.5 但取112 向下取整
        # Extract Features
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.conv_d = nn.Conv2d(512,512,kernel_size=3,padding=1)
        # ************************* Decoder ***************************
        
        # ************************* Feature Map Upsample ***************************
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear')
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear')
        

        self.cbr0 = CBR(64,64)
        self.cbr1 = CBR(64,64)
        self.cbr2 = CBR(128,64)
        self.cbr3 = CBR(256,64)
        self.cbr4 = CBR(512,64)
        self.cbr5 = CBR(512,64)
        

        self.GR5 = GC(128)
        self.GR4 = GC(192)
        self.GR3 = GC(192)
        self.GR5_1 = GC(128)
        self.GR4_1 = GC(192)
        self.GR3_1 = GC(192)

        self.conv1x1_5 = nn.Conv2d(128,128,kernel_size=1)
        self.conv1x1_4 = nn.Conv2d(192,128,kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(192,128,kernel_size=1)
        self.GA4 = GA(128)
        self.GA3 = GA(128)

        
        self.SC3 = SC(48)
        self.SC2 = SC(48)
        self.SC1 = SC(48)

        self.Ma2 = MutualAttention(192)
        self.Ma1 = MutualAttention(192)
        self.conv_m = nn.Conv2d(192,128,kernel_size=3,padding=1)
        self.edge_s1 = EG(128)
        self.edge_s2 = EG(192)
        self.edge_s3 = EG(192)
        self.edge_s  = nn.Conv2d(128,1,kernel_size=1)
        self.conv_edge = nn.Conv2d(512,128,kernel_size=3,padding=1)
       
        self.FF = FF(128)
        
        self.sup_L = nn.Conv2d(128,1,kernel_size=1)
        self.sup_M = nn.Conv2d(128,1,kernel_size=1)
        

        self.sup_LM = nn.Conv2d(128,1,kernel_size=1)
        self.sup_out = nn.Conv2d(128,1,kernel_size=1)

        self.e1_sup = nn.Conv2d(128,1,kernel_size=1)
        self.e2_sup = nn.Conv2d(192,1,kernel_size=1)
        self.e3_sup = nn.Conv2d(192,1,kernel_size=1)
    def forward(self, x):
        tx = self.conv1(x)
        tx = self.bn1(tx)
        f0 = self.relu(tx)
        tx = self.maxpool(f0)
        # Extract Features
        f1 = self.encoder1(tx)
        f2 = self.encoder2(f1)
        f3 = self.encoder3(f2)
        f4 = self.encoder4(f3)
        f5 = self.downsample(self.conv_d(f4))

        
        f0 = self.cbr0(f0)
        f1 = self.cbr1(f1)
        f2 = self.cbr2(f2)
        f3 = self.cbr3(f3)
        f4 = self.cbr4(f4)
        f5 = self.cbr5(f5)
        

        f_g5 = self.conv1x1_5(self.GR5_1(self.GR5(torch.cat((f5,self.downsample(f4)),1))))
        f_g4 = self.conv1x1_4(self.GR4_1(self.GR4(torch.cat((f4,self.downsample(f3),self.upsample1(f5)),1))))
        f_g3 = self.conv1x1_3(self.GR3_1(self.GR3(torch.cat((f3,self.downsample(f2),self.upsample1(f4)),1))))
        
        f_l4 = self.GA4(f_g4,f_g5)
        f_L = self.GA3(f_g3,f_l4)

        
        f_p3 = self.SC3(torch.cat((f3,self.upsample1(f4),self.downsample(f2)),1))
        f_p2 = self.SC2(torch.cat((f2,self.upsample1(f3),self.downsample(f1)),1))
        f_p1 = self.SC1(torch.cat((f1,self.upsample1(f2),self.downsample(f0)),1))

        f_ma2 = self.Ma2(f_p2,f_p3)
        f_M = self.conv_m(self.Ma1(f_p1,f_ma2))
        
        
        S_l = self.upsample3(self.sup_L(f_L))
        S_m = self.upsample1(self.sup_M(f_M))
        

        E1 = self.edge_s1(f_M)
        E2 = self.upsample1(self.edge_s2(f_ma2))
        E3 = self.upsample2(self.edge_s3(f_p3))
        E = self.conv_edge(torch.cat((E1,E2,E3),1))
        E_sup = self.edge_s(E)
        E_s = F.sigmoid(E)

        f_LM = self.FF(f_M,f_L,E_s)
        f_out = self.upsample1(f_LM)
        S_lm = self.upsample1(self.sup_LM(f_LM))
        S_out = self.sup_out(f_out)

        e1 = self.upsample1(self.e1_sup(E1))
        e2 = self.upsample1(self.e2_sup(E2))
        e3 = self.upsample1(self.e3_sup(E3))

        return F.sigmoid(S_out),F.sigmoid(S_l),F.sigmoid(S_m),F.sigmoid(S_lm),F.sigmoid(e1),F.sigmoid(e2),F.sigmoid(e3),F.sigmoid(self.upsample1(E_sup))


