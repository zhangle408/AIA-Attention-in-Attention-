import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.nn.init as init
import torchvision

def get_feature_cood(feature_width):
    array=2*((np.arange(feature_width)*1.0)/(feature_width-1)) - 1
    return torch.FloatTensor(np.float32(array)).view(1,1,-1,1)
    
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

def get_1d_dct(i, freq, L):
    result=math.cos(math.pi * freq *(i+0.5)/L) / math.sqrt(L*1.0)
    if freq ==0:
        return result
    else:
        return result * math.sqrt(2.0)

def dct_filters(in_channels, num_frequency, kernel_size):
    c_part = in_channels//num_frequency
    oneD_dct_weight=np.zeros((1,in_channels, kernel_size, 1), dtype=np.float32)#c,h,w
    for i in range(num_frequency):
        for x in range (kernel_size):
            oneD_dct_weight[:,i*c_part : (i+1)*c_part, x, : ]=get_1d_dct(x,i, kernel_size)
    if in_channels>(c_part*num_frequency):
        for x in range (kernel_size):
            oneD_dct_weight[:,c_part*num_frequency :, x, : ]=get_1d_dct(x,num_frequency, kernel_size)
    return torch.FloatTensor(oneD_dct_weight)

'''class DCT_attention(nn.Module):
    def __init__(self, in_channels, num_frequency,kernel_size,reduction):
        super(DCT_attention, self).__init__()
        self.oneD_dct_weight=nn.Parameter(dct_filters(in_channels, num_frequency, kernel_size), requires_grad=False)
        mip = max(8, in_channels // reduction)
        self.SE=nn.Sequential(
            nn.Conv2d(in_channels,mip,kernel_size=1, stride=1, padding=0,bias=False),
            #nn.ReLU(inplace=True),
            h_swish(),
            nn.Conv2d(mip,in_channels,kernel_size=1, stride=1, padding=0,bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):#x_size={B,C,h,w}
        y = torch.sum(x * self.oneD_dct_weight, dim=[2,3],keepdim=True) #y_size:[B,C,1,1]
        attention=self.SE(y)
        return x*attention.expand_as(x)'''

        
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, num_frequency,kernel_size, reduction_coord=32, reduction_dct=32):
        super(CoordAtt, self).__init__()
        self.oneD_dct_weight=nn.Parameter(dct_filters(inp, num_frequency, kernel_size), requires_grad=False)
        mip_coord = max(8, inp // reduction_coord)
        mip_dct = max(8, inp // reduction_dct)
        self.dct_conv=nn.Conv2d(inp,mip_dct,kernel_size=1, stride=1, padding=0,bias=False)
        self.dct_act=h_swish()
        self.dct_conv_h=nn.Conv2d(mip_dct,inp,kernel_size=1, stride=1, padding=0,bias=False)
        self.dct_conv_w=nn.Conv2d(mip_dct,inp,kernel_size=1, stride=1, padding=0,bias=False)
        #self.dct_atten_h=DCT_attention(inp, num_frequency,kernel_size,reduction)
        #self.dct_atten_w=DCT_attention(inp, num_frequency,kernel_size,reduction)
        self.cood = get_feature_cood(kernel_size) # SIZE[1,1,WIDTH,1]
        
        self.conv1 = nn.Conv2d(inp+1, mip_coord, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip_coord)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip_coord, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip_coord, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h=torch.sum(x*self.oneD_dct_weight.permute(0,1,3,2), dim=[3], keepdim=True)
        x_w=torch.sum(x*self.oneD_dct_weight,                  dim=[2], keepdim=True).permute(0, 1, 3, 2)
        
        dct_atten_h = torch.sum(x_h * self.oneD_dct_weight, dim=[2,3],keepdim=True) #y_size:[B,C,1,1]
        #print('dct_atten_h.shape',dct_atten_h.shape)
        dct_atten_w = torch.sum(x_w * self.oneD_dct_weight, dim=[2,3],keepdim=True) #y_size:[B,C,1,1]
        dct_atten=torch.cat([dct_atten_h, dct_atten_w], dim=2)
        dct_atten=self.dct_act(self.dct_conv(dct_atten))
        #print('dct_atten.shape',dct_atten.shape)
        dct_atten_h, dct_atten_w = torch.split(dct_atten, [1, 1], dim=2)
        dct_atten_h=self.dct_conv_h(dct_atten_h).sigmoid()
        dct_atten_w=self.dct_conv_w(dct_atten_w).sigmoid()
        x_h=x_h*dct_atten_h.expand_as(x_h)
        x_w=x_w*dct_atten_w.expand_as(x_w)
        
        #x_h=self.dct_atten_h(x_h)
        #x_w=self.dct_atten_h(x_w)
        
        cood=self.cood.expand(x_h.size(0),1,x_h.size(2),x_h.size(3)).to(x_h.device)
        x_h_cood = torch.cat([x_h, cood], dim=1) #size[B, c+1, h,1]
        x_w_cood = torch.cat([x_w, cood], dim=1)#size[B, c+1, w,1]

        y = torch.cat([x_h_cood, x_w_cood], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out