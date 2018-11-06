# -------------------------------------------------------------------------------------------------------------------------------
# 
# ConvBlock Module :
# 
#   METHODS ---- 
#   forward                     : public    - Run nn.Module
#   widenAsNext                 : private   - to be used at child nodes for the node widened via widenAsCurrent
#   widenAsCurrent              : private   - to be used at the current node to be widened 'conv'
# 
#   VARIABLES ----     
#   padval                      : 0
#   c                           : nn.Conv2d(in_channels,out_channels,kernel_size,padding=padval)
#   b                           : nn.BatchNorm2d(out_channels)
#   a                           : activation 
# 
# 
# MaxPool Module :
# 
#   METHODS ---- 
#   forward                     : public    - Run nn.Module
# 
#   VARIABLES ----
#   kernel_size                 : kernel for maxpool selected [2,3,4] in NASGraph->applyMorph
#   m                           : maxpool layer
# 
# Merge Module :
# 
#   METHODS ---- 
#   forward                     : public    - Run nn.Module
#
#   VARIABLES ----
#   NONE
# 
# 
# Add Module :
# 
#   METHODS ---- 
#   forward                     : public    - Run nn.Module <- dict method to override kwargs
# 
#   VARIABLES ----
#   NONE
# 
# 
# Version : 1.2
# 
# Changehistory : (from Nov 5th 2018)  ./change.txt
# 
# Author : Mudit 
# 
# -------------------------------------------------------------------------------------------------------------------------------




import torch.nn as nn

from NASGraphBaseClasses import Node, Graph


class ConvBlock(nn.Module,Node) :
    def __init__(self, in_channels, out_channels, kernel_size, activation='relu' ,stride=1, padding=False, dilation=1, groups=1, bias=True) :
        # missing bn params : eps=1e-05, momentum=0.1, affine=True, track_running_stats=True

        super(ConvBlock,self).__init__()
        Node.__init__(self,'conv')
        
        padval = 0
        if padding :
            padval = int(kernel_size/2)

        self.c = nn.Conv2d(in_channels,out_channels,kernel_size,padding=padval)
        self.b = nn.BatchNorm2d(out_channels)
        
        if activation == 'relu' :
            self.a = nn.ReLU()
        elif activation == 'tanh' :
            self.a = nn.Tanh()
        #add others if you want.
    
    def forward(self,x) :
        self.outputTensor = self.a(self.b(self.c(x)))
        return self.outputTensor
    
    def widenAsNext(self,in_channels):
        #when widen as child
        next_layer = self.c
        orig_channels = next_layer.in_channels
        weights = [next_layer.weight.data]
        next_layer = nn.Conv2d(in_channels,
                               next_layer.out_channels,
                               kernel_size=next_layer.kernel_size
                              )
        
        next_layer.weight.data[:,0:orig_channels] = weights[0]
        self.c = next_layer
        
        
    def widenAsCurrent(self,factor):
        #affects channels only, no effect on kernel size.
        #when widen as current node
        '''
            @inputs : 
            widening factor : 2,4

            Example :
                a1 = Conv
                a1.widen(2)
        '''     
        current_layer = self.c
        bn_current = self.b         
            
        orig_channels = current_layer.out_channels
        
        weights = [current_layer.weight.data]  
        bn_weights = [bn_current.weight.data]
        
        current_layer = nn.Conv2d(current_layer.in_channels,
                                  orig_channels*factor,
                                  kernel_size=current_layer.kernel_size,
                                  stride=current_layer.stride)       
        bn_current = nn.BatchNorm2d(current_layer.out_channels)

        current_layer.weight.data[0:orig_channels,:] = weights[0]
        bn_current.weight.data[:orig_channels] = bn_weights[0][:]

        self.c = current_layer
        self.b = bn_current





class MaxPool(nn.Module,Node) :
    def __init__(self, kernel_size):
        super(MaxPool,self).__init__()
        Node.__init__(self,type='maxpool')
        
        self.m = nn.MaxPool2d(kernel_size)
        
    def forward(self, x) :
        self.outputTensor = self.m(x)
        return self.outputTensor
    
    def widen(self, kernel_size) :
        self.m = nn.MaxPool2d(kernel_size)




class Merge(nn.Module,Node) :
    def __init__(self) :
        super(Merge,self).__init__()
        Node.__init__(self,'merge')
        
        
    def forward(self,x1,x2) :
        self.outputTensor = torch.cat([x1,x2],1)
        return self.outputTensor




class Add (nn.Module,Node) :
    def __init__(self,) :
        super(Add,self).__init__()
        Node.__init__(self,'add')
        
        
    def forward(self,**x) :
#         assert x1.shape == x2.shape , 'Size mismatch while Adding tensors'
        ke = x.keys()
        t = x[ke[0]]
        for i in ke[1:] :
            t = t+x[i]

        self.outputTensor = t
        return self.outputTensor
    
# ------------------------ SAMPLE OF ADD USAGE ------------------------
# x = torch.randn((1,2))
# y = torch.randn((1,2))
# z = torch.randn((1,2))
# a = Add()
# d = d={'a'+str(i):x_x for i,x_x in enumerate([x,y,z])}
# a(**d).shape
        
    