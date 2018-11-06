from torchsummary import summary
from block import ConvBlock,Merge,Add,MaxPool
import torch.nn as nn
import torch
from pytorchmodelviz import viz

class Model(nn.Module) :
    def __init__(self) :
        super(Model,self).__init__()
        
        self.c1 = ConvBlock(1,32,3)
        self.c2 = ConvBlock(32,32,3)
        self.c3 = ConvBlock(32,32,3)
        self.c4 = ConvBlock(32,32,3)
        self.fc = nn.Linear(3200,1000)
        self.fc2 = nn.Linear(1000,10)

        
    def forward(self,x) :
        out = self.c1(x)
        out1 = self.c2(out)
        out2 = self.c3(out)
        out3 = self.c4(out)
#         add = Add()
        p = [out1,out2,out3]
        d={'a'+str(i):x_x for i,x_x in enumerate(p)}
        out = a(**d)
        print out.shape
        out = out.view(-1,3200)
        out = self.fc2(self.fc(out))
        return out

x = torch.randn((1,1,14,14))
m = Model()
m(x)


summary(m,(1,14,14))

viz((2,1,14,14),m)