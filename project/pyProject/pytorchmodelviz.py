# --------------------------------------------------------------
#  	Visualization via PyTorch's torchviz 
# 	Call viz(inputs,model)
# 	@inputs : sample input (batch,channel,img_x,img_y)
# 	@model 	: pytorch nn.Module model 
# --------------------------------------------------------------

import torch
import torchviz

def viz(inputs,model) :
    x = torch.randn(inputs)
    y = model(x)
    f = torchviz.make_dot(y , params = dict(model.named_parameters()))
    return f