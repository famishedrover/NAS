# --------------------------------------------------------------
# DEPENDECIES 	: 	torch, numpy, matplotlib, graphviz,
# 					math, random, torchsummary
# 
# COMPLETED 	: 	NasGraph Architecture Genereation 	 
#					1. 	ConvLayer
# 					2. 	WidenConv
# 					3. 	MaxPool Layer 
# 					4. 	Skip Connections
# 
# PARAMETERS 	: 	Graph Init Params (main.py)
# 					Morph Operation Distribution (NASGraph.py)
# 					Kernel Selection (NASGraph.py)
# 					Default Channel ConvLayer (NASGraph.py)
# 					Conv Padding Probability (block.py)
# Version : 1.2
# 
# Changehistory : (from Nov 5th 2018)  ./change.txt
# 
# Author : Mudit 
# 
# --------------------------------------------------------------

# IMPORTS
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt 

from torchsummary import summary

import torchviz

import networkx as nx
import matplotlib.pyplot as plt

from graphviz import Digraph

import math

from NASGraph import NASGraph

import random

# --------------------------------------------------------------
# RANDOM SEED FOR random pkg, 
# TODO Add for torch and numpy 
random.seed('Na')


BATCH = 4
BEGIN_IN_CHANNELS = 3 
BEGIN_OUT_CHANNELS = 16
BEGIN_KERNEL_SIZE = 3
BEGIN_PADDING = False
BEGIN_BLOCK = 'conv'
operationdist = [0.3,0.2,0.2,0.15,0.15]
# --------------------------------------------------------------
# INIT GRAPH
# (4, 3, 32, 32)
gr = NASGraph(([BATCH,3, 32, 32]),operationdist)
p = {'block':BEGIN_BLOCK,'in_channels':BEGIN_IN_CHANNELS,'out_channels':BEGIN_OUT_CHANNELS,'kernel_size':BEGIN_KERNEL_SIZE,'padding':BEGIN_PADDING}
gr.addInGraph(**p)

for _ in range(10):
    gr.applyMorph()
x = torch.randn((BATCH,gr.nodes[gr.begin].c.in_channels,32,32))
# --------------------------------------------------------------
# MORPHISM

ITERATIONS = 27
for _ in range(ITERATIONS):
	print _ , ' :',
	status = gr.applyMorph(log=False)
	print ' STATUS :',status
#     gr.plotGraph2(filename=str(_)+'.gv')
# print 'TopSort:',
gr.order = gr.topologicalSort()
# gr.topologicalSort(name=True)
# gr.createModel()


# gr.compatCheck(log=True)

print '\n','-'*20
for each in gr.nodes :
    print gr.nodes[each].name , gr.nodes[each].input, gr.nodes[each].output


# gr.topologicalSort(name=True)
gr.compatCheck()
gr.applyNecessaryAddNodes()
# gr.compatCheck()
# gr.showAllNodes()
gr.plotGraph()


# --------------------------------------------------------------
# Testing FORWARD

print 'SAMPLE testing forward'
print 'INPUT:',x.shape
t = gr(x)
print 'OUTPUT:',t.shape




# --------------------------------------------------------------
# PARAMS OF MODEL
print '-'*20

gr.createModel()

# print 'NASGRAPH PARAMS'
# for z,v in gr.named_parameters() :
# 	print z


# --------------------------------------------------------------
# CREATING DUMMY FINAL WITH LINEAR LAYERS

class Model(nn.Module) :
    def __init__(self,nasgr,nasout) :
        super(Model,self).__init__()
        self.nasgr = nasgr
        out = nasout[1]*nasout[2]*nasout[3]
        self.fc1 = nn.Linear(out,16)
        self.fc2 = nn.Linear(16,10)
    	self.batch_size = nasout[0]
    def forward(self,x) :
    	out = self.nasgr(x)
    	out = out.view(self.batch_size,-1)
    	out = self.fc1(out)
    	out = self.fc2(out)

    	return out


net = Model(gr,t.shape)

# --------------------------------------------------------------
# LOOKING AT net params
print 'FINAL NET PARAMS'
for z,v in net.named_parameters() :
	print z

# TO CUDA 
if torch.cuda.is_available() :
	net = net.cuda()



# print id(net.nasgr.nodes[net.nasgr.begin]) ,' : ', id(net.nasgr.nodesStr[net.nasgr.nodes[net.nasgr.begin].type + str(net.nasgr.nodes[net.nasgr.begin].name)])

# --------------------------------------------------------------
# TRAINING FOR 2 EPOCHS 


from trainmodel import Train ,Test
Train(net,1)
Test(net)


