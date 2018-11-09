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
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=2)
# classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# GET DATA AND TRAINING CODE AS IMPORTA FROM TRAINMODE.PY

from trainmodel import Train,Test,ValidationPerformance,argMax,trainloader,testloader, addLinearLayers, removeLinearLayers



# some start model_0
#BATCH = 4
# INIT GRAPH
#operationdist = [0.3,0.2,0.2,0.15,0.15]
# (4, 3, 32, 32)
#model_0 = NASGraph(([BATCH,3, 32, 32]),operationdist)
#n_steps = 10
#n_neigh = 4
#n_nm = 3
#epoch_neigh = 30
#epoch_final = 50
#lr_start = 0.01
#lr_end = 0.001  # annealed via SGDR

# assuming the input model is NASGraph object, 
# we need to add linear layers at the end 
# TODO : create a checkpt version of code to read back trained weights
# upon crash 

SGDR = 'SGDR'

def NASH(model_0,n_steps,n_neigh,n_nm,epoch_neigh,epoch_final,lr_start,lr_end):

	# model_0 is NASGraph object, "treated" already with createModel command 
	model_best = model_0

# hill climbing
	for i in range(n_steps) :  
		print 'STEP ',i,'-'*20
		curr_generator_model = model_best 
		allModels=[]

		for j in range(n_neigh-1) :
			print 'J Neighs :',j
			for k in range(n_nm):
				model_best.nasgr.applyMorph()     	

			model_best.nasgr.applyNecessaryAddNodes()
			# model_best.modifyLinear() 
			Train(SGDR,model_best,epoch_neigh,lr_start,lr_end)
			allModels.append(model_best)

			print 'TRAINED!',j
			model_best = curr_generator_model
        print 'Done J loop'
    	# paper says  : "last model obtained is infact the best model therefore via hillclimbing we choose this."
    	model_best.nasgr.applyNecessaryAddNodes()
    	# model_best.modifyLinear()

    	Train(SGDR,model_best,epoch_neigh,lr_start,lr_end)

    	allModels.append(model_best)   
    	#best model on validation set. 
    	#SELECT MAX ---------------------

    	model_best = allModels[argMax([ValidationPerformance(model_j) for model_j in allModels])]


	model_best.nasgr.applyNecessaryAddNodes()
	# model_best.modifyLinear()

	Train(SGDR,model_best,epoch_final,lr_start,lr_end)

	return model_best



# def NASH(model_0,n_steps,n_neigh,n_nm,epoch_neigh,epoch_final,lr_start,lr_end):

# 	# model_0 is NASGraph object, "treated" already with createModel command 
# 	model_best = model_0

# # hill climbing
# 	for i in range(n_steps) :  
# 		print 'STEP ',i,'-'*20
# 		curr_generator_model = model_best 
# 		model=[]

#     	for j in range(n_neigh-1) :
#     		print 'J Neighs :',j
#     		for k in range(n_nm):
#         		model_best.applyMorph()     	

#         	model_best.applyNecessaryAddNodes()
#         	model_best_final = addLinearLayers(model_best)
#         	Train(SGDR,model_best_final,epoch_neigh,lr_start,lr_end)
#         	model.append(model_best_final)

#         	print 'TRAINED!',j
#         	model_best = curr_generator_model
#         print 'Done J loop'
#     	# paper says  : "last model obtained is infact the best model therefore via hillclimbing we choose this."
#     	model_best.applyNecessaryAddNodes()
#     	model_best_final = addLinearLayers(model_best)
#     	print(model_best_final)
#     	print model_best.nodes[model_best.order[-1]].output
#     	Train(SGDR,model_best_final,epoch_neigh,lr_start,lr_end)

#     	model.append(model_best_final)   
#     	model_best = removeLinearLayers(model_best_final)
#     	#best model on validation set. 
#     	#SELECT MAX ---------------------

#     	model_best = model[argMax([ValidationPerformance(model_j) for model_j in model])]
#     	model_best = removeLinearLayers(model_best)
#     	print 'Plotting Best Graph Seen So Far'
#     	# model_best.plotGraph()


# 	model_best.applyNecessaryAddNodes()
#    	model_best = addLinearLayers(model_best)
# 	model_best_final = Train(SGDR,model_best_final,epoch_final,lr_start,lr_end)

# 	return model_best



# ------------------------------------------------------------------------------------------------
# MAIN CODE FOR HILL CLIMBING
# run.py




