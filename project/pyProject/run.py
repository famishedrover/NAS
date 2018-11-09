# ------------------------------------------------------------------------------------
#  TAKE CARE OF REDUNDANT IMPORTS 

from hillclimbing import NASH
from NASGraph import NASGraph

# ------------------------------------------------------------------------------------



BATCH = 4
BEGIN_IN_CHANNELS = 3 
BEGIN_OUT_CHANNELS = 16
BEGIN_KERNEL_SIZE = 3
BEGIN_PADDING = False
BEGIN_BLOCK = 'conv'
operationdist = [0.3,0.2,0.2,0.15,0.15]
IMAGE_X = 32
IMAGE_Y = 32


# n_steps = 10
# n_neigh = 4
# n_nm = 3
# epoch_neigh = 30
# epoch_final = 50
# lr_start = 0.01
# lr_end = 0.001  # annealed via SGDR


n_steps = 3
n_neigh = 3
n_nm = 5
epoch_neigh = 1
epoch_final = 1
lr_start = 0.01
lr_end = 0.001  # annealed via SGDR

# --------------------------------------------------------------
# INIT GRAPH

gr = NASGraph(([BATCH,BEGIN_IN_CHANNELS, IMAGE_X, IMAGE_Y]),operationdist)
p = {'block':BEGIN_BLOCK,'in_channels':BEGIN_IN_CHANNELS,'out_channels':BEGIN_OUT_CHANNELS,'kernel_size':BEGIN_KERNEL_SIZE,'padding':BEGIN_PADDING}
gr.addInGraph(**p)

print gr

gr.createModel()
# --------------------------------------------------------------
# RUN HILL CLIMBING / OTHER OPTIM 

final_trained = NASH(model_0=gr,
					n_steps=n_steps,n_neigh=n_neigh,
					n_nm=n_nm,epoch_neigh=epoch_neigh,
					epoch_final=epoch_final,
					lr_start=lr_start,lr_end=lr_end)






