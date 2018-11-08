# |--------------------------------------------------------------|
# |                                                              |
# |          ABSTRACT LEVEL IMPLEMENTATION	Ver : 1.1 	         |
# |                                                              |
#  ---------------------------------------------------------------


# GET TRAIN FN from trainmodel.py
# def Train(optim, model, epochs, lr_start, lr_end):
# 	criterion = torch.nn.MSELoss(reduction='sum')

# 	# use lr_start, lr_end for annealing 
# 	# use SGDR instead of SGD : experiment with other optims
# 	optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

# 	# crate a pytroch test/trainloader and load in batches...
# 	# CIFAR10

# 	# y = from batch
# 	y = None
# 	for t in range(epochs):
# 	    y_pred = model(x)
# 	    loss = criterion(y_pred, y)
# 	    optimizer.zero_grad()
# 	    loss.backward()
# 	    optimizer.step()

def argMax(listOfPerformance):
	pass

def ValidationPeformance(model) :
	# @returns validation accuracy
	pass

# some start model_0
model_0 = 0
n_steps = 10
n_neigh = 4
n_nm = 3
epoch_neigh = 30
epoch_final = 50
lr_start = 0.01
lr_end = 0.001  # annealed via SGDR

model_best = model_0


# hill climbing

for i in range(n_steps) :
    
    #NEIGHBOURS ---------------------
    
    # get n_neigh neighbours of model_best
    for j in range(n_neigh-1) :
        model_j = applyMorph(model_best,n_nm)
        
        #train this model_j for a few epochs.
        model_j = Train(SGDR,model_j,epoch_neigh,lr_start,lr_end)
     
    # paper says  : "last model obtained is infact the best model therefore via hillclimbing we choose this."
    model_n_neigh = Train(SGDR,model_best,epoch_neigh,lr_start,lr_end)
    
    #best model on validation set.
    
    #SELECT MAX ---------------------
    model_best = argMax([ValidationPerformance(model_j) for model_j in models_1__n_neigh])
    
    #train final model.
    model_best = Train(SGDR,model_best,epoch_neigh,lr_start,lr_end)
