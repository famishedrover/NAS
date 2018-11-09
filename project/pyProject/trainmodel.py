# --------------------------------------------------------------
# Version : 1.3
# 
# Changehistory : (from Nov 5th 2018)  ./change.txt
# 
# Author : Mudit 
# 
# --------------------------------------------------------------


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


print trainloader

def Test(net) :
	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	with torch.no_grad():
	    for data in testloader:
	        images, labels = data

	        if torch.cuda.is_available() :
	        	images, labels = images.cuda() , labels.cuda()
	        outputs = net(images)
	        _, predicted = torch.max(outputs, 1)
	        c = (predicted == labels).squeeze()
	        for i in range(4):
	            label = labels[i]
	            class_correct[label] += c[i].item()
	            class_total[label] += 1

	for i in range(10):
		print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))



# def Train(optim, net, epochs, lr_start, lr_end):
# 	criterion = torch.nn.MSELoss(reduction='sum')
# 	# use lr_start, lr_end for annealing 
# 	# use SGDR instead of SGD : experiment with other optims
# 	optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
# 	# crate a pytroch test/trainloader and load in batches...
# 	# CIFAR10
# 	# y = from batch
# 	for epoch in range(epochs):  # loop over the dataset multiple times
# 	    running_loss = 0.0
# 	    for i, data in enumerate(trainloader, 0):
# 	    	print len(trainloader)
# 	        # get the inputs
# 	        inputs, labels = data
# 	        # zero the parameter gradients
# 	        optimizer.zero_grad()
# 	        # forward + backward + optimize
# 	        outputs = net(inputs)
# 	        loss = criterion(outputs, labels)
# 	        loss.backward()
# 	        optimizer.step()
# 	        # print statistics
# 	        running_loss += loss.item()
# 	        viewerval = 100
# 	        if i % viewerval == viewerval-1:    # print every 2000 mini-batches
# 	            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / viewerval))
# 	            running_loss = 0.0


def Train(optim, net, epochs, lr_start, lr_end):
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	# use lr_start, lr_end for annealing 
	# use SGDR instead of SGD : experiment with other optims
	for epoch in range(epochs):  # loop over the dataset multiple times

	    running_loss = 0.0
	    for i, data in enumerate(trainloader, 0):
	    	# print len(trainloader)

	        # get the inputs
	        inputs, labels = data
	        if torch.cuda.is_available() :
	        	inputs, labels = inputs.cuda() , labels.cuda()
	        # zero the parameter gradients
	        optimizer.zero_grad()

	        # forward + backward + optimize
	        outputs = net(inputs)
	        loss = criterion(outputs, labels)
	        loss.backward(retain_graph=True)
	        optimizer.step()

	        # print statistics
	        running_loss += loss.item()

	        viewerval = 100

	        print 'Done [{}|{}]'.format(i,running_loss)
	        break

	        if i % viewerval == viewerval-1:    # print every 2000 mini-batches
	            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / viewerval))
	            running_loss = 0.0

	print 'DONE TRAIN'

def argMax(listOfPerformance):
	max_index=0
	max_performance=listOfPerformance[0]
	for i in range(len(listOfPerformance)) :
		if listOfPerformance[i]>max_performance :
			max_performance=listOfPerformance[i]
			max_index=i
	return max_index		


def ValidationPerformance(model) :
	print 
	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	with torch.no_grad():
	    for data in testloader:
	        images, labels = data
	        outputs = model(images)
	        _, predicted = torch.max(outputs, 1)
	        c = (predicted == labels).squeeze()
	        for i in range(4):
	            label = labels[i]
	            class_correct[label] += c[i].item()
	            class_total[label] += 1
	sum_val=0;            
	for i in range(10):
		sum_val=sum_val+((100*(class_correct[i]/class_total[i]))*(100*(class_correct[i]/class_total[i])))
	sum_val=sum_val/10
	return sum_val	




# LINEAR HANDLER ------------------------------------------------------

class NASModule(nn.Module) :
    def __init__(self, input, output=10):
        super(NASModule, self).__init__()
        
        self.input = input
        self.output = output
        
        fc1 = nn.Linear(input,32)
        fc2 = nn.Linear(32,16)
        fc3 = nn.Linear(64,output)

        self.layers = [fc1,fc2,fc3]
        
        self.nns = nn.Sequential(*self.layers)

    def forward(self,x) :
        out = self.nns(x)
        return out
    
    
    def modify_linear(self,no_of_neurons,layer=0):
        # No. of neurons is the actual neurons to be kept 
        # for img output (b,c,x,y) from conv 
        # no_of_neurons = c*x*y

        current_layer = self.layers[layer]
        next_layer = self.layers[layer+1]
   
        I = current_layer.weight.shape[1]
        H = current_layer.weight.shape[0]
        O = next_layer.weight.shape[0]


        weights = [current_layer.weight.data, next_layer.weight.data]

        current_layer = torch.nn.Linear(I,no_of_neurons)
        next_layer = torch.nn.Linear(no_of_neurons,O)

        if no_of_neurons <= H :
        	current_layer.weight.data[0:no_of_neurons,:] = weights[0]
        	next_layer.weight.data[:,0:no_of_neurons] = weights[1]
        else :
        	current_layer.weight.data[0:H,:] = weights[0]
        	next_layer.weight.data[:,0:H] = weights[1]

        self.layers[layer] = current_layer
        self.layers[layer+1] = next_layer
        
        self.nns = nn.Sequential(*self.layers)   




# class Model(nn.Module) :
#     def __init__(self,nasgr,nasout) :
#         super(Model,self).__init__()
#         self.nasgr = nasgr
#         out = nasout[1]*nasout[2]*nasout[3]
#         self.fc1 = nn.Linear(out,16)
#         self.fc2 = nn.Linear(16,10)
#     	self.batch_size = nasout[0]
#     def forward(self,x) :
#     	out = self.nasgr(x)
#     	out = out.view(self.batch_size,-1)
#     	out = self.fc1(out)
#     	out = self.fc2(out)

#     	return out


class Model(nn.Module) :
    def __init__(self,nasgr) :
        super(Model,self).__init__()
        self.nasgr = nasgr

		t = gr(torch.randn((BATCH,BEGIN_IN_CHANNELS,32,32))).shape
		out = t[1]*t[2]*t[3]

		self.fcs = NASModule(out)

		self.batch_size = t[0]


    def forward(self,x) :
    	out = self.nasgr(x)
    	out = out.view(self.batch_size,-1)

    	out = self.fcs(out)
    	return out





# constants --------------------------------------------------------------------------------

BATCH = 4
BEGIN_IN_CHANNELS = 3 
# BEGIN_OUT_CHANNELS = 16
# BEGIN_KERNEL_SIZE = 3
# BEGIN_PADDING = False
# BEGIN_BLOCK = 'conv'
# operationdist = [0.3,0.2,0.2,0.15,0.15]

def addLinearLayers(gr):
	t = gr(torch.randn((BATCH,BEGIN_IN_CHANNELS,32,32))).shape
	net = Model(gr,t)
	return net


def removeLinearLayers(net):
	return net.nasgr












