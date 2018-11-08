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
	        if i % viewerval == viewerval-1:    # print every 2000 mini-batches
	            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / viewerval))
	            running_loss = 0.0
	            break

def argMax(listOfPerformance):
	max_index=0
	max_performance=listOfPerformance[0]
	for i in range(len(listOfPerformance)) :
		if listOfPerformance[i]>max_performance :
			max_performance=listOfPerformance[i]
			max_index=i
	return max_index		


def ValidationPerformance(model) :
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






