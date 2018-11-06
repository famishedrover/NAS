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


# ADD PARAMS
# optim, model, epochs, lr_start, lr_end
def Train(net,epochs=2):
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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








