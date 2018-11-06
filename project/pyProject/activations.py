# -------------------------------------------------------------------------------------------------------------------------------
# 
# AriA2 Module :
# 
#   METHODS ---- 
#   forward                     : public    - Run nn.Module
# 
#   VARIABLES ----     
#   b                           : <AriA docstring>
#   a                           : <AriA docstring>
# 
# 
# AriA Module :
# 
#   METHODS ---- 
#   forward                     : public    - Run nn.Module
# 
#   VARIABLES ----
# 	A  							: lower asymptote, values tested were A = -1, 0, 1
# 	k  							: upper asymptote, values tested were K = 1, 2
#	B 							: exponential rate, values tested were B = [0.5, 2]
# 	v 							: v > 0 the direction of growth, values tested were v = [0.5, 2]
# 	C  							: constant set to 1
# 	Q  							: related to initial value, values tested were Q = [0.5, 2]
# 
# swish fn 						: implements swish activation
# 
# swish_beta fn 				: implements swish beta activation 
# 
# Version : 1.1
# 
# Changehistory : (from Nov 5th 2018)  ./change.txt
# 
# Author : Mudit 
# 
# -------------------------------------------------------------------------------------------------------------------------------



import torch
import torch.nn as nn
from torchsummary import summary

class AriA2(nn.Module):
    def __init__(self, a=1.5, b = 2.):
        super(AriA2, self).__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        aria2 = 1 + ((torch.exp(-x) ** self.b) ** (-self.a)) 
        return x * aria2

class AriA(nn.Module):
    def __init__(self, A=0, K=1., B = 1., v=1., C=1., Q=1.):
        super(AriA, self).__init__()
        # ARiA parameters
        self.A = A # lower asymptote, values tested were A = -1, 0, 1
        self.k = K # upper asymptote, values tested were K = 1, 2
        self.B = B # exponential rate, values tested were B = [0.5, 2]
        self.v = v # v > 0 the direction of growth, values tested were v = [0.5, 2]
        self.C = C # constant set to 1
        self.Q = Q # related to initial value, values tested were Q = [0.5, 2]

    def forward(self, x):
        aria = self.A + (self.k - self.A) / ((self.C + self.Q * torch.exp(-x) ** self.B) ** (1/self.v))
        return x * aria




if __name__ == '__main__' :
	# one hot encoded input for bi-class classification

	class Model(nn.Module) :
	    def __init__(self) :
	        super(Model,self).__init__()
	        
	        self.fc1 = nn.Linear(2,4)
	        self.fc2 = nn.Linear(4,2)
	        self.fc3 = nn.Linear(2,2)
	        
	        self.sigmoid = nn.Sigmoid() 
	        
	        self.swish_bias = nn.Parameter(torch.Tensor([1]))
	        
	        # through default parameters
	        self.aria1 = AriA()
	        #aria2 is faster 
	        self.aria2 = AriA2()
	        self.relu = nn.ReLU()
	        self.relu6 = nn.ReLU6()
	        self.leaky = nn.LeakyReLU()
	        
	        self.activations = [self.noact,self.swish,self.swish_beta,self.aria1,self.aria2,self.relu,self.relu6,self.leaky]
	        self.name = {
	                self.noact : 'No Activation',
	                self.swish : 'Swish',
	                self.swish_beta : 'Beta Swish',
	                self.aria1 : 'AriA',
	                self.aria2 : 'AriA-2',
	                self.relu : 'ReLU',
	                self.relu6 : 'ReLU6',
	                self.leaky : 'LeakyReLU'
	               }
	    
	    def forward(self,x,log=False) :
	        
	        for eachac in self.activations :
	            ans = self.fc1(x)
	            ans = eachac(ans)
	            if log : 
	                print '{:15} : {:1}'.format(self.name[eachac],ans.data[0].cpu().numpy())
	        
	        act = self.swish
	        
	        x = act(self.fc1(x))
	        x = act(self.fc2(x))
	        x = act(self.fc3(x))
	        
	        return x
	    
	    def noact(self,x) :
	        return x
	    def swish(self,x):
	        return x* self.sigmoid(x)

	    def swish_beta(self,x):
	        return x*self.sigmoid(self.swish_bias*x)


	x = torch.randn((1,2))
	m = Model()
	out = m(x,log=True)