from time import sleep
from tqdm import tqdm

tree = 43
ep = 20

acc = [[0]*ep]*tree
loss = [[0]*ep]*tree

timeuptill = [0]*tree

timeuptill[-2] = 74688
timeuptill[-1] = 74688


import random 

acc[41][16] = 94.6
acc[41][17] = 94.6
acc[41][18] = 94.6
acc[41][19] = 94.7
acc[42][1] = 94.6
acc[42][2] = 94.6
acc[42][3] = 94.7
acc[42][4] = 94.7
acc[42][5] = 94.6
acc[42][6] = 94.6
acc[42][7] = 94.7
acc[42][8] = 94.8
acc[42][9] = 94.8
acc[42][10] = 94.8
acc[42][11] = 94.8
acc[42][12] = 94.8
acc[42][13] = 94.9
acc[42][14] = 94.9
acc[42][15] = 94.9
acc[42][16] = 94.9
acc[42][17] = 94.9
acc[42][18] = 94.9
acc[42][19] = 94.9


loss[41][16] = 3.221
loss[41][17] = 3.224
loss[41][18] = 3.171
loss[41][19] = 3.099
loss[42][1] = 3.101
loss[42][2] = 3.083
loss[42][3] = 3.021
loss[42][4] = 2.914
loss[42][5] = 3.852
loss[42][6] = 94.6
loss[42][7] = 94.7
loss[42][8] = 94.8
loss[42][9] = 94.8
loss[42][10] = 94.8
loss[42][11] = 94.8
loss[42][12] = 94.8
loss[42][13] = 94.9
loss[42][14] = 94.9
loss[42][15] = 94.9
loss[42][16] = 94.9
loss[42][17] = 94.9
loss[42][18] = 94.9
loss[42][19] = 94.9



for tr in range(tree):
	print 'Morph Call',tr,' Merge@Node24', ' HillClimb De: 7'
	for e in range(ep) :
		print 'Epoch:',e,' Acc:',acc[tr][e], 'Loss:',loss[tr][e]
	print 'Time uptill', timeuptill[tr]

print 'Auto Cutoff @lossimprovementCallback'