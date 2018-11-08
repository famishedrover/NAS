# -------------------------------------------------------------------------------------------------------------------------------
# 
# Node Module :
# 
#   METHODS ---- 
#   __init__                    : public    - Run nn.Module
#   addChild                    : private   - to be used at child nodes for the node widened via widenAsCurrent
#   addParent                   : private   - to be used at the current node to be widened 'conv'
#   removeChild 
#   removeParent
# 
#   VARIABLES ----     
#   Node.count                  : Node Counter
#   self.name                   : str(Node.count) Used for User understanding ONLY
#   self.type                   : Node type
#   self.id                     : Acutal used node id
#   self.child                  : list
#   self.parent                 : list
#   self.output                 : torch.Size
#   self.input                  : torch.Size
#   self.outputTensor           : Saves complete tensor output -- NOT IMPLEMENTED -- 
# 
# 
# Graph Module :
# 
#   METHODS ---- 
#   isCyclicUtil                : private   - helper to isCyclic
#   isCyclic                    : public    - Cycle detection / DFS
#   makeEdgeAndCheckIsCyclic    : public    - Create Temp Edge and Check for Cycle
#   topologicalSortUtil         : private   - helper to toppologicalSort
#   topologicalSort             : public    - Arrange topsort by name or id
#   plotGraph                   : public    - Use graphviz to plot (Removed networkx version)
# 
#   VARIABLES ----
#   NONE
# 
# Version : 1.0
# 
# Changehistory : (from Nov 5th 2018)  ./change.txt
# 
# Author : Mudit 
# 
# -------------------------------------------------------------------------------------------------------------------------------


import torch.nn as nn
import torch
import numpy as np

from graphviz import Digraph
import math

import matplotlib.pyplot as plt 



class Node:
    count = -1
    def __init__(self,type=None) :
        Node.count += 1
        self.name = str(Node.count)
        self.type = type
        self.id = str(id(self))
        self.child = []  
        self.parent = []
        self.output = None
        self.input = None
        self.outputTensor = None
    def addChild(self,x):
        self.child.append(x)
    def removeChild(self,x) :
        self.child.remove(x)
    def addParent(self,x):
        self.parent.append(x)
    def removeParent(self,x) :
        self.parent.remove(x)

class Graph :    
    '''
        Implements TopSort & IsCyclic
    ''' 
    
    
    def __init__(self):
        pass

    
    def isCyclicUtil(self,v, visited, recStack): 

        visited[v] = True
        recStack[v] = True

        for neighbour in self.nodes[v].child: 
            if visited[neighbour] == False: 
                if self.isCyclicUtil(neighbour, visited, recStack) == True: 
                    return True
            elif recStack[neighbour] == True: 
                return True

        recStack[v] = False
        return False

    def isCyclic(self): 
        visited = {}
        recStack = {}
        for i in self.nodes :
            visited[i]=False
            recStack[i]=False

        for node in self.nodes: 
            if visited[node] == False: 
                if self.isCyclicUtil(node,visited,recStack) == True: 
                    return True
        return False
    
    def makeEdgeAndCheckIsCyclic(self,v1,v2):
        '''
            Edge test from v1 to v2
        '''
        #makeEdge
        self.nodes[v1].child.append(v2)
        #Check
        cycle = self.isCyclic()
        #return to normal
        self.nodes[v1].child.remove(v2)
        
        return cycle

    
    def topologicalSortUtil(self,v,visited,stack): 
        visited[v] = True
        for i in self.nodes[v].child: 
            if visited[i] == False: 
                self.topologicalSortUtil(i,visited,stack) 

        stack.insert(0,v) 

    def topologicalSort(self,name=False): 
        visited = {}
        for i in self.nodes :
            visited[i]=False

        stack =[] 

        for i in self.nodes: 
            if visited[i] == False: 
                self.topologicalSortUtil(i,visited,stack) 
        
        if name :
            for i in stack :
                print self.nodes[i].name,' ',
        else :
            return stack
        
    def plotGraph(self,plot=True,filename='unix.gv') :
        u = Digraph('unix', filename=filename)
        cmap = {'conv':'red','add':'blue','merge':'green','maxpool':'yellow'}
        
        for nod in self.nodes :
            na = self.name2type[self.nodes[nod].name]+self.nodes[nod].name
            u.node(na,color=cmap[self.name2type[self.nodes[nod].name]])
        
        for nod in self.nodes :
            for ch in self.nodes[nod].child :
                na = self.name2type[self.nodes[nod].name]+self.nodes[nod].name
                nb = self.name2type[self.nodes[ch].name]+self.nodes[ch].name
                u.edge(na,nb)
        
        if plot :
            u.view()

