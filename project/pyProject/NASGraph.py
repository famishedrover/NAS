# -------------------------------------------------------------------------------------------------------------------------------
# 
# NASGraph Module :
# 
#   METHODS ---- 
#   __init__                    : public    - Initialize NASGraph with example input dimen
#   addNode                     : private   - add a given node
#   addInGraph                  : private   - called to add node in graph
#   showAllNodes                : public    - helper @debugging
#   createModel                 : public    - creates ModuleDict for addition of params to model
#   forward                     : public    - Called by user for run
#   f_forward                   : private   - helper to comaptCheck
#   getoperation                : private   - helper to applymorph - selects operation
#   findTwoVerticesForConv      : private   - helper to applyMorph->conv
#   findTwoForAdd               : private   - helper to applyNecessaryAddNodes
#   convWithSingleConvParent    : private   - helper to applyMorph->deepen
#   applyMorph                  : public    - to be called for morphism operation
#   findTwoVerticesForSkip      : private   - helper to applyMorph->skip
#   applyNecessaryAddNodes      : private   - helper to add 'add' nodes
#   findTwoVerticesForMaxPool   : private   - helper to applyMorph->maxpool
#   removeUnecessaryAddNodes    : public    - redundant add nodes removal -- NOT IMPLEMENTED -- 
#   compatCheck                 : public    - run compat check 
#   findConvWithDim             : private   - returns list of valid nodes with dimen compat
#   findTwoVerticesForMerge     : private   - helper to applyMorph->merge   
# 
#   VARIABLES ---- 
#   self.name2type              : DEPRICATE - convert node.name to type <- replace by node.type
#   self.nodes                  : dict      - saves actual nn.Module corresponding to key
#   self.nodesByType            : dict      - {'conv':[],'maxpool':[],'add':[],'merge':[]}
#   self.begin                  : node      - starting node
#   self.alloperations          : list      - ['conv','deepen','skip','maxpool','merge']
#   self.nns                    : nn.Module - Sequential Model -- REPLACE / NOT FUNCTIONAL-- 
#   self.order                  : list      - topsorted nodes
#   self.samplex                : tensor    - sample input tensor for compatCheck
# 
# 
# Version : 1.3
# 
# Changehistory : (from Nov 5th 2018)  ./change.txt
# 
# Author : Mudit 
# 
# -------------------------------------------------------------------------------------------------------------------------------

import torch
from NASGraphBaseClasses import Node,Graph
from block import ConvBlock, Merge, Add, MaxPool
import random

import numpy as np

np.random.seed(2)

Node.count = -1
class NASGraph(torch.nn.Module,Graph) :
    def __init__(self,input,operationdist=[0.5,0.2,0.2,0.05,0.05]) :
        #input is 3d always!!!!
        super(NASGraph,self).__init__()
#         Graph.__init__(self)
        
        
        self.name2type = {}
#         name2type has been replaced by type attribute of Node
#         begin changing code to incorporate node.type instead of self.name2type
        
        self.nodes = {}   # use this only as nn.ModuleDict()
        # self.nodesStr = {}
        self.nodesByType = {'conv':[],
                           'maxpool':[],
                           'add':[],
                           'merge':[]}
        self.begin = None
#         self.alloperations = ['conv','deepen','skip']
#         self.skipoperations = ['add','merge']
        self.alloperations = ['conv','deepen','skip','maxpool','merge']
        
        self.nns = None
        self.order = None
        self.samplex = torch.randn((input[0],input[1],input[2],input[3]))
        print self.samplex.shape
        self.operationdist = operationdist
        
    def addNode(self,**args):
        '''
            for Conv :
            block = conv
            in_channels, out_channels,kernel_size
            
            for MaxPool
            block = 'maxpool'
            kernel_size
            
            for Concat -- not implemented
            block = 'concat'
            
            for Merge
            block = 'merge
            
            for Add 
            block = 'add'
            @returns : created node
        '''
        
        if(args['block'] == 'conv') :
            node = ConvBlock(args['in_channels'],args['out_channels'],args['kernel_size'],padding=args['padding'])
            self.nodesByType['conv'].append(node.id)
            self.name2type[node.name] = 'conv'
            
        elif(args['block'] == 'maxpool') :
            node = MaxPool(args['kernel_size'])
            self.name2type[node.name] = 'maxpool'
            
        elif(args['block'] == 'add') :
            node = Add()
            self.name2type[node.name] = 'add'
            
        elif(args['block'] == 'merge') :
            node = Merge()
            self.name2type[node.name] = 'merge'
#         elif(args['block'] == 'concat') :
#             node = Concat()
        
        try :
            self.nodes[node.id] = node
        except : 
            raise "Hashmap ERR"
            
        if len(self.nodes) == 1 :
            self.begin = node.id
        
        return node.id
    
    def addInGraph(self,log=False,prevnodes=None,nextnodes=None,**args) :
        '''
            @prevnodes = list of node.ids which are parents of this conv node
        '''
        n = self.addNode(**args)
        
        if prevnodes is not None :
            for pnodes in prevnodes :
                assert type(pnodes)==type('1') , 'Check input Type addInGraph'
                self.nodes[pnodes].addChild(n)
                self.nodes[n].addParent(pnodes)
                
        if nextnodes is not None :
            for nnodes in nextnodes :
                assert type(nnodes)==type('1') , 'Check input Type addInGraph'
                self.nodes[n].addChild(nnodes)
                self.nodes[nnodes].addParent(n)
        if log :
            print 'Added node : ', self.nodes[n].name
        return n

    def showAllNodes(self):
        cnt = 0
        for ir in self.nodes.keys() :
            print 'Node Number : ',cnt
            print 'Node ID : ',self.nodes[ir].name

            print 'Node Children :',
            for ch in self.nodes[ir].child :
                print self.nodes[ch].name,
            print ''

            print 'Node Parent :',
            for ch in self.nodes[ir].parent :
                print self.nodes[ch].name,
            print ''


            print self.nodes[ir]
            print '-'*30
            cnt+=1
    
    # def createModel(self,runTopSort=True):
    #     allNodes = []
    #     for ir in self.nodes.keys() :
    #         allNodes.append(self.nodes[ir])

    #     self.nns = torch.nn.Sequential(*allNodes)
        
    #     if runTopSort :
    #         self.order = self.topologicalSort()
    
    def createModel(self):

        # self.nodesStr = {self.nodes[k].type + str(self.nodes[k].name):v for k,v in self.nodes.iteritems()}
        # self.nodesStr = torch.nn.ModuleDict(self.nodesStr)

        self.nodes = torch.nn.ModuleDict(self.nodes)
        
    # def forward(self,x) :
    #      return self.f_forward(x)



    def forward(self,x) :
        # give x to the first one as 


        # print self.nodes
        # print self.order 
        
        self.nodes[self.order[0]](x)

        # do top traversal 
        for i in range(1,len(self.order)):

            # if this node has single parent then pass
            # if then node has multiple parents then check for type of node must be add or concat

            if len(self.nodes[self.order[i]].parent) == 1 :
                # print 'NODE TYPE :' , self.nodes[self.order[i]].type,
                self.nodes[self.order[i]] ( self.nodes[self.nodes[self.order[i]].parent[0]].outputTensor )

            elif self.nodes[self.order[i]].type == 'add' :
                # print 'IN ADD'
                allInpsTensors = []
                for eachpa in self.nodes[self.order[i]].parent :
                    allInpsTensors.append(self.nodes[eachpa].outputTensor)
                d = {'a'+str(i):x_x for i,x_x in enumerate(allInpsTensors)}
                self.nodes[self.order[i]](**d)

            # print 'PROCESSED NODE : ',self.nodes[self.order[i]].name
            # TODO MERGE LAYER 

        return self.nodes[self.order[-1]].outputTensor
    
    # def f_forward(self,x) :
    #     ## INCOMPLETE FUNCTION ... MANAGE WHEN MULTI PARENTS SUPPLY INP TO CHILD

    #     if c == len(self.order) :
    #         return

    #     curr = self.order[c]

    #     if len(self.nodes[curr].parent) != 0 :
    #         self.nodes[curr].outputTensor = self.nodes[curr](trial).shape
    #     else :
    #         self.nodes[curr].input = trial.shape
    #         self.nodes[curr].output = self.nodes[curr](trial).shape

    #     if log :
    #         print '-'*20
    #     self.compatCheck(c=c+1,log=log)
        
     
    def getoperation(self) :
        # op = self.alloperations[random.randint(0,len(self.alloperations)-1)]
        # self.alloperations = ['conv','deepen','skip','maxpool','merge']
        op = np.random.choice(np.arange(0, len(self.alloperations)),p=self.operationdist)
        op = self.alloperations[op]
#         if op == 'skip' :
#             op = self.skipoperations[random.randint(0,len(self.skipoperations)-1)]
        return op
    
    def findTwoVerticesForConv(self,kernel_size,padding,log=False) :
        
        # ------------------------------ PLEASE OPTIMIZE ------------------------------
        posParent = -1
        posChild = -1


        
        idxordered = [val for val in self.order if self.nodes[val].type in ['conv','maxpool']]
        a = random.choice(range(0,len(idxordered)))
            
        #now newindxordered must contain only those conv which are of same 
        # output D as a child of self.nodes[a] if a child exist.
        
#             if len(self.nodes[idxordered[a]].child) > 0 :
        # output of self.nodes[a] must match with output of all parents of next node
        
        # supports finding nodes which can later be converted to add 
        # SUPPORT for 'Merge' and Maxpool req.

        outP = self.nodes[idxordered[a]].output

        if not padding :
            tp = outP
            zp = torch.Size([tp[0],tp[1],tp[2]-kernel_size+1,tp[3]-kernel_size+1])
            outP = zp 


            if log :
                print '-'*20
                print 'SELECTED A' , self.nodes[idxordered[a]].name

                for epsa in range(a,len(idxordered)) :
                    idx = idxordered[epsa]
                    # print 'NODE OUTSHAPE',self.nodes[idx].name, self.nodes[idx].output

                    if self.nodes[idx].output == zp  :
                        print 'NODE OUTSHAPE',self.nodes[idx].name
                    else :
                        print self.nodes[idx].output, zp 


        b = -1
        
        if a != len(idxordered)-1 :
            newidxordered = self.findConvWithDim(dimen=outP,dType='input',afterNode=idxordered[a])
            newidxordered = [val for val in newidxordered if self.nodes[val].type == 'conv']

            # byname = [self.nodes[i].name for i in newidxordered]
            # print 'byname:',byname

            if len(newidxordered) == 0 :
                a = len(idxordered)-1
            else :

                b = random.choice(newidxordered)
                if log :
                    print 'NOT NILL'
                    print 'SELECTED B' , self.nodes[b].name , 'FULL LIST:',newidxordered
        

        
        a = idxordered[a]
        
        if log : 
            print 'A/B :',a,b
        posParent = a
        posChild = -1
        if b != -1 :
            posChild = b

        #directed from posParent to posChild
        # 1. Check for direct Child
        # 2. makeEdgeandCheckForCycle
        # NO MORE REQ. DUE TO TOPSORT
        if posChild != -1:
            # if posChild in self.nodes[posParent].child :
            #     continue  # cont in loop
            #make and test
            # if self.makeEdgeAndCheckIsCyclic(posParent,posChild) :
            #     continue
            # else :
            prevnodes = [posParent]
            nextnodes = [posChild]
            
#           print self.nodes[posParent].name,self.nodes[posChild].name    
            return prevnodes, nextnodes , outP
        else :
            prevnodes = [posParent]
            nextnodes = []
#                 print self.nodes[posParent].name,-1
            return prevnodes, nextnodes , outP
        
            #------ TODO ------
            #prev must always be smaller than next
            #need to check this somehow......else there'll be a cycle BAD BAD BAD BAD BAD BAD
            #NEED TO TAKE CARE OF CYCLES -- NO CYCLES ALLOWED
            
            
    def findTwoForAdd(self):
        pass
    
    #for deepen
    def convWithSingleConvParent(self) :
        reqpairs = []
        for eachconv in self.nodesByType['conv'] :
            #if this node has only one child and its child has only this parent.
            child = self.nodes[eachconv].child
            if len(child) == 1 :
                if self.nodes[child[0]].type == 'conv' :
                    parent = self.nodes[child[0]].parent
                    if len(parent) == 1:
                        reqpairs.append((eachconv,child[0]))
        return reqpairs
    
    def applyMorph(self,log=False) :
        #update output shapes for all nodes.
        self.order = self.topologicalSort()
        self.compatCheck()
        
        op = self.getoperation()
        #op = 'conv'
        # if log :

        completed = False
        print 'OPERATION CALLED : ',op,
        layer = {'block' : op}



        if op == 'conv' :
            #choose kernel
            kernel_size = 3
            if random.random() > 0.5 :
                kernel_size = 5
                
            
            boolPad = False 
            # PADDING_THRESH = 0.5
            if random.random() > 0.2 :
                boolPad = True
                if log :
                    print 'PAD APPLIED'
            #choose position
            prevnodes, nextnodes , exp_out = self.findTwoVerticesForConv(kernel_size=kernel_size,padding = boolPad)
            
            if log :
                print 'Nodes Selected :', self.nodes[prevnodes[0]].name ,
                try :
                    print self.nodes[nextnodes[0]].name ,
                except :
                    print [] ,
                    
            #describe factors
            # layer['in_channels'] = self.nodes[prevnodes[0]].c.out_channels #out of prev one
            layer['in_channels'] = self.nodes[prevnodes[0]].output[1]
            if len(nextnodes) != 0 : 
                layer['out_channels'] = self.nodes[nextnodes[0]].c.in_channels #in of next one
            else :
                layer['out_channels'] = 16 #DEFAULT
                
                
            layer['kernel_size'] = kernel_size
            layer['padding'] = boolPad

            n = self.addInGraph(prevnodes=prevnodes,nextnodes=nextnodes,**layer)
            completed = True
            if log :
                print 'Between : ', prevnodes,' ', nextnodes
                print ' Added :',self.nodes[n].name


            self.nodes[n].output = exp_out
            self.nodes[n].input = self.nodes[prevnodes[0]].output
            
        elif op == 'deepen' :
            # use widenAsCurrent for this node (changes output)
            # use widenAsNext for all its child nodes -> C (changes input)
            # use widenAsCurrent for all the parents of C except this node itself and so on... (changes output)
            # perform above two recursively ...
            
            #choose factor
            factor = 2
            if random.random() > 0.5 :
                factor = 4
            
            #works only for conv->conv one to one nodes.
            pairs = self.convWithSingleConvParent()
            #if nothing found.
            if len(pairs) == 0 :
                if log :
                    print 'FAILED DEEPEN'
                return False
            
            idx = random.choice(range(0,len(pairs)))
            prevN,nextN = pairs[idx]
            
            if log :
                print 'Between : ', self.nodes[prevN].name,' ', self.nodes[nextN].name
            
            self.nodes[prevN].widenAsCurrent(factor)
            self.nodes[nextN].widenAsNext(self.nodes[prevN].c.out_channels)
            completed = True
            
        elif op == 'add' :
            # select two nodes say p,q where p<q topologically 
            # Apply add node here 
            # child of this add node = {all the child nodes of q, some child nodes of q}
            
            # best is to apply at the end.
            pass
            
        elif op == 'merge' :

            # find 2 nodes i,j,k such that 
            # i.output [2,3] == j.output[2,3] == k.input[2,3]
            # and i.output[1] + j.output[1] == k.input[1]
            # we connect merge(i,j) to k

            # prevnodes = [node1, node2]
            prevnodes, nextnodes= self.findTwoVerticesForMerge()

            if prevnodes != -1 :
                n = self.addInGraph(prevnodes=prevnodes,nextnodes=nextnodes,**layer)
                print 'prev :',prevnodes
                completed = True

        elif op == 'skip' :
       
            #choose position
            prevnodes, nextnodes= self.findTwoVerticesForSkip()

            if prevnodes != -1 :

                self.nodes[prevnodes].addChild(nextnodes)
                self.nodes[nextnodes].addParent(prevnodes)
                completed = True
                # print 'SKIP PERFORMED ----------'

            # print 'SKIP OP NOT PERFORMED!'

        elif op == 'maxpool' :
            # choose all those for which maxpool(conv_i).output == (conv_j).input
            # and the last conv
            # randomly sample among these

            # BY CURRENT IMPLEMENTATION OF ADD CONV <- picks 'conv' nodes only so chnage it to take any node and add conv in between

            kernel_size = random.choice([2,3,5])
            layer['kernel_size'] = kernel_size
            prevnodes, nextnodes= self.findTwoVerticesForMaxPool(kernel_size)

            if prevnodes != -1 :
                if nextnodes == -1 :
                    nextnodes = []
                else :
                    nextnodes = [nextnodes]
                n = self.addInGraph(prevnodes=[prevnodes],nextnodes=nextnodes,**layer)


                completed = True
                # print 'MAXPOOL ADDED ---------------'
            # print 'MAXPOOL FAILED'

        return completed
#         self.applyNecessaryAddNodes()
#         self.removeUnecessaryAddNodes()
    

    def findTwoVerticesForMerge(self) :

        # TODO -- IMPLEMENT MERGE WITH ADD CONV :
        # That is, add a conv block according to 2 merged layers...

            # find 2 nodes i,j,k such that 
            # i.output [2,3] == j.output[2,3] == k.input[2,3]
            # and i.output[1] + j.output[1] == k.input[1]
            # we connect merge(i,j) to k
        prevnodes,nextnodes = -1,-1
        allLists = []

        
        idxordered = [val for val in self.order if self.nodes[val].type in ['conv','maxpool']]

        for i in range(len(idxordered)-2) : 
            for j in range(i+1,len(idxordered)):
                for k in range(j+1, len(idxordered)) :
                    ith = self.nodes[idxordered[i]].output 
                    jth = self.nodes[idxordered[j]].output 
                    kth = self.nodes[idxordered[k]].input 

                    if (ith[2] == jth[2] == kth[2]) and (ith[3] == jth[3] == kth[3]) and (ith[1]+jth[1] == kth[1]) :
                        allLists.append(((idxordered[i],idxordered[j]),idxordered[k]))



        if len(allLists) != 0 :
            a = random.choice(range(0,len(allLists)))
            a = allLists[a]
            prevnodes,nextnodes = a[0],a[1]

        return prevnodes,nextnodes     

    def findTwoVerticesForMaxPool(self,kernel_size) :
        prevnodes , nextnodes = -1 , -1
        allLists = []

        
        idxordered = [val for val in self.order if self.nodes[val].type == 'conv']
        # print idxordered

        for eachN in idxordered : 
            tp = self.nodes[eachN].output 
            dimen = torch.Size([tp[0],tp[1],int(tp[2]/kernel_size),int(tp[3]/kernel_size)])

            rList = self.findConvWithDim(dimen,dType='input',forceTop = False,afterNode=eachN)
            
            for eachelement in rList :
                allLists.append((eachN,eachelement))

        # last conv added for maxpool only if last conv doesn't already have a maxpool
        if self.nodes[self.order[-1]].type=='maxpool' :
            pass
        else :
            allLists.append((idxordered[-1],-1))


        if len(allLists) != 0 :
            a = random.choice(range(0,len(allLists)))
            a = allLists[a]
            prevnodes,nextnodes = a[0],a[1]

        return prevnodes,nextnodes



    def findTwoVerticesForSkip(self) :
        # to find all pairs of nodes for which 
        # the input and output match! 
        # they also must be topologically correct! 

        # TODO : extend for 'add'
        prevnodes , nextnodes = -1 , -1
        allLists = []
        for eachN in self.order : 
            if self.nodes[eachN].type == 'conv':
                dimen = self.nodes[eachN].output
                rList = self.findConvWithDim(dimen,dType='input',forceTop = False,afterNode=eachN)
                
                for eachelement in rList :
                    #check for direct child if not then add.
                    if eachelement in self.nodes[eachN].child :
                        continue
                    else :
                        allLists.append((eachN,eachelement))

        if len(allLists) != 0 :
            a = random.choice(range(0,len(allLists)))
            a = allLists[a]
            prevnodes,nextnodes = a[0],a[1]

        # print 'PREV :',prevnodes , 'NEXT :',nextnodes
        return prevnodes,nextnodes

        # print 'ALL LISTS ',allLists
        # for ea,eb in allLists :
        #     print 'ea', self.nodes[ea].output
        #     print 'eb', self.nodes[eb].input


        
    def applyNecessaryAddNodes(self,log=False) :
        #create a new Add node 
        #make all parents of this node and parents of add node
        #make this node child of add node
        
        for no in self.nodesByType['conv']:
            
            
            if len(self.nodes[no].parent)>1 :
                prevnodes = self.nodes[no].parent
                
                #check parents compatible for add operation.
                flag = True
                chkoutput = self.nodes[prevnodes[0]].output
                for ipx in prevnodes[1:] :
                    if self.nodes[ipx].output != chkoutput :
                        if log :
                            print 'Cannot Apply Add'
                        flag = False
                        
                if flag : 
                    nextnodes = [no]

                    for pa in prevnodes :
                        self.nodes[pa].removeChild(no)

                    n = self.addInGraph(prevnodes=prevnodes,nextnodes=nextnodes,**{'block':'add'})
                    self.nodes[no].parent = [n]
                    self.nodes[no].output = self.nodes[prevnodes[0]].output
                
                #remove no as child from its parents list.

        self.order = self.topologicalSort()

    def removeUnecessaryAddNodes(self) :
        #for each add node 
        # if all the children are add nodes
        # take parent node of this main add node 
        # remove this add node
        # create proper links 
#         for no in self.nodes :
#             if self.name2type[self.nodes[no].name] == 'add' :
#                 for ch in self.nodes[no].child :
#                     if self.name2type[self.nodes[ch].name] == 'add' :
        pass



    def compatCheck(self,c=0,log=False) :
        # modify compatCheck for merge operation
        # current implemetation only caters to add where all parents must have the same output shape
        
        x = self.samplex
        if c == len(self.order) :
            return

        curr = self.order[c]

        if log :
            print 'Type :',self.nodes[curr].type , ' Name :',self.nodes[curr].name
        
        if log and self.nodes[curr].type=='conv':
            print 'NODE :',self.nodes[curr].name , ' IN_C :', self.nodes[curr].c.in_channels ,' OUT_C :', self.nodes[curr].c.out_channels 

        if len(self.nodes[curr].parent) != 0 :
            outshape = self.nodes[self.nodes[curr].parent[0]].output
            # print 'OUTSHAPE = ',outshape , self.nodes[self.nodes[curr].parent[0]].name , self.nodes[self.nodes[curr].parent[0]].type
            for par in self.nodes[curr].parent[1:] :
                if log :
                    print 'OutShape Comparison :',outshape,self.nodes[par].output
                

                if (self.nodes[par].output!=outshape) :
                    self.showAllNodes()
                assert (self.nodes[par].output==outshape) ,'CompatCheckErorr'


                outshape = self.nodes[par].output

            trial = torch.randn(outshape)
            self.nodes[curr].input = trial.shape
            
            if self.nodes[curr].type is 'add' :
                #done to simulate add node -- otherwise inp = out for add
                d = {'a'+str(i):x_x for i,x_x in enumerate([trial,trial])}
                self.nodes[curr].output = self.nodes[curr](**d).shape

            
            else :
                self.nodes[curr].output = self.nodes[curr](trial).shape
        else :
            if log :
                print 'No parents...'
            trial = x
            self.nodes[curr].input = trial.shape
            
            if self.nodes[curr].type is 'add' :
            #done to simulate add node -- otherwise inp = out for add
                d = {'a'+str(i):x_x for i,x_x in enumerate([trial,trial])}
                self.nodes[curr].output = self.nodes[curr](**d).shape
            else :
                self.nodes[curr].output = self.nodes[curr](trial).shape


        if log :
            print '-'*20
        self.compatCheck(c=c+1,log=log)
        
        
    def findConvWithDim(self,dimen,dType='output',forceTop = False,afterNode=-1) :
        '''
            dimen = dimen to be searched for
            dType = 'output/input' dimen 

            Run compatCheck before calling this.
        '''
        
        if forceTop :
            self.order = self.topologicalSort()
            self.compatCheck()
        
        order = self.order
        result = []
        if afterNode != -1 :
            # print 'Found'
            pos = order.index(afterNode)
            if pos == len(self.order) - 1:
                return result
            order = order[pos+1:]
        
        convOrdered = [val for val in order if self.nodes[val].type == 'conv']


        for eachconv in convOrdered :
            if dType == 'input' :
                if self.nodes[eachconv].input == dimen :
                    result.append(eachconv)
            elif dType == 'output' :
                if self.nodes[eachconv].output == dimen :
                    result.append(eachconv)
        return result




                    