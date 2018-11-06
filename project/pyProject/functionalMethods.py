# |--------------------------------------------------------------|
# |                                                              |
# |          TESTER SCRIPT ---- NOT USED FOR main.py Run         |
# |                                                              |
#  ---------------------------------------------------------------




# order = gr.topologicalSort()
# curr = order[0]
# x = torch.randn((2,gr.nodes[gr.begin].c.in_channels,128,128))
def compatCheck(c,gr,log=False) :
    x = torch.randn((2,gr.nodes[gr.begin].c.in_channels,128,128))
    if c == len(gr.order) :
        return
    
    curr = gr.order[c]
    
    if log:
        print 'NODE :',gr.nodes[curr].name , ' IN_C :', gr.nodes[curr].c.in_channels ,' OUT_C :', gr.nodes[curr].c.out_channels 
    
    if len(gr.nodes[curr].parent) != 0 :
        outshape = gr.nodes[gr.nodes[curr].parent[0]].output
        for par in gr.nodes[curr].parent[1:] :
            if log :
                print 'OutShape Comparison :',outshape,gr.nodes[par].output
            assert (gr.nodes[par].output==outshape) ,'CompatCheckErorr'
            outshape = gr.nodes[par].output
            
        trial = torch.randn(outshape)
        gr.nodes[curr].input = trial.shape
        gr.nodes[curr].output = gr.nodes[curr](trial).shape
    else :
        if log :
            print 'No parents...'
        trial = x
        gr.nodes[curr].input = trial.shape
        gr.nodes[curr].output = gr.nodes[curr](trial).shape
        
    
    if log :
        print '-'*20
    compatCheck(c+1,gr,log)


x = torch.Size([2, 16, 116, 116])
input = torch.Size([2,16,128,128])
def findConvWithDim(gr,dimen,dType='output') :
    '''
        dimen = dimen to be searched for
        dType = 'output/input' dimen 
        
        Run compatCheck before calling this.
    '''
    def fn(x):
        return gr.name2type[gr.nodes[x].name]
    
    convOrdered = [gr.order[i] for i,val in enumerate(map(fn,gr.order)) if val=='conv']
    result = []
    for eachconv in convOrdered :
        if dType == 'input' :
            if gr.nodes[eachconv].input == dimen :
                result.append(eachconv)
        elif dType == 'output' :
            if gr.nodes[eachconv].output == dimen :
                result.append(eachconv)
    return result



Node.count = -1
gr = NASGraph(([8, 128, 128]))
p = {'block':'conv','in_channels':8,'out_channels':16,'kernel_size':3}
gr.addInGraph(**p)
for _ in range(6):
    gr.applyMorph()
print 'TopSort:',
gr.order = gr.topologicalSort()
gr.topologicalSort(name=True)
gr.createModel()
x = torch.randn((2,gr.nodes[gr.begin].c.in_channels,128,128))

print '-'*20
print 'ALL NODES'
gr.topologicalSort(name=True)
compatCheck(0,gr,log=True)
for each in gr.nodes :
    print gr.nodes[each].name , gr.nodes[each].input, gr.nodes[each].output

print '-'*20
print 'OUTPUT DIMEN',[2, 16, 124, 124]
resout = findConvWithDim(gr,torch.Size([2, 16, 124, 124]),'output')
for r in resout :
    print gr.nodes[r].name
print '-'*20
print 'INPUT DIMEN',[2, 16, 124, 124]
resout = findConvWithDim(gr,torch.Size([2, 16, 124, 124]),'input')
for r in resout :
    print gr.nodes[r].name


gr.plotGraph2()