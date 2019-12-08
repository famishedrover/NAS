# Expanded Neural Architecture Search 

We are following [1] as the baseline paper.

Expected Contributions : 
- [ ] Learning Curve Prediction.
- [x] Addition of Maxpool Node.
- [x] Addition of Conv Layer without padding.
- [ ] Critique Morphism as in [1].
- [ ] Exploring other methods that may alter image dimen.
- [ ] Improve EER.
- [x] Exploring activation functions


# Tasks :
- [x] Plot Graph & torchviz Tensor Graph
- [x] Correct find 2 conv nodes to connect function.
- [x] Add padding as fn params of Nodes (Conv etc.)
- [x] Concept : Add support for adding Maxpool Node.
- [x] Modify compatCheck to get @forward fn.
- [x] Run Hill Climbing.
- [X] Testing of Merge Code 
- [x] Write Merge 
- [x] Merge with Conv 
- [x] Implement Swish, AriA , AriA2, LeakyReLU, ReLU6 , Swish_Beta
- [x] Add Swish & Beta, AriA , etc as parameters to NASGraph
- [x] Add Final Layers
- [x] Run Training Loop 
- [x] Dynamic Final Linear Layers to accomodate changing convs
- [x] Linear Widen for Dynamic Linear Layers
- [ ] Find Contribution Source eg. Ref Papers/Implementations


Ref ---
[1] https://ieeexplore.ieee.org/abstract/document/8791709
[2] https://arxiv.org/abs/1711.04528
