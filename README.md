# Expanded Neural Architecture Search 

This is a Bachelor's project on Neural Architecture Search undertaken by students of Delhi Technological University, India.

We are following [1] as the baseline paper.

Expected Contributions : 
- [ ] Learning Curve Prediction.
- [x] Addition of Maxpool Node.
- [x] Addition of Conv Layer without padding.
- [ ] Critique Morphism as in [1].
- [ ] Exploring other methods that may alter image dimen.
- [ ] Improve EER.
- [ ] Exploring activation functions


# Tasks :
- [x] Plot Graph & torchviz Tensor Graph
- [x] Correct find 2 conv nodes to connect function.
- [x] Add padding as fn params of Nodes (Conv etc.)
- [x] Concept : Add support for adding Maxpool Node.
- [ ] Modify compatCheck to get @forward fn.
- [ ] Run Hill Climbing.
- [ ] Testing of Merge Code 
- [x] Write Merge 
- [ ] Merge with Conv 
- [x] Implement Swish, AriA , AriA2, LeakyReLU, ReLU6 , Swish_Beta
- [ ] Add Swish & Beta, AriA , etc as parameters to NASGraph
- [ ] Dynamic Final Linear Layers to accomodate changing convs
- [x] Linear Widen for Dynamic Linear Layers
- [ ] Find Contribution Source eg. Ref Papers/Implementations


Ref ---
[1] https://arxiv.org/abs/1711.04528
