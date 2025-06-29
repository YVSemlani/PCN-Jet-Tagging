# PCN: A Deep Learning Approach to Jet Tagging Utilizing Novel Graph Construction Methods and Chebyshev Graph Convolutions

Jet tagging is a classification problem in high-energy physics experiments that
aims to identify the collimated sprays of subatomic particles, jets, from particle collisions
and ‘tag’ them to their emitter particle. Advances in jet tagging present opportunities
for searches of new physics beyond the Standard Model. Current approaches use deep
learning to uncover hidden patterns in complex collision data. However, the representation of jets as inputs to a deep learning model have been varied, and often, informative features are withheld from models. In this study, we propose a graph-based representation of a jet that encodes the most information possible. To learn best from this
representation, we design Particle Chebyshev Network (PCN), a graph neural network
(GNN) using Chebyshev graph convolutions (ChebConv). ChebConv has been demonstrated as an effective alternative to classical graph convolutions in GNNs and has yet
to be explored in jet tagging. PCN achieves a substantial improvement in accuracy over
existing taggers and opens the door to future studies into graph-based representations
of jets and ChebConv layers in high-energy physics experiments.

### Notes

Environment setup is very challenging but the code provided is extremely simple so you should be able to directly run our experiments and modify as you wish! Our core contribution is the novel architecture which can be found here. We chose to use DGL but in retrospect I recommend using PyTorch Geometric. 

Read the paper [here](https://arxiv.org/abs/2309.08630)
