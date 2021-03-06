1.CSPML (Crystal Structure Prediction with Machine Learning-based element substitution)

CSPML is a unique methodology for the crystal structure prediction (CSP) that relies on a machine learning algorithm (Binary classification neural network model). CSPML predict stable structure
for any given query composition, by automatically selecting from a crystal structure database a set of template crystals with nearly identical stable structures to which atomic substitution is to be applied. The pre-trained model is used for the selection of the template crystals. 33,153 candidate compounds (all candidate templates; obtained from Materials Project) and pre-trained models are embedded in CSPML.

2.How to use CSPML.
Set CSPML file as current directory, then open tutorial.ipynb with jupyter notebook
for starting a tutorial which explains how to use the CSPML module (CSPML.py).

3.Dependencies of CSPML:
pandas version =  1.3.3
numpy version = 1.19.2 # tensorflow is compatible with numpy=<1.19.2 (2022/01/14).
tensorflow version = 2.6.0
pymatgen version = 2020.1.28
xenonpy version = 0.4.2
torch version = 1.10.0 # peer dependency for xenonpy.

Environment of author:
Python 3.8.8
macOS Big Sur version 11.6

For calculating structure fingerprint with local order parameters
matminer version = 0.6.2
