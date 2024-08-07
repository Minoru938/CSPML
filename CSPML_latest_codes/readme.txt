﻿
About this file:

This file contains the latest version of CSPML including training codes. This code corresponds to the result of the paper "Shotgun crystal structure prediction using machine-learned formation energies" (https://arxiv.org/abs/2305.02158). See "Details of the CSPML model" section in supplementary information of the paper for details.


How to build a conda environment for CSPML:

1. cd into this directory.
2. Build a conda environment from CSPML.yml by conda env create -n CSPML -f CSPML.yml  


Usage:

・ To immediately reproduce the crystal structure prediction results reported in the paper, run "CSPML_Structure_Prediction.ipynb" in Jupyter Notebook. You should get the same prediction results as those contained in "cif_files_for_90crystals.zip".

・ If you want to start with training the model, run "Create_strcmp_fgp.ipynb" → "CSPML_Creating_MLdata.ipynb" → "CSPML_training.ipynb" → "CSPML_Structure_Prediction.ipynb " in that order.

################################################################
# If the yml file does not work properly, please refer to the following to build the environment manually

Dependencies:

pandas version = 1.5.1
numpy version = 1.22.4
tensorflow-macos version = 2.9.0
tensorflow-metal = 0.5.1
pymatgen version = 2022.5.26   
matminer version = 0.7.8
scipy version == 1.8.1
joblib version == 1.2.0 
matplotlib version == 3.7.1  
scikit-learn version == 1.1.3  
keras version == 2.9.0
optuna version == 3.0.3 
qpsolvers version == 2.6.0 # peer dependency for KmdPlus.py.

Environment of author:
Python 3.9.16
macOS Ventura 13.4.1
