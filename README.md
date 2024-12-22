# CSPML (crystal structure prediction with machine learning-based element substitution)
 
CSPML is a unique methodology for the crystal structure prediction (CSP) that relies on a machine learning algorithm (binary classification neural network model). CSPML predicts a stable structure
for any given query composition, by automatically selecting from a crystal structure database a set of template crystals with nearly identical stable structures to which atomic substitution is to
be applied. Pre-trained models are used to select the template crystals. The 33,153 stable compounds (all candidate crystals; obtained from the [Materials Project](https://materialsproject.org) database) and the pre-trained models are embedded in CSPML.

For more details, please see our paper:
[Crystal structure prediction with machine learning-based element substitution](https://doi.org/10.1016/j.commatsci.2022.111496) (Accepted 3 May 2022).
 
# Dependencies
 
* pandas version =  1.3.3
* numpy version = 1.19.2 # tensorflow is compatible with numpy=<1.19.2 (01/14/2022).
* tensorflow version = 2.6.0
* pymatgen version = 2020.1.28
* xenonpy version = 0.4.2 (see [this page](https://xenonpy.readthedocs.io/en/latest/installation.html) for installation)
* torch version = 1.10.0 # peer dependency for xenonpy.
* matminer version = 0.6.2 (optional; for calculating the structure fingerprint with [local structure order parameters](https://pubs.rsc.org/en/content/articlelanding/2020/ra/c9ra07755c))
 
# Usage
 
1. First install the dependencies listed above.

2. Clone the `CSPML` github repository:
```bash
git clone https://github.com/Minoru938/CSPML.git
```

Note: Due to the size of this repository (about 500MB), this operation can take tens of minutes.

3. `cd` into `CSPML` directory.

4. Run `jupyter notebook` and open `tutorial.ipynb` to demonstrate `CSPML`.


# Environment of author
* Python 3.8.8
* macOS Big Sur version 11.6

# Addition of the latest version of CSPML (2024/07/09)

The latest version of CSPML has been added to this repository as the file "CSPML_latest_codes." This file contains the CSPML training codes, which addressed bias in training data with an updated TensorFlow environment. Please refer to read_me.txt in this file for details on usage. This file corresponds to the result of the paper "[Shotgun crystal structure prediction using machine-learned formation energies](https://doi.org/10.1038/s41524-024-01471-8)". See the "Details of the CSPML model" section in the paper's supplementary information for details. If you want to use CSPML for actual crystal structure prediction or as a comparison method, I recommend using this version of CSPML.

The article titled “Shotgun crystal structure prediction using machine-learned formation energies” has been officially published in *npj Computational Materials* (20 December 2024).

# Reference

1. [Materials Project]: A. Jain, S. P. Ong, G. Hautier, W. Chen, W. D. Richards, S. Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, et al., Commentary: The materials project:
A materials genome approach to accelerating materi- als innovation, APL materials 1 (1) (2013) 011002.

2. [XenonPy]: C. Liu, E. Fujita, Y. Katsura, Y. Inada, A. Ishikawa, R. Tamura, K. Kimura, R. Yoshida, Machine learning to predict quasicrystals from chemical compositions,
Advanced Materials 33 (36) (2021) 2170284.

3. [Local structure order parameters]: N. E. Zimmermann, A. Jain, Local structure order parameters and site fingerprints for quantification of coordination environment and
crystal structure similarity, RSC Advances 10 (10) (2020) 6063–6081.


 


 

