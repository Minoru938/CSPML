# CSPML (Crystal Structure Prediction with Machine Learning-based element substitution)
 
CSPML is a unique methodology for the crystal structure prediction (CSP) that relies on a machine learning algorithm (Binary classification neural network model). CSPML predict stable structure
for any given query composition, by automatically selecting from a crystal structure database a set of template crystals with nearly identical stable structures to which atomic substitution is to
be applied. The pre-trained model is used for the selection of the template crystals. The 33,153 candidate compounds (all candidate templates; obtained from [Materials Project](https://materialsproject.org)) and pre-trained models are embedded in CSPML.
 
# Dependencies
 
* pandas version =  1.3.3
* numpy version = 1.19.2 # tensorflow is compatible with numpy=<1.19.2 (2022/01/14).
* tensorflow version = 2.6.0
* pymatgen version = 2020.1.28
* xenonpy version = 0.4.2 (see [this page](https://xenonpy.readthedocs.io/en/latest/installation.html) for installation)
* torch version = 1.10.0 # peer dependency for xenonpy.
* matminer version = 0.6.2 (optional; for calculating structure fingerprint with [local structure order parameters](https://pubs.rsc.org/en/content/articlelanding/2020/ra/c9ra07755c))
 
# Usage
 
1. Install the dependencies (listed above) first.

2. Clone the `CSPML` github repository:
```bash
git clone https://github.com/Minoru938/CSPML.git
```

Note: Due to the size of this repository (about 500MB), this operation can take tens of minutes.

3. `cd` into `CSPML` directory.

4. Run `jupyter notebook` and open `tutorial.ipynb` for demonstration of `CSPML`.


# Environment of author
* Python 3.8.8
* macOS Big Sur version 11.6

# Reference

1. [Materials Project]  A. Jain, S. P. Ong, G. Hautier, W. Chen, W. D. Richards, S. Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, et al., Commentary: The materials project:
A materials genome approach to accelerating materi- als innovation, APL materials 1 (1) (2013) 011002.

2. [Xenonpy] C. Liu, E. Fujita, Y. Katsura, Y. Inada, A. Ishikawa, R. Tamura, K. Kimura, R. Yoshida, Machine learning to predict quasicrystals from chemical compositions,
Advanced Materials 33 (36) (2021) 2170284.

3. [Local structure order parameters] N. E. Zimmermann, A. Jain, Local structure order parameters and site fingerprints for quantification of coordination environment and
crystal structure similarity, RSC Advances 10 (10) (2020) 6063–6081.


 


 

