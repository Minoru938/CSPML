# coding: utf-8
# author: Minoru Kusaba (ISM, kusaba@ism.ac.jp)
# last update: 2022/07/31
# minor update: 2023/07/21

"""
This module contains a class for treating Kernel mean descriptor (KMD),
and a function for generating descriptors with summary statistics.
"""

# Import libraries.
import pandas as pd
import numpy as np
from statistics import median
from scipy.spatial import distance_matrix
from pymatgen.core.composition import Composition
from qpsolvers import solve_qp

# Load preset data.
element_features = pd.read_csv("data_set/element_features.csv", index_col= 0) # element-level descriptors of shape (94, 58).
elements = list(element_features.index) # 94 elements, "H" ~ "Pu".

# Define functions.
def formula_to_composition(formula, elements = elements):
    """
    Convert a chemical formula to a composition vector for the predifened elements.

    Args
    ----
    formula: str
          Chemical formula (e.g. "SiO2").
    elements: a list of str
          Chemical elements (e.g. ["H", "He", ...]).
    Returns
    ----
    vec: numpy.ndarray of shape (len(elements),).
    """  
    comp = Composition(formula)
    vec = np.array([comp.get_atomic_fraction(elements[i]) for i in range(len(elements))])
    return vec

class KMD():
    """
    Kernel mean descriptor (KMD).
    """
    def __init__(self, method = "1d"):
        """
        Parameters
        ----
        method: str, default = "1d"
              method must be "md" or "1d".
              For "md", KMD is generated on a multidimensional feature space.
              For "1d", KMD is generated for each feature, then combined.
        ----
        """
        self.method = method
    
    def transform(self, weight, component_features, n_grids = None, sigma = "auto", scale = True):
        """
        Generate kernel mean descriptor (KMD) with the Gaussian kernel (materials → descriptors).
        
        Args
        ----
        weight: array-like of shape (n_samples, n_components)
              Mixing ratio of constituent elements that make up each sample.
        component_features: array-like of shape (n_components, n_features)
              Features for each constituent element.
        n_grids: int, default = None
              The number of grids for discretizing the kernel mean.
              The kernel mean is discretized at the n_grids equally spaced grids 
              between a maximum and minimum values for each feature.
              This argument is only necessary for "1d".
        sigma: str or float, default = "auto"
              A hyper parameter defines the kernel width.
              If sima = "auto", the kernel width is given as the inverse median of the nearest distances
              for "md", and as the inverse of the grid width for "1d".
        scale: bool, default = True
              IF scale = True, component_features is scaled.
        Returns
        ----
        KMD: numpy array of shape (n_samples, n_components) for "md", and (n_samples, n_features*n_grids) for "1d".
        """  
        self.component_features = component_features
        self.sigma = sigma
        self.scale = scale
        # Generate KMD on a multidimensional feature space.
        if self.method == "md":
          
            # Standardize each feature to have mean 0 and variance 1 (for "md").
            if scale == True:
                component_features = (component_features - component_features.mean(axis=0))/component_features.std(axis=0, ddof=1)
            else:
                pass
            
            # Set the kernel width as the inverse median of the nearest distances.
            if sigma == "auto":
                d = distance_matrix(component_features, component_features)**2
                min_dist = [np.sort(d[i,:])[1] for i in range(component_features.shape[0])] # the nearest distances
                gamma = 1/median(min_dist)
                kernelized_component_features = np.exp(-d * gamma)
                KMD = np.dot(weight, kernelized_component_features)
                return KMD
            
            # Manually set the kernel width.
            else:
                d = distance_matrix(component_features, component_features)**2
                kernelized_component_features = np.exp(-d/(2*sigma**2))
                KMD = np.dot(weight, kernelized_component_features)
                return KMD
         
        # Generate KMD for each feature, then combine them.
        elif self.method == "1d":
            
            if n_grids == None:
                print('For self.method = "1d", please set n_grids')
                return
            else:
                pass
            
            # Min-Max Normalization (for "1d").
            if scale == True:
                component_features = (component_features - component_features.min(axis=0))/(component_features.max(axis=0) - component_features.min(axis=0))
            else:
                pass
            
            # Set the kernel width as the inverse of the grid width.
            if sigma == "auto":
                max_cf = component_features.max(axis=0) 
                min_cf = component_features.min(axis=0)
                x = np.asarray(component_features)
                k = []
                for i in range(component_features.shape[1]):
                    grid_points = np.linspace(min_cf[i], max_cf[i], n_grids) 
                    gamma = 1/(grid_points[1] - grid_points[0])**2
                    d = np.array([(x[j,i] - grid_points)**2 for j in range(x.shape[0])])
                    k.append(np.exp(-d*gamma))
                kernelized_component_features = np.concatenate(k, axis=1)
                KMD = np.dot(weight, kernelized_component_features)
                return KMD
                    
            # Manually set the kernel width.
            else:
                max_cf = component_features.max(axis=0) 
                min_cf = component_features.min(axis=0)
                x = np.asarray(component_features)
                k = []
                for i in range(component_features.shape[1]):
                    grid_points = np.linspace(min_cf[i], max_cf[i], n_grids) 
                    d = np.array([(x[j,i] - grid_points)**2 for j in range(x.shape[0])])
                    k.append(np.exp(-d/(2*sigma**2)))
                kernelized_component_features = np.concatenate(k, axis=1)
                KMD = np.dot(weight, kernelized_component_features)
                return KMD       
        else:
            print('self.method must be "md" or "1d"')
    
    def inverse_transform(self, KMD): 
        """
        Derive the weights of the constituent elements for a given kernel mean descriptors 
        by solving a quadratic programming (descriptors → materials).
        
        Args
        ----
        KMD: array-like of shape (n_samples, n_components) for "md", (n_samples, n_features*n_grids) for "1d".
              Kernel mean descriptor (KMD).
        Returns
        ----
        weight: numpy array of shape (n_samples, n_components).
        """  
        component_features = self.component_features
        sigma = self.sigma
        scale = self.scale
        if self.method == "md":
            
            # Standardize each feature to have mean 0 and variance 1 (for "md").
            if scale == True:
                component_features = (component_features - component_features.mean(axis=0))/component_features.std(axis=0, ddof=1)
            else:
                pass
            
            KMD = np.asarray(KMD)
            n_components = KMD.shape[1]
            
            # Set the kernel width as the inverse median of the nearest distances.
            if sigma == "auto":
                d = distance_matrix(component_features, component_features)**2
                min_dist = [np.sort(d[i,:])[1] for i in range(component_features.shape[0])] # the nearest distances
                gamma = 1/median(min_dist)
                kernelized_component_features = np.exp(-d * gamma)
                P = np.dot(kernelized_component_features, kernelized_component_features.T) 
                if min(np.linalg.eigvals(P)) <= 0:
                    print("Given KMD is not inversible: smaller sigma may solve the problem")
                    return
                else:
                    pass
                # Equality constraints.
                A = np.ones(P.shape[0])
                b = np.array([1.])
                # Inequality constraints.
                G = np.diag(-A)
                h = np.zeros(P.shape[0])
                # Solve quadratic programming.
                w_raw = np.array([solve_qp(P, -np.dot(kernelized_component_features, KMD[i])
                                                     , G, h, A, b, solver="quadprog") for i in range(KMD.shape[0])])
                w = np.round(abs(w_raw), 12)       
                weight = w/w.sum(axis=1)[:, None]  
                return weight
                    
            # Manually set the kernel width.
            else:
                d = distance_matrix(component_features, component_features)**2
                kernelized_component_features = np.exp(-d/(2*sigma**2))
                P = np.dot(kernelized_component_features, kernelized_component_features.T) 
                if min(np.linalg.eigvals(P)) <= 0:
                    print("Given KMD is not inversible: smaller sigma may solve the problem")
                    return
                else:
                    pass
                # Equality constraints.
                A = np.ones(P.shape[0])
                b = np.array([1.])
                # Inequality constraints.
                G = np.diag(-A)
                h = np.zeros(P.shape[0])
                # Solve quadratic programming.
                w_raw = np.array([solve_qp(P, -np.dot(kernelized_component_features, KMD[i])
                                                     , G, h, A, b, solver="quadprog") for i in range(KMD.shape[0])])
                w = np.round(abs(w_raw), 12)       
                weight = w/w.sum(axis=1)[:, None]  
                return weight
            
        elif self.method == "1d":
            
            KMD = np.asarray(KMD)
            n_grids = int(KMD.shape[1]/component_features.shape[1])
            
            # Min-Max Normalization (for "1d").
            if scale == True:
                component_features = (component_features - component_features.min(axis=0))/(component_features.max(axis=0) - component_features.min(axis=0))
            else:
                pass
            
            # Set the kernel width as the inverse of the grid width.
            if sigma == "auto":
                max_cf = component_features.max(axis=0) 
                min_cf = component_features.min(axis=0)
                x = np.asarray(component_features)
                k = []
                for i in range(component_features.shape[1]):
                    grid_points = np.linspace(min_cf[i], max_cf[i], n_grids) 
                    gamma = 1/(grid_points[1] - grid_points[0])**2
                    d = np.array([(x[j,i] - grid_points)**2 for j in range(x.shape[0])])
                    k.append(np.exp(-d*gamma))
                kernelized_component_features = np.concatenate(k, axis=1)
                P = np.dot(kernelized_component_features, kernelized_component_features.T) 
                if min(np.linalg.eigvals(P)) <= 0:
                    print("Given KMD is not inversible: consider increasing the number of grids (n_grids)")
                    return
                else:
                    pass
                # Equality constraints.
                A = np.ones(P.shape[0])
                b = np.array([1.])
                # Inequality constraints.
                G = np.diag(-A)
                h = np.zeros(P.shape[0])
                # Solve quadratic programming.
                w_raw = np.array([solve_qp(P, -np.dot(kernelized_component_features, KMD[i])
                                                     , G, h, A, b, solver="quadprog") for i in range(KMD.shape[0])])
                w = np.round(abs(w_raw), 12)       
                weight = w/w.sum(axis=1)[:, None]  
                return weight
            
            # Manually set the kernel width.
            else:
                max_cf = component_features.max(axis=0) 
                min_cf = component_features.min(axis=0)
                x = np.asarray(component_features)
                k = []
                for i in range(component_features.shape[1]):
                    grid_points = np.linspace(min_cf[i], max_cf[i], n_grids) 
                    d = np.array([(x[j,i] - grid_points)**2 for j in range(x.shape[0])])
                    k.append(np.exp(-d/(2*sigma**2)))
                kernelized_component_features = np.concatenate(k, axis=1)
                P = np.dot(kernelized_component_features, kernelized_component_features.T) 
                if min(np.linalg.eigvals(P)) <= 0:
                    print("Given KMD is not inversible: consider increasing the number of grids (n_grids)")
                    return
                else:
                    pass
                # Equality constraints.
                A = np.ones(P.shape[0])
                b = np.array([1.])
                # Inequality constraints.
                G = np.diag(-A)
                h = np.zeros(P.shape[0])
                # Solve quadratic programming.
                w_raw = np.array([solve_qp(P, -np.dot(kernelized_component_features, KMD[i])
                                                     , G, h, A, b, solver="quadprog") for i in range(KMD.shape[0])])
                w = np.round(abs(w_raw), 12)       
                weight = w/w.sum(axis=1)[:, None]  
                return weight
                
        else:
            print('self.method must be "md" or "1d"')
            
def StatsDescriptor(formula, component_features, stats = ["mean", "sum", "var", "max", "min"]):
    """
    Generate descriptors for mixture systems using summary statistics.

    Args
    ----
    weight: array-like of shape (n_samples, n_components)
          Mixing ratio of constituent elements that make up each sample.
    component_features: array-like of shape (n_components, n_features)
          Features for each constituent element.
    stats: a list of str, default = ["mean", "sum", "var", "max", "min"]
          Type of summary statistics for generating descriptors.
          Only "mean", "sum", "var", "max" and "min" are supported.
    Returns
    ----
    SD: numpy array of shape (n_samples, n_features*len(stats)).
    """  
    n_samples = len(formula)
    # Get comp weight.
    w = np.array([formula_to_composition(formula[i]) for i in range(n_samples)])
    # as array.
    cf = np.asarray(component_features)

    s = []
    for x in stats:
        # Weighted mean.
        if x == "mean":
            wm = np.dot(w, cf)
            s.append(wm)
        # Weighted mean.
        elif x == "sum":
            wm = np.dot(w, cf)
            n_atoms = np.array([Composition(formula[i]).num_atoms for i in range(n_samples)]) # only for sum.
            n_atoms_array = np.array([n_atoms for i in range(wm.shape[1])]).T
            s.append(wm * n_atoms_array)
        # Weighted variance.
        elif x == "var":
            wm = np.dot(w, cf)
            wv = np.array([np.dot(w[i], (cf - wm[i])**2) for i in range(n_samples)])
            s.append(wv)
        # Maximum pooling.
        elif x == "max":
            nonzero = (w != 0) 
            maxp = np.array([cf[nonzero[i]].max(axis = 0) for i in range(n_samples)])
            s.append(maxp)
        # Minimum pooling.
        elif x == "min":
            nonzero = (w != 0) 
            minp = np.array([cf[nonzero[i]].min(axis = 0) for i in range(n_samples)])
            s.append(minp)
        else:
            print(f'"{x}" is not supported: only "mean", "var", "max" and "min" are supported as stats')

    SD = np.concatenate(s, axis = 1)
    return SD