# coding: utf-8
# author: Minoru Kusaba (SOKENDAI, kusaba@ism.ac.jp)
# last update: 2022/01/12

"""
CSPML is a unique methodology for the crystal structure prediction (CSP) that relies on a
machine learning algorithm (Binary classification neural network model). CSPML predict stable structure
for any given query composition, by automatically selecting from a crystal structure database a set of
template crystals with nearly identical stable structures to which atomic substitution is to be applied.
The pre-trained model is used for the selection of the template crystals.
33,153 candidate compounds (all candidate templates; obtained from Materials Project) and pre-trained models
are embedded in CSPML.
"""

# Import libraries.
import pandas as pd
import numpy as np
from pymatgen.core.composition import Composition
from xenonpy.descriptor import Compositions
import pickle
import itertools
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Load preset data.
# Elements handled in CSPML.
elements = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V",
 "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
 "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
 "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr",
 "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]

# Candidate templates for CSPML.
with open("./data_set/MP_candidates.pkl", "rb") as f: # preset 33,153 candidate compounds.
    MP_candidates = pickle.load(f)

with open("./data_set/MP_structures.pkl", 'rb') as f: # preset 33,153 candidate structures.
    MP_structures = pickle.load(f)

# Pre-calculated velues for standardizing the XenonPy-calculated descroptor.
with open("./data_set/descriptor_standardization.pkl", 'rb') as f:
    descriptor_standardization = pickle.load(f)

xenonpy_mean = descriptor_standardization["mean"] # equal to the mean of the 33,153 XenonPy-calculated descriptor.
xenonpy_std = descriptor_standardization["std"] # equal to the std of the 33,153 XenonPy-calculated descriptor.

# Dissimilarity of any element pairs for the above-defined elements.
with open("./data_set/element_dissimilarity.pkl", 'rb') as f:
    element_dissimilarity = pickle.load(f)

# Load pre-trained models (Ensemble of NN-binary classifieres).
model1 = tf.keras.models.load_model("./data_set/model1_tau=0.3")
model2 = tf.keras.models.load_model("./data_set/model2_tau=0.3")
model3 = tf.keras.models.load_model("./data_set/model3_tau=0.3")
model4 = tf.keras.models.load_model("./data_set/model4_tau=0.3")
model5 = tf.keras.models.load_model("./data_set/model5_tau=0.3")

models = list([model1, model2, model3, model4, model5])

# Define functions.
def formula_to_composition(formula, elements = elements):
    """
    Transform a pretty formulas (single str object) to a vector of the composition ratio (np.array).
    Args:
           formula (str): single pretty formula (like "SiO2").
           elements (list): a list consists of the element names for creating the vector of the composition ratio.

    Returns: a vector of the composition ratio (np.array).
    """
    comp = Composition(formula)
    vec = np.zeros(len(elements))
    for i in range(0, len(elements)):
        vec[i] = comp.get_atomic_fraction(elements[i])
    return vec

def formula_to_Composition(formula):
    """
    Transform a list of pretty formulas to Composition class objects (pymatgen.core.composition).
    Args:
           formula (list): a list of pretty formulas (like ["SiO2","Li4Ti5O12"]).

    Returns: a list of Composition class objects.
    """
    comp = []
    for i in range(len(formula)):
        comp.append(Composition(formula[i]))
    return comp

def Composition_to_descriptor(comp, mean = xenonpy_mean, std = xenonpy_std):
    """
    Transform a list of Composition class objects (pymatgen.core.composition) to the descriptors
    calculated by xenonpy.descriptor.
    Args:
           comp (list): a list of Composition class objects.
           mean = xenonpy_mean (pandas.Series): pre-calculated mean for nomalizing the descriptors.
           std = xenonpy_std (pandas.Series): pre-calculated standard deviation for nomalizing the descriptors.
    Returns: a pd.Dataframe containing the XenonPy-calculated descriptors (d=290).
    """
    descp = Compositions().transform(comp)
    descp_scaled = (descp - xenonpy_mean)/xenonpy_std
    return descp_scaled

def formula_to_sortedcomposition(formula, elements = elements):
    """
    Transform a list of pretty formulas to the sorted composition ratios.
    Args:
           formula (list): a list of pretty formulas (like ["SiO2","Li4Ti5O12"]).
           elements (list): a list of element's names (str).

    Returns: a pd.Dataframe containing the sorted composition ratios of given formulas.
    """
    N_data = len(formula)
    sorted_composition = np.zeros((N_data, len(elements)))
    for i in range(0, N_data):
        sorted_composition[i,] = np.sort(formula_to_composition(formula[i], elements))[::-1]
    sorted_composition_pd = pd.DataFrame(sorted_composition)
    return sorted_composition_pd

def ensemble_models(X, models = models):
    """
    Calculate an ensemble of the estimated class probabilities of being classified into similar pairs.
    Args:
           X (np.array): the descriptors for paired-formulas (an absolute value of the difference of xenonpy-descriptors).
           models (list): a list of pre-trained models (keras.engine.functional.Functional).

    Returns: a np.array showing an ensemble of the estimated class probabilities of being classified into similar pairs.
    """
    preds = list()
    for i in range(0, len(models)):
        pred = models[i](X)
        preds.append(pred[:,1])
    return np.sum(np.array(preds), axis = 0)/len(models)

def Narrowingdown_candidates(query_formula, candidates = MP_candidates, elements = elements):
    """
    Narrowing down the candidate compounds by the composition ratios of the given query formulas.
    Args:
           query_formula (list): a list of (query) pretty formulas (like ["SiO2","Li4Ti5O12"]).
           candidates (dictionary): a dictionary consists of three keys,'property', 'composition', 'descriptor'.
           Each of their keys contains pandas.DataFrame object which lists properties, composition ratios, and
           chemical composition descriptors of the candidate compounds, respectively.
           elements (list): a list of element's names (str).

    Returns: a list of the dictionaries consists of three keys,'query_formula', 'candidates_num', 'candidates_id'.
    The 'query_formula' shows a query formula (str) which was used for narrowing down candidates. The 'candidates_num'
    shows the number of narrowed-down candidates for a given query formula. The 'candidates_id' shows the material-ids
    of the narrowed-down candidates for a given query formula.
    """
    all_comp = candidates["composition"]
    query_comp = formula_to_sortedcomposition(query_formula, elements)

    survived = []
    for i in range(len(query_formula)):
        ix = np.sum(all_comp == query_comp.iloc[i,:], axis = 1) == len(all_comp.columns)
        candidates_id = candidates["property"][ix]["material_id"].reset_index(drop=True)

        if len(candidates_id) == 0:
            print(f"None of the candidates had the same composition ratio as {query_formula[i]}.")
            candidates_id = list()
            candidates_num = 0

        else:
            candidates_id = list(candidates_id)
            candidates_num = len(candidates_id)

        result = {"query_formula":query_formula[i], "candidates_num":candidates_num,
                  "candidates_id":candidates_id}
        survived.append(result)

    return survived

def Screening_candidates(query_formula, top_K, candidates=MP_candidates, prediction_models=models,
                         mean=xenonpy_mean,std=xenonpy_std,cut_off=0.5, elements = elements):
    """
    Screening the candidate compounds by the pre-trained models into top-K candidates for the given query formulas.
    Args:
           query_formula (list): a list of (query) pretty formulas (like ["SiO2","Li4Ti5O12"]).
           top_K (int): Candidates are screened up to top-K candidates.
           candidates (dictionary): a dictionary consists of three keys,'property', 'composition', 'descriptor'.
           Each of their keys contains pandas.DataFrame object which lists properties, composition ratios, and
           chemical composition descriptors of the candidate compounds, respectively.
           models (list): a list of pre-trained models (keras.engine.functional.Functional).
           mean = xenonpy_mean (pandas.Series): pre-calculated mean for nomalizing the descriptors.
           std = xenonpy_std (pandas.Series): pre-calculated standard deviation for nomalizing the descriptors.
           cut_off (float; default = 0.5): The probability used for cutting-off any candidates of which
           the estimated class-probabilities (of being classified into similar pairs) are not greater than the value.
           elements (list): a list of element's names (str).

    Returns: a list of the dictionaries consists of four keys,"query_formula","topK_formula","topK_id"
    , and "topK_pred". The "query_formula" shows a query formula (str) which was used for screening candidates.
    The "topK_formula" shows the formulas of the screened top-K candidates for a given query formula.
    The "topK_id" shows the material-ids of the screened top-K candidates for a given query formula.
    The "topK_pred" shows the estimated class-probabilities (of being classified into similar pairs)
    of the screened top-K candidates for a given query formula.
    """
    all_comp = candidates["composition"]
    query_comp = formula_to_sortedcomposition(query_formula,elements)
    x = formula_to_Composition(query_formula)
    query_descp = Composition_to_descriptor(x, mean, std)

    predictions = []
    for i in range(len(query_formula)):
        ix = np.sum(all_comp == query_comp.iloc[i,:], axis = 1) == len(all_comp.columns)
        candidates_descp = candidates["descriptor"][ix]
        candidates_id = candidates["property"][ix]["material_id"].reset_index(drop=True)
        candidates_formula = candidates["property"][ix]["pretty_formula"].reset_index(drop=True)

        if len(candidates_id) == 0:
            print(f"None of the candidates had the same composition ratio as {query_formula[i]}.")
            topK_id = list()
            topK_pred = 0
            topK_formula = list()

        else:
            pred = ensemble_models(abs(candidates_descp - query_descp.iloc[i,:]).values,
                                prediction_models)
            topK_id = list(candidates_id[np.argsort(pred)[::-1]][:top_K])
            topK_formula = list(candidates_formula[np.argsort(pred)[::-1]][:top_K])
            topK_pred = np.sort(pred)[::-1][:top_K]

            # Cutting-off candidates.
            surviving = topK_pred>cut_off

            if sum(surviving) == 0:
                print(f"None of the candidates had the class probabilities greater than {cut_off} at {query_formula[i]}.")
                topK_id = list()
                topK_pred = 0
                topK_formula = list()
            else:
                topK_id = [topK_id[j] for j in range(len(topK_id)) if surviving[j]]
                topK_formula = [topK_formula[j] for j in range(len(topK_formula)) if surviving[j]]
                topK_pred = topK_pred[surviving]

        prediction_result = {"query_formula":query_formula[i],"topK_formula":topK_formula,
                             "topK_id":topK_id,"topK_pred":topK_pred}
        predictions.append(prediction_result)

    return predictions

def Structure_prediction(query_formula, top_K, candidates=MP_candidates, structures=MP_structures,elements = elements,
                         prediction_models=models, mean=xenonpy_mean, std=xenonpy_std, element_dissimilarity = element_dissimilarity,
                         cut_off=0.5, SI = False, save_cif = False, save_cif_filename = ""):
    """
    Predicting stable structures for the given query fomulas by element-substitution of the screened top-K candidate
    structures. The screening is performed using pre-trained models with pre-defined candidate set.
    The predicted structures are automatically saved as .cif files into the directory (save_cif_filename), if save_cif = True.
    Args:
           query_formula (list): a list of (query) pretty formulas (like ["SiO2","Li4Ti5O12"]).
           top_K (int): Candidates are screened up to top-K candidates.
           candidates (dictionary): a dictionary consists of three keys,'property', 'composition', 'descriptor'.
           Each of their keys contains pandas.DataFrame object which lists properties, composition ratios, and
           chemical composition descriptors of the candidate compounds, respectively.
           structures (dictionary): a dictionary consists of (at least) two keys,'material_id', 'structure'.
           The 'material_id' should be a np.array containing material-ids for the candidate compounds.
           The 'structure' should be a list containing Structure objects (pymatgen.Structure) for the candidate compounds.
           elements (list): a list of element's names (str).
           models (list): a list of pre-trained models (keras.engine.functional.Functional).
           mean = xenonpy_mean (pandas.Series): pre-calculated mean for nomalizing the descriptors.
           std = xenonpy_std (pandas.Series): pre-calculated standard deviation for nomalizing the descriptors.
           element_dissimilarity (np.arrray): a np.array containing dissimilarities for all pairs of the elements.
           cut_off (float; default = 0.5): The probability used for cutting-off any candidates of which
           the estimated class-probabilities (of being classified into similar pairs) are not greater than the value.
           SI (bool; default = False): If true, supplementary information of the predicted structures are also returned.
           save_cif (bool; default = False): If true, .cif files of the predicted structures are saved as .cif files.
           The top-jth predicted structure of the ith query formula (query_formula[i]) is saved as a "query_formula[i]_j.cif".
           save_cif_filename (str): Name of the directory of which .cif files are saved.

    Returns: (predictions) a list of lists containing pymatgen.Structure objects. predictions[i][j] shows
    the top-(j+1)th predicted structure for the query_formula[i].
    (screened; optionally returned if SI=True) a list of the dictionaries consists of four keys,"query_formula","topK_formula","topK_id"
    , and "topK_pred". The "query_formula" shows a query formula (str) which was used for screening candidates.
    The "topK_formula" shows the formulas of the screened top-K candidates for a given query formula.
    The "topK_id" shows the material-ids of the screened top-K candidates for a given query formula.
    The "topK_pred" shows the estimated class-probabilities (of being classified into similar pairs)
    of the screened top-K candidates for a given query formula. These screened top-K candidates are template structures
    which are used for generating the predicted structures by element-substitution.
    """
    # Screening top_K candidates using pre-trained model for each query formula.
    screened = Screening_candidates(query_formula, top_K, candidates, prediction_models,
                         mean,std,cut_off)
    element_symbol = np.array(elements)
    predictions = []

    for i in range(len(query_formula)):

        predicted_structures = []
        scr_num = len(screened[i]["topK_id"])

        if scr_num == 0:
            pass

        else:
            for j in range(scr_num):

                # The ith query formula.
                vec = formula_to_composition(query_formula[i],elements)
                N_ele = sum(vec != 0)
                comp_index = np.argsort(vec)[::-1][:N_ele]

                # Top-jth suggested formula for ith query formula.
                sug_formula = screened[i]['topK_formula'][j]
                vec_sug = formula_to_composition(sug_formula,elements)
                comp_sug_index = np.argsort(vec_sug)[::-1][:N_ele]

                # Composition of ith fomula (quary & suggested) and it's unique composition ratio.
                comp = np.sort(vec)[::-1][:N_ele]
                keys = np.sort(list(set(comp)))[::-1]

                # Grouping composition-index(=element species) according to unique composition ratio.
                group_index = []
                group_sug_index = []
                for k in range(0, len(keys)):
                    x = (comp == keys[k])
                    group_index.append(comp_index[x])
                    group_sug_index.append(comp_sug_index[x])

                # Find out elements-replacement that minimize element-dissimilarity and make dict showing replacement.
                replacement = []
                for l in range(0, len(keys)):
                    # Replacement is unique.
                    if len(group_index[l]) == 1:
                        replacement.append(group_sug_index[l])
                    # Replacement is not unique.
                    else :
                        seq = group_sug_index[l]
                        pmt = list(itertools.permutations(seq))
                        K = len(pmt)
                        dis_sum = np.zeros(K)
                        for m in range(0, K):
                            dis_sum[m] = sum(element_dissimilarity[group_index[l], pmt[m]])
                        replacement.append(np.array(pmt[np.argmin(dis_sum)]))
                rep_index = np.concatenate(replacement)
                q_ele = element_symbol[comp_index]
                rep_ele = element_symbol[rep_index]
                rep_dict = dict(zip(rep_ele,q_ele))

                # Generating top-jth candidate structure for ith query formula.
                str_index = np.where(structures["material_id"] == screened[i]["topK_id"][j])[0][0] # id to index
                query_str = copy.deepcopy(structures["structure"][str_index])
                query_str.replace_species(rep_dict)
                predicted_structures.append(query_str)

                # Save the structure object as a .cif file into dir = filename (if save_cif=True).
                if save_cif:
                    text =  f"{save_cif_filename}/{query_formula[i]}_{j+1}.cif"
                    query_str.to(filename=text)
                else:
                    pass

        predictions.append(predicted_structures)

    # Return the predicted structures (+ optionally the supplementary information of the predicted structures).
    if SI:
        return predictions, screened
    else:
        return predictions
