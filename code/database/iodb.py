# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:26:46 2019

Project: V2DB (Virtual 2D Material Database)
Content: input-output operations

@author: severin astruc, murat cihan sorkun
"""

import numpy as np
from sklearn.externals import joblib
import formula_to_onehotvector as fo
import re

# Initialize the path
path_encoders = '../../results/encoders/'

# Get the saved encoders
prototype_label_encoder_all = joblib.load(path_encoders + "label_encoder_xproto_all.sav")
prototype_onehot_encoder_all = joblib.load(path_encoders + "onehot_encoder_xproto_all.sav")


def get_prototype_vector(materials_df):
    """
    Get the prototype vector (onehotvector).
    """
    label_prototype = prototype_label_encoder_all.transform(materials_df[1].tolist())
    prototype_vector = prototype_onehot_encoder_all.transform(np.array(label_prototype).reshape(-1,1))
    return prototype_vector


def get_input_data(materials_df, elements_list, EN=False, NA=False):
    """
    Get and stack the input element
    Parameters
    ----------
    materials_df: dataframe
        row: material and his mother prototype
        columns 0: materials
        columns 1: mother prototypes
    elements_list: list
        list of the element used in the generation
    EN: boolean, default is False
        If true, stack the electronegativity vector to the output.
    NA: boolean, default is False
        If true, stack the number of atoms  to the output.
    """
    # Get the number of atoms for each materials of the input. As the input
    # is from one csv file, each material has the same prototype and size. So
    # we can the the first of the list (index 0) to get the size.
    
    all_elements = re.findall('[A-Z][^A-Z]*', materials_df[0][0])
    size_mat = len(all_elements)
    natoms = [size_mat] * len(materials_df)

    # Create the input vectors
    frac_vector = fo.create_fractionalvector_by_order(materials_df[0].tolist(),elements_list, natoms)
    en_values= fo.create_en_values(materials_df[0].tolist())
    prototype_vector = get_prototype_vector(materials_df)

    # Create the electronegativity vector only if the input EN is set to True.
    if (EN and NA):
        # Stack all the input in one narray.
        input_data = np.column_stack((natoms, en_values, prototype_vector, frac_vector))
        
    elif EN:
        # Stack only EN
        input_data = np.column_stack((en_values, prototype_vector, frac_vector))
        
    elif NA:
        # Stack only NA
        input_data = np.column_stack((natoms, prototype_vector, frac_vector))
                
    else:
        # wihout only EN and NA 
        input_data = np.column_stack((prototype_vector, frac_vector))
               
    return input_data
