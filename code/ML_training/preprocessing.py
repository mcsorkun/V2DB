# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:37:00 2019

Project: V2DB (Virtual 2D Material Database)
Content: preprocesses the data for ML development

@author: severin astruc, murat cihan sorkun
"""
import numpy as np
import ase.db

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
import formula_to_onehotvector as fo


# Initialize data directory
db_dir = '../../data/c2db.db'
path_encoders = '../../results/encoders/'

# Initialize the elements that will be not used. 
# Selection is conducted based on minimum number of occurance intraining data

exclude_elements=("C","P","B","Li","K","Rb","Cs","Mg","Na","Fr","Be","Ra",
                  "Rf","Db","Sg","Tc","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl",
                  "Mc","Po","Lv","At","Ts","He","Ne","Ar","Kr","Xe","Rn","Og")

# List of the prototype to be used as base structure
include_prototypes = ("BN", "GeSe", "BiTeI", "CdI2", "GeS2", "MoS2",
                      "MoSSe", "AuSe", "CH", "FeSe", "GaS", "GaSe",
                      "ISb", "NiSe", "PbS", "PbSe", "RhO", "SnS",
                      "FeOCl", "MnS2", "PdS2", "WTe2",)

# Default encoder used for stability and bandgap prediction. We define the
# encoders only for the features we want to use.
# {name: [Label encoder, Onehot encoder, Is the feature to fit_tranform? Not
# is just transform]}
# We developped the encoder for fractional_xnumbat, so it does not need sklearn encoder.
encoders = {'xstoich': [LabelEncoder(), OneHotEncoder(sparse=False), True],
            'xproto': [LabelEncoder(), OneHotEncoder(sparse=False), True],
            'frac_xnumbat': [True]}


def common_element(list1, list2):
    """
    Find if there are a common element in two lists
    return: boolean
    """
    result = False
    for x in list1:
        for y in list2:
            if x == y:
                result = True
                return result
    return result


def target_selection(row, target):
    """
    Get the target from the C2DB database
    Parameters
    ----------
    row: row object
        one row from the C2DB database
    target: str
        Name of the target to get from the C2DB. List of str to use:
            'Egap': target is the electronic band gap.
            'CBM': target is the Conduction Band Minimum.
            'VBM': target is the Valence Band Maximum.
            'WF': target is the work function.
            'HF': target is the heat of formation.
            'D': target is a boolean that say if the material has a direct
                band gap or not.
            'Stab': target is a boolean for which 1 indicates that the
                    materials is stable and 0 unstable.
            'MS': target is the magnetic state.
            'EHull': target is the energy above convex hull.
            'IsMagn': target is the boolean that say if the material is
            magnetic.
    """
    if target == 'Egap':
        return row.gap
    elif target == 'CBM':
        return row.cbm
    elif target == 'VBM':
        return row.vbm
    elif target == 'WF':
        return row.work_function
    elif target == 'HF':
        return row.hform
    elif target == 'D':
        return row.is_dir_gap
    elif target == 'Stab':
        # Return 1 if the material is stable and 0 if not. Stable means that
        # both thermodynamic and dynamic stability are high ( value == 3)
        try:
            # dynamic stability is not always define
            ds_level = row.dynamic_stability_level
        except:
            # When it is not define, we set the dsl to 0, which exlude the
            # materials from the stable list.
            ds_level = 0
        if(ds_level == 3 and row.thermodynamic_stability_level == 3):
            return 1
        else:
            return 0
    elif target == 'MS':
        return row.magstate
    elif target == 'EHull':
        return row.ehull
    elif target == 'IsMagn':
        return row.is_magnetic
    else:
        return row.gap


def data_import(select='gap>=0', target='Egap', save=False,
                encoders=encoders):
    """
    Import features and target from the C2DB database for ML.
    Convert them to onehot  vector, fractional vector or scaled it.
    Parameters
    ----------
    select: str, optional='gap>=0'
        Give the choice of selection the materials. Example: 'gap_gw>=0',
        'gap>0', 'prototype='MoS2'', ...
    target: str, optional:'Egap'
        Name of the target to get from the C2DB. List of str to use:
            'Egap': target is the electronic band gap.
            'CBM': target is the Conduction Band Minimum.
            'VBM': target is the Valence Band Maximum.
            'WF': target is the work function.
            'HF': target is the heat of formation.
            'D': target is a boolean that say if the material has a direct
                band gap or not.
            'Stab': target is a boolean for which 1 indicates that the
                    materials is stable and 0 unstable.
            'IsMagn': target is the boolean that say if the material is
                magnetic.
            'MS': target is the magnetic state.
            'EHull': target is the energy above convex hull.
    encoders: dic of list of encoders
        Get the encoder and the onehot_encoder for the features to encode.
    """
    # Connect to the database
    db = ase.db.connect(db_dir)

    # Select elements according to the select variable
    rows = db.select(select)

#    xstoich = []  # stoichiometry
    xproto = []  # prototype
    xform = []  # formulas (will be used to generate onehotvector of elements)
    xnumbat = []  # number of atoms
    x_target = []  # target value, defined by the parameter target.

    # get the data from C2DB database
    for row in rows:
        # Get the class for selection
        try:
            m_class = row.get("class")
        except:
            m_class = ""
        # Get the stability for selection
        try:
            ds_level = row.dynamic_stability_level
        except:
            ds_level = 0

        # Filter materials going to be used for training        
        selection = (row.prototype in include_prototypes and not
                     common_element(row.symbols, exclude_elements)
                     and row.natoms <= 6 
                     and m_class != "MXene" 
                     and ds_level != 0
                     and fo.check_by_group(row.formula))

        if selection:
            xproto.append(row.prototype)
            xform.append(row.formula)
            xnumbat.append(row.natoms)
            x_target.append(target_selection(row, target))


    #create encoders for prototype vector
    onehot_xproto, label_encoder_xproto, onehot_encoder_xproto = encode(xproto,
                                                                        encoders['xproto'][0],
                                                                        encoders['xproto'][1],
                                                                        encoders['xproto'][-1])
    # Get the list of element used in ou selection
    if(select=="gap>0"):
        elements_list = fo.get_element_list(nonzero=True)
    else:
        elements_list = fo.get_element_list(nonzero=False)
    
    
    # If the target is defined only on positive or 0 bandgap, then the number
    # of element used might be fewer than those with all bandgap. Then, we
    # we save the list of elements from bandgap regression and use it now to
    # onehot encode the fractional vector.
    if encoders['frac_xnumbat'][-1]:
        fractional_xnumbat = fo.create_fractionalvector_by_group(xform, elements_list, xnumbat)

    else:
        fractional_xnumbat = fo.create_fractionalvector_by_group(xform, encoders['frac_xnumbat'][0], xnumbat)

    # Create the 4 length vector of electronegativity. (not used anymore)
    onehot_xen = fo.create_en_vector(xform)

    # Create the 2 length vector of geometric mean of electronegativity (A and B).
    en_AB = fo.create_en_values(xform)


    # If it is needed, the encoders are save with a template name
    if save:
        subname = get_subname(select)
        encoder_to_save = {'label_encoder_xproto': label_encoder_xproto,
                           'onehot_encoder_xproto': onehot_encoder_xproto,
                           'frac_xnumbat': elements_list}
        save_encoder(encoder_to_save, subname)

    # Stack the features to use into one dictionary.
    features = {'onehot_xproto': onehot_xproto,
                'fractional_xnumbat': fractional_xnumbat,
                'onehot_xen': onehot_xen,
                'xnumbat': xnumbat,
                'en_AB': en_AB, }

    return xproto, xform, features, x_target


def get_subname(select):
    """
    Get the subname for saving from the select string. It is an addition to the
    name of the encoders. It gives details on the selection (bandgap) used.
    """
    subname = ''
    if select == 'gap>=0':
        subname = '_all'
    if select == 'gap>0':
        subname = '_positive'
    if select == 'gap=0':
        subname = '_null'
    return subname


def save_encoder(encoders, subname):
    """
    Save the label and onehot encoder as joblib pickle.
    The template name is: encodername_encoder_feature_subname.sav. Encoder is
    [label, onehot, frac], feature can be [xstoich, xproto or xnumbat],
    subname is [all, positive or null].
    """
    for name in encoders.keys():
        name_gap = path_encoders + name + subname + '.sav'
        joblib.dump(encoders[name], name_gap)


def encode(x, label_encoder, onehot_encoder, fit=True):
    """
    Encode the labels and return the encoded label, the label and the
    onehot encoder.
    """
    if fit:
        label_x = label_encoder.fit_transform(x)
    else:
        label_x = label_encoder.transform(x)
    onehot_x, onehot_encoder = scale(label_x, onehot_encoder, fit)
    return onehot_x, label_encoder, onehot_encoder


def scale(x, scaler, fit=True):
    """
    Encode x and return the encoder
    """
    if fit:
        scaled_x = scaler.fit_transform(np.array(x).reshape(-1, 1))
    else:
        scaled_x = scaler.transform(np.array(x).reshape(-1, 1))
    return scaled_x, scaler


def get_X(features, features_key):
    """
    Select and stack the features from feature dictionary
    """
    selected_features = []
    for key in features_key:
        selected_features.append(features[key])
    return np.column_stack(tuple(selected_features))


def get_data(features, features_key, y):
    """
    Get the features from the features dictionary and the target as an array.
    """
    X = get_X(features, features_key)
    y_value = np.array(y)
    return X, y_value
