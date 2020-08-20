#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:53:14 2019

Project: V2DB (Virtual 2D Material Database)
Content: utility file to convert formula list into vectors

@author: murat cihan sorkun, severin astruc 
"""
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
from scipy.stats.mstats import gmean

path_type = '../../data/types.csv'
path_enA = '../../data/en_A.csv'
path_enB = '../../data/en_B.csv'


# parses elements and number of occurences from element-number couple
def split_number(text):
    element = text.rstrip('0123456789')
    number = text[len(element):]
    if number == "":
        number = 1
    return element, int(number)



def create_en_values(formula_list):
    """
    Create two lists for electronegativity of A and B elements.
    A: geometric mean of A elements in the unitcell
    B: geometric mean of B elements in the unitcell
    """
    # Import the electronegativity and the group elements from the csv    
    element_grups = pd.read_csv(path_type, header=None)
    element_grups.columns = ['name', 'type']
    
    #Read A and B EN values from CSV and scale it       
    df_A = pd.read_csv(path_enA, header=None, sep='\t')
    df_A.columns = ['name', 'en']
    df_A = df_A.set_index('name')
    min_max_scaler = MinMaxScaler(feature_range=(0.01, 1))
    scaled_value_A = min_max_scaler.fit_transform(df_A.values.reshape(-1, 1))
    df_A['En scaled'] = scaled_value_A
    en_A = df_A['En scaled']

    df_B = pd.read_csv(path_enB, header=None, sep='\t')
    df_B.columns = ['name', 'en']
    df_B = df_B.set_index('name')
    min_max_scaler = MinMaxScaler(feature_range=(0.01, 1))
    scaled_value_B = min_max_scaler.fit_transform(df_B.values.reshape(-1, 1))
    df_B['En scaled'] = scaled_value_B
    en_B = df_B['En scaled']


    # Fill the vector list. This list is filled with 4-length vector
    enA_list = []
    enB_list = []
    for formula in formula_list:
        element_list = re.findall('[A-Z][^A-Z]*', formula)

        # Create a enA_list . Example: [Fe2,O,Cl] -> [0.1,0.1,0.5,0.6]
        element_enA_list = []
        element_enB_list = []
        for elem in element_list:
            element, number = split_number(elem)
            group = element_grups[element_grups["name"] == element]["type"].values[0]
            for i in range(number):
                if(group=="A"):
                    element_enA_list.append(en_A[element])
                if(group=="B"):
                    element_enB_list.append(en_B[element])

        enA_list.append(gmean(element_enA_list))
        enB_list.append(gmean(element_enB_list))
        
    return pd.DataFrame(index=formula_list, data=list(zip(enA_list, enB_list)), columns=['A', 'B'])

# parses elements and number of occurences from the formula list and
# creates chemical compositon vector
def create_fractionalvector_by_order(formula_list,
                                     all_elements_list,
                                     number_of_atoms_list):
    vectors = []
    for formula, number_of_atoms in zip(formula_list, number_of_atoms_list):
        element_vector = [0] * len(all_elements_list)
        element_list = re.findall('[A-Z][^A-Z]*', formula)
        if(number_of_atoms == 2):
            for elem in element_list:
                try:
                    element, number = split_number(elem)
                    element_index = all_elements_list.index(element)
                    element_vector[element_index] = number
                except:
                    pass
        elif(number_of_atoms == 3):
            element_number = 1
            for elem in element_list:
                try:
                    element, number = split_number(elem)
                    element_index = all_elements_list.index(element)
                    if(element_number == 1):
                        element_vector[element_index] = 1
                    else:
                        element_vector[element_index] = element_vector[element_index] + 0.5 * number
                except:
                    pass
                element_number = element_number+number
        elif(number_of_atoms == 4):
            for elem in element_list:
                try:
                    element, number = split_number(elem)
                    element_index = all_elements_list.index(element)
                    element_vector[element_index] = element_vector[element_index] + 0.5 * number
                except:
                    pass
        elif(number_of_atoms == 6):
            element_number = 1
            for elem in element_list:
                try:
                    element, number = split_number(elem)
                    element_index = all_elements_list.index(element)
                    if(element_number <= 2):
                        element_vector[element_index] = element_vector[element_index] + 0.5 * number
                    else:
                        element_vector[element_index] = element_vector[element_index] + 0.25 * number
                except:
                    pass

                element_number = element_number+number
        vectors.append(element_vector)
    onehotvector = pd.DataFrame(index=formula_list,
                                data=vectors,
                                columns=all_elements_list)
    return onehotvector
