#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:53:14 2019

Project: V2DB (Virtual 2D Material Database)
Content: utility file to convert formula list into vectors

@author: murat cihan sorkun, severin astruc
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import re
from scipy.stats.mstats import gmean


path_type = '../../data/types.csv'
path_en = '../../data/en.csv'
path_enA = '../../data/en_A.csv'
path_enB = '../../data/en_B.csv'
path_elements = '../../data/elements.csv'
path_nonzero_elements = '../../data/elements_nonzero.csv'

# Create list for all elements dataset includes
def create_elemen_list(formula_list):
    all_elements_list = []
    for formula in formula_list:
        element_list = re.findall('[A-Z][^A-Z]*', formula)
        for elem in element_list:
            element, number = split_number(elem)
            if element not in all_elements_list:
                all_elements_list.append(element)

    return all_elements_list


# Get default element list from file
def get_element_list(nonzero=False):

    if nonzero:      
        elements_df = pd.read_csv(path_nonzero_elements)
        
    else:        
        elements_df = pd.read_csv(path_elements)
        
    return elements_df['elements'].tolist()


# parses elements and number of occurences from element-number couple
def split_number(text):

    element = text.rstrip('0123456789')
    number = text[len(element):]
    if number == "":
        number = 1
    return element, int(number)


# Not used anymore
def create_en_vector(formula_list):
    """
    Create a list of 4-length vector filled with electronegativity of B
    elements.
    For BBBB case it fills for each B independently: B1B2B3B4
    For BB case it fills B1B2B1B2
    For B case: B1B1B1B1
    Electronegativity are collected from wikipedia and stored in a xlsx file.

    Parameters
    ----------
    formula_list: list of str
        All the formula we want the electronegativity vector

    Output
    ------
    onehot_vector: dataframe
        index: the formula of each material
        Columns: the electronegativity for the 4 B. When the material has less
                than 4 B, the columns are repeated.
    """
    # Import the electronegativity and the group elements from the csv
    df = pd.read_csv(path_en, header=None, sep='\t')
    df.columns = ['name', 'en']
    element_grups = pd.read_csv(path_type, header=None)
    element_grups.columns = ['name', 'type']

    df = df.set_index('name')
    min_max_scaler = MinMaxScaler()
    scaled_value = min_max_scaler.fit_transform(df.values.reshape(-1, 1))
    df['En scaled'] = scaled_value
    En = df['En scaled']

    # Fill the vector list. This list is filled with 4-length vector
    vectors = []
    for formula in formula_list:
        element_list = re.findall('[A-Z][^A-Z]*', formula)

        # Create a developped formula. Example: [Fe2,O,Cl] -> [Fe,Fe,O,Cl]
        developped_formula = []
        for elem in element_list:
            element, number = split_number(elem)
            for i in range(number):
                developped_formula.append(element)

        # Create the electronegativity vector
        en_vec = []
        for j in range(len(developped_formula)):
            element = developped_formula[j]
            group = element_grups[element_grups["name"] == element]["type"].values[0]
            if(group == "B") or (group == "AB"):
                en_vec.append(En[element])

        # If group ABB and the first A element is AB, then remove the first.
        # Example: A can be from the group AB -> [AB,B,B]
        # If group AABBBB and the first or the two first are AB, then remove
        # the first or the two first
        if len(en_vec) == 3 or len(en_vec) == 5:
            en_vec = en_vec[1:]
        if len(en_vec) == 6:
            en_vec = en_vec[2:]
        # Fit the electronegativity vector to a 4-length vector
        element_vector = en_vec*int(4/len(en_vec))
        vectors.append(element_vector)
    return pd.DataFrame(index=formula_list, data=vectors,
                        columns=['B1', 'B2', 'B3', 'B4'])
    
    

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



# parses elements and number of occurences from the formula list
# and creates fractional vector
def create_fractionalvector_by_group(formula_list,
                                     all_elements_list, number_of_atoms_list):
    element_grups = pd.read_csv(path_type, header=None)
    element_grups.columns = ['name', 'type']
    vectors = []
    for formula, number_of_atoms in zip(formula_list, number_of_atoms_list):
        element_vector = [0]*len(all_elements_list)
        element_list = re.findall('[A-Z][^A-Z]*', formula)
        if(number_of_atoms == 2):
            for elem in element_list:
                element, number = split_number(elem)
                element_index = all_elements_list.index(element)
                group = element_grups[element_grups["name"] == element]["type"].values[0]
                if(group == "A"):
                    element_vector[element_index] = element_vector[element_index] + 1 * number
                else:
                    element_vector[element_index] = element_vector[element_index] + 1 * number
        elif(number_of_atoms == 3):
            for elem in element_list:
                element, number = split_number(elem)
                element_index = all_elements_list.index(element)
                group = element_grups[element_grups["name"] == element]["type"].values[0]
                if(group == "A"):
                    element_vector[element_index] = element_vector[element_index] + 1 * number
                else:
                    element_vector[element_index] = element_vector[element_index] + 0.5 * number

        elif(number_of_atoms == 4):
            for elem in element_list:
                element, number = split_number(elem)
                element_index = all_elements_list.index(element)
                group = element_grups[element_grups["name"] == element]["type"].values[0]
                if(group == "A"):
                    element_vector[element_index] = element_vector[element_index] + 0.5 * number
                else:
                    element_vector[element_index] = element_vector[element_index] + 0.5 * number
        elif(number_of_atoms == 6):
            for elem in element_list:
                element, number = split_number(elem)
                element_index = all_elements_list.index(element)
                group = element_grups[element_grups["name"]==element]["type"].values[0]
                if(group=="A"):
                    element_vector[element_index]=element_vector[element_index]+0.5*number
                else:
                    element_vector[element_index]=element_vector[element_index]+0.25*number

        vectors.append(element_vector)    
    onehotvector=pd.DataFrame(index = formula_list, data=vectors, columns=all_elements_list)

    return onehotvector

#Check the formula is belongs to defined groups
def check_by_group(formula):
    
    element_grups = pd.read_csv(path_type, header=None)
    element_grups.columns = ['name', 'type']
    numA = 0
    numB = 0

    element_list=re.findall('[A-Z][^A-Z]*', formula)
    for elem in element_list:
        element, number = split_number(elem)                
        group=element_grups[element_grups["name"]==element]["type"].values[0]                
        if(group=="A"):
            numA=numA+number
        if(group=="B"):
            numB=numB+number

    numT = numA*10+numB

    if(numT in [11,12,22,24]):
        return True
    else:
#        print(formula+": A"+str(numA)+"B"+str(numB))
        return False
