# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:27:37 2019
Updated on Mon Jul  8 09:08:01 2019

Project: V2DB (Virtual 2D Material Database)
Content: input-output operations

@author: astruc

functions
---------
get_symmetries(sym_dir):
    Import symmetries label and group from CSV file.
get_prototype(groups_dir, group, label):
    Import prototypes for a group and a label from CSV file
list_strint(l):
    Convert a list of str to a list of int.
get_charges(charges_dir):
    Get the charges of each element from a CSV file
get_element_properties(charges_dir, types_dir):
    Get properties (charge, type A or B) of element from a CSV file.
save_csv(material_list, prototypes, group):
    Save all the materials for one prototype.
"""
import pandas as pd
import csv
from numpy import nan


def get_symmetries(sym_dir):
    """
    Import symmetries label and group from CSV file.
    """
    return pd.read_csv(sym_dir, sep='\t', header=None).values


def get_prototype(groups_dir, group, label):
    """
    Import prototypes for a group and a label from CSV file
    """
    df = pd.read_csv(groups_dir, sep='\t', header=None)
    df.columns = ['groups', 'prototypes', 'labels']
    df = df[df['labels'] == label]
    df = df[df['groups'] == group]
    return df['prototypes'].values


def list_strint(l):
    """ Convert a list of str to a list of int."""
    L = []
    for s in l:
        try:
            L.append(int(s))
        except ValueError:
            return nan
    return L


def get_charges(charges_dir):
    """
    Get the charges of each element from a CSV file
    Parameter
    ---------
    charges_dir: str
        Directorty of the CSV file
    """
    charges = []
    with open(charges_dir) as csvfile:
        readCSV = csv.reader(csvfile, delimiter='\t')
        for row in readCSV:
            charges.append([row[0], list_strint(row[1].split(','))])
    return charges


def get_element_properties(charges_dir, types_dir):
    """
    Get the properties (charge, type A or B) of element from CSV files.

    Parameters:
    ----------
    types_dir: str
        Directorty of the groups CSV file.
    charges_dir: str
        Directorty of the CSV file.
    """
    charges = get_charges(charges_dir)
    df_charges = pd.DataFrame(charges,
                              columns=['element',
                                       'charges']).set_index('element')
    df_types = pd.read_csv(types_dir, header=None)
    df_types.columns = ['element', 'types']
    df_types.set_index('element', inplace=True)
    df = pd.concat([df_charges, df_types], axis=1)
    df = df.dropna(subset=['charges'])
    df.reset_index(inplace=True)
    return df


def save_csv(material_list, prototypes, group, file_dir):
    """
    Save all the materials for one prototype.

    Parameters
    ----------
    name: str
        name of the file (without .csv)
    material_list: list of list of material
        list of material for each prototype, to save
    """
    for prototype in prototypes:
        # Save the same generated material for all the prototypes that share
        # the same label and same group

        # Initialize the name of the csv file according to the prototype
        directory = file_dir + group + '_' + prototype + '.csv'
        with open(directory, mode='w', newline='') as group_csv:
            group_writer = csv.writer(group_csv, delimiter=',')
            for i in range(len(material_list)):
                # Loop over all the material from material list
                name = ''
                for k in material_list[i]:
                    # Get the name of the material of size
                    # len(material_list[i])
                    name += k.element
                group_writer.writerow([name, prototype])


def save_subpart(subparts, fil_dir=''):
    """
    Save subpart in a csv file: A.csv B.csv AA.csv BB.csv AA(R).csv BB(R).csv
    BB(R)BB(R).csv (BB(R)BB(R))(R).csv
    """
    for subpart in subparts.keys():
        directory = fil_dir + subpart + '.csv'
        part_materials = subparts[subpart]
        with open(directory, mode='w', newline='') as group_csv:
            group_writer = csv.writer(group_csv, delimiter=',')
            for i in range(len(part_materials)):
                # Loop over all the material from the subpart
                name = ''
                for k in part_materials[i]:
                    # Get the name of the material of size
                    # len(material_list[i])
                    name += k.element
                group_writer.writerow([name])
