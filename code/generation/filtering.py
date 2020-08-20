# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:24:00 2019

@author: astruc

Project: V2DB (Virtual 2D Material Database)
Content: Symmetry and Neutrality filtering


functions
---------
sub_mask:
    Create a mask for symmetric filtering and apply it on a material tree.
neutrality_reduction(materials, group):
    Filter non neutral materials for one prototype materials list.
"""
from itertools import compress
import numpy as np
from v2db import Material


def neutrality_reduction(materials, group):
    """
    Filter non neutral materials for one prototype materials list.

    materials: list of materials
        list of materials for one prototype
    """
    f1_material_list = []
    for mat in materials:
        if Material(mat, group).neutral:
            f1_material_list.append(mat)
    return f1_material_list


def sub_mask(tree):
    """
    Create a mask for symmetric filtering and apply it on a material tree.

    The tree is build such a way that the symmetric element are periodic
    1 AA: keep    4 BA: remove  7 CA: remove
    2 AB: keep    5 BB: keep    8 CB: remove
    3 AC: keep    6 BC: keep    9 CC: keep
    """
    number_material = len(tree.generated_submat)
    # number_atoms = Number of atoms in the same material
    number_atoms = tree.get_number_unique_element()

    # mask indicates for each material if we keep it or not
    mask = [1]*number_material
    # size = length of a sequence of the same element
    size = int(number_material/number_atoms)
    for i in range(size):
        # create the mask for each first element (columns in description)
        mask[size*i:size*i+i] = [0]*i
    # return an array of the tree list with a binary mask filtering
    return np.array(list(compress(tree.generated_submat, mask)))
