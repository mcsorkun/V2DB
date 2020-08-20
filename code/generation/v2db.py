# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:30:24 2019

Project: V2DB (Virtual 2D Material Database)
Content: Classes and functions to generate new 2D materials

@author: astruc severin


class
-----
MatrixTree:
    Generate submaterials with A and B element via tree construction
Element:
    Get the charges of an element and his type
Material:
    Store all the properties of a material and find is neutrality.

functions
---------
tree:
    Make a tree with the coefficient of each subtree.
element_list:
    Transform dataframe of elements to a list of element class.
get_elt_type:
    Give a list of A and B from a list of element
generate(subgroup_list, group_name, label):
    Generate new 2D materials from a subgroup list [A, AA, BB, ...], the group
    and the label.
group_subdivision(subgroup_list, group_name, label):
    Get the right subgroups to generate new materials for a prototype
    (define by a group and a label)
"""
import numpy as np
from itertools import product


class MatrixTree:
    """
    Generate submaterials with A and B element via tree construction

    It allows to set the value of the nodes via the structure of
    submaterial.
    Nodes [[1,2],[2,1]] give:
                /\\
               1  2
              /\\  /\\
             2  1 2 1

    Attributes
    ----------
    subgroup : str
        subgroup name taken with A type and B
        type. Writed only with A and B (A, AA, B, BB,...).
        It defines the tree.
    depth : int
        Depth of the tree = size of the subgroup
    usable_element: list of element
        list of elements that can be use for each part of the subgroup
    generated_submat: list of list of element
        list of material that have been generated for the submaterial

    Methods
    -------
    get_usable_element(A, B):
        Get the 2D list of element that can be used for each position.
    generate_materials():
        Generate all the materials by substitution of each element from
        the prototype.
    get_number_unique_element(): int
        Get the number of element used into generation
    """

    def __init__(self, subgroup, A, B):
        """
        Parameters
        ----------
        subgroup : str
            name of the subgroup
        A : list of element
            list of the A type elements
        B : list of element
            list of the B type elements
        """
        self.subgroup = subgroup

        # depth is the number of level of the tree
        self.depth = len(subgroup)

        self.usable_element = []
        self.generated_submat = []

        self.get_usable_element(A, B)
        self.generate_material()

        # If we generate submaterials from AA or BB. Then the A or B part is
        # a list and when we use our tree algorithm we obtain a list of list
        # Code below flat this list of list to a list
        if (type(B) or type(A)) is np.ndarray:
            self.generated_submat = self.generated_submat.transpose(1, 0, 2)
            tmp = []
            for i in range(len(self.generated_submat)):
                # Flat the list of list to a list
                tmp.append([x for sublist in self.generated_submat[i]
                            for x in sublist])
            self.generated_submat = np.array(tmp)

    def get_usable_element(self, A, B):
        """
        Get the 2D list of element that can be used for each position.
        """
        for element in self.subgroup:
            if element is 'A':
                self.usable_element.append(A)
            else:
                self.usable_element.append(B)

    def generate_material(self):
        """
        Generate all the materials by substitution of each element from the
        prototype.
        """
        self.generated_submat = tree(self.usable_element)

    def get_number_unique_element(self):
        """ Get the number of element used into generation. """
        return len(self.usable_element[0])

    def __str__(self):
        """
        Overloading of print function.
        """
        for elt in self.generated_submat:
            print(elt)
        return 'Done'

    def __repr__(self):
        """ Overloading of repr methods."""
        return str(self)


class Element:
    """
    Get the charges of an element and its type

    Attributes
    ----------
    element : str
        Name of the element
    charge : list of int
        Possible charges for the element
    type: str
        type of the element: A, B or both type
    """

    def __init__(self, element, charge_list, default_type=''):
        """
        Parameters
        ----------
        element : str
            Name of the element
        charge_list : list of int
            All charges for the element.
        default_type: str
            type of the element: A, B or both type, (default is
            no type)
        """
        self.element = element
        self.charge = charge_list
        self.type = default_type

    def __str__(self):
        """ Overloading of print function."""
        return self.element

    def __repr__(self):
        """ Overloading of repr methods."""
        return str(self)


class Material:
    """
    Store all the properties of a material and find is neutrality.

    Attributes
    ----------
    group: str
        Name of the group of the material
    elements: list of elements
        List of each element of the compound
    charge: 2D array of int
        List all of the possibilities to have a charge (sum of charges)
        equal to zero. Columns: charge for one elements of the material,
        Rows : combination of charges for the materials
    neutral: boolean
        Indicate if the material can be neutral (sum of charges is 0)
        or not

    Methods
    -------
    isneutral()
        return if the material is neutral or not
    """
    def __init__(self, elements, group):
        """
        Parameters
        ----------
        elements: list of elements
            List of each element of the compound
        group: str
            Name of the group of the material
        """
        self.elements = elements
        self.group = group
        self.charge = self.__find_charges()
        self.neutral = self.isneutral()

    def __find_charges(self):
        """
        Find all possible charges from element with a sum of zero.

        List all of the possibilities to have a charge (sum of charges)
        equal to zero. Columns: charge for one elements of the material,
        Rows : combination of charges for the materials

        For all element in the material example: ['Mo', 'S', 'S']
        Get all the charges and create a tree. First level, the
        charges of Mo, second level, charges of S for each subtree
        under a node of one Mo charge...
        """
        length = len(self.elements)
        charges_map = [[]]*length

        if length == 1:
            # Case of one element in the material
            return self.elements[0].charge

        for i in range(length):
            # for each level, determine all the nodes
            k1 = 1
            k2 = 1
            tmp = []
            for k in range(i+1, length):
                # coefficient of repetition of the nodes for a level
                k2 *= len(self.elements[k].charge)
            for k in range(i):
                k1 *= len(self.elements[k].charge)
            for j in range(len(self.elements[i].charge)):
                # find the sequence to complete the level under one node
                tmp += [self.elements[i].charge[j]] * k2
                # repeat the sequence k1 time to complete the level
            charges_map[i] = tmp*k1
        charges_map = np.array(charges_map).T
        # return all the charges list for which the sum is 0
        return charges_map[(np.sum(charges_map, axis=1)) == 0]

    def isneutral(self):
        """
        return if the material is neutral or not.
        """
        if len(self.charge) != 0:
            return True
        else:
            return False

    def __str__(self):
        """ Overloading of print function."""
        string = ''
        for i in self.name:
            string += ''.join(i)
        return string

    def __repr__(self):
        """ Overloading of repr methods."""
        return str(self)


def tree(l):
    """
    Make a tree with the coefficient of each subtree.

    if l is [[1,2],[2,3]], it gives:
                    /\\
                   1  2
                  /\\  /\\
                 2  3 2 3

    Parameters:
    ----------
    l: list of objects/list
        list of the object or list which describe each subtree of a level.
        Object have to work like a list.
    """
    tree = [[]]*len(l)
    for i in range(len(l)):
        # for each level, determine all the nodes
        k1 = 1
        k2 = 1
        tmp = []
        for k in range(i+1, len(l)):
            # coefficient of repetition of the nodes for a level
            k2 *= len(l[k])
        for k in range(i):
            k1 *= len(l[k])
        for j in range(len(l[i])):
            # find the sequence to complete the level under one node
            tmp += [l[i][j]] * k2
            # repeat the sequence k1 time to complete the level
        tree[i] = tmp*k1
    return np.array(tree).T


def element_list(elements_dataframe):
    """
    Transform dataframe of elements to a list of element class.

    Parameters:
    ----------
    elements_dataframe: pandas dataframe
        DataFrame with the data(name,type,charge) for each element
    """
    elements = []
    for i in elements_dataframe.index:
        d = elements_dataframe.iloc[i]
        elements.append(Element(d[0], d[1], default_type=d[2]))
    return elements


def get_elt_type(element_list):
    """
    Give a list of A element and B element from a list of element.
    There is the case of AB element which are in both list. But before
    give them in the list, it is necessary to remove the positive charges
    in the B list and the negative values when we give the AB element to the
    A list.
    """
    A_type = []
    B_type = []
    for elt in element_list:
        if elt.type == 'A':
            A_type.append(elt)
        if elt.type == 'B':
            B_type.append(elt)
        if elt.type == 'AB':
            # When AB we add the element in both list: A list and
            # B list
            charge_array = np.array(elt.charge)
            # Take only the positive charge of the element
            mask = charge_array > 0
            A_part = Element(elt.element, list(charge_array[mask]),
                             default_type='A')
            A_type.append(A_part)

            # Take only the negative charge of the element
            mask = np.invert(mask)
            B_part = Element(elt.element, list(charge_array[mask]),
                             default_type='B')
            B_type.append(B_part)
    return [A_type, B_type]


def generate(subgroup_list, group_name, label):
    """
    Generate new 2D materials from a subgroup list [A, AA, BB, ...], the group
    and the label.
    """
    # Before generating, it is necessary to get the right subgroups according
    # to the group name and the symmetry. Indeed, two group can have different
    # symmetry label, it implies to use reduced subgroup or not.
    A, B = group_subdivision(subgroup_list, group_name, label)
    new_material = []

    # Compute the product of the two subgroup. For example, take ABB, A part is
    # [Mo, W] and BB part is [SS, SSe, SeSe]. Then the concatenated product
    # is:
    # [ MoSS, MoSSe, MoSeSe, WSS, WSSe, WSeSe]
    for material in np.asarray(list(product(A, B))):
        new_material.append(np.concatenate(material))
    return new_material


def group_subdivision(subgroup_list, group_name, label):
    """
    Get the right subgroups to generate new materials for a prototype
    (define by a group and a label)
    """
    As, Bs = 0, 0

    # Count the number of A and B in the group
    for i in group_name:
        if i == 'A':
            As += 1
        else:
            Bs += 1

    # Default case is the group ABB
    A = subgroup_list['A']
    B = subgroup_list['BB']
    if As == 2 and (label == 'XY' or label == 'XYY'):
        A = subgroup_list['AAR']
    if As == 2 and (label == 'Y' or label == 'Z'):
        A = subgroup_list['AA']

    if Bs == 1:
        B = subgroup_list['B']
    if Bs == 2:
        if label == 'Z':
            B = subgroup_list['BB']
        else:
            B = subgroup_list['BBR']
    if Bs == 4:
        if label == 'XY':
            B = subgroup_list['BBBBR']
        else:
            B = []
            for b in list(product(subgroup_list['BBR'], repeat=2)):
                B.append(np.concatenate(b))
            B = np.array(B)
    return A, B
