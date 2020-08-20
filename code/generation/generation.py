# -*- coding: utf-8 -*-
"""
Created on Wed Jul 3 15:36:58 2019

Project: V2DB (Virtual 2D Material Database)
Content: Main file for generation of 2D materials.

@author: severin astruc, murat cihan sorkun
"""
import v2db
import filtering
import iodb

# =============================================================================
# Initialization
# =============================================================================
print('#####GENERATION#####')
print('Initialization')
path_data = '../../data/'
path_subparts = '../../results/subparts/'
path_results = '../../results/generation/'
charges_csv = path_data+'charges.csv'
symmetries_csv = path_data+'symmetries.csv'
types_csv = path_data+'types.csv'
groups_csv = path_data+'groups.csv'

subgroups = ['A', 'B', 'AA', 'BB']


# Import properties of each element from csv files
element_properties = iodb.get_element_properties(charges_csv, types_csv)

# Create list of element instance
element_list = v2db.element_list(element_properties)

# Separate the A and B element
element_type = v2db.get_elt_type(element_list)

# =============================================================================
# Compute subgroup A, AA, AAr, BB, BBr and BBBBr (r means symmetry filtered)
# =============================================================================
print('Compute subgroup')
tree_dic = {}
sub_mat = {}
for subgroup in subgroups:
    # Compute the subgroup with duplicate
    tree_dic[subgroup] = v2db.MatrixTree(subgroup, element_type[0],
                                         element_type[1])
    sub_mat[subgroup] = tree_dic[subgroup].generated_submat

# Create the AAr, BBr and BBBBr
sub_mat['AAR'] = filtering.sub_mask(tree_dic['AA'])
sub_mat['BBR'] = filtering.sub_mask(tree_dic['BB'])

# Create the BBBB to create the BBBr, BBBB is not used in the generation part
tree_dic['BBBB'] = v2db.MatrixTree('B4', [], sub_mat['BBR'])
sub_mat['BBBBR'] = filtering.sub_mask(tree_dic['BBBB'])

# Save the subparts
iodb.save_subpart(sub_mat, path_subparts)
# =============================================================================
# Generation of new material according to the symmetry and the group.
# Prototypes with the same label and symmetry have the same set of generated
# materials.
# =============================================================================
print('Generation of new materials')
symmetries = iodb.get_symmetries(symmetries_csv)
# The symmetries array is the following:
# array([['AB', 'Z'],
#        ['ABB', 'Z'],
#        ['ABB', 'Y'],
#        ['AABB', 'Y'],
#        ['AABB', 'XY'],
#        ['AABB', 'Z'],
#        ['AABBBB', 'XYY'],
#        ['AABBBB', 'XY']])
size_generation = []

# Here we can select the couple group/symmetry to generate the new materials by
# changing the value inside the symmetries bracket. Value range from 0 to 7 to
# select the value in the last showed array.
# It is possible to use all of them (without bracket) but the last two can be very long
# to compute because they create around 15 millions materials. If downloaded
# in personal computer, the RAM cannot be enought to run it depending on the
# hardware.
# for group, label in symmetries:
print('[group, symmetry label, duplicate filtering size, neutral filtering size]')
for group, label in symmetries[0:3]:
    # Generate the new materials following the group and the label.
    new_materials = v2db.generate(sub_mat, group, label)

    # Neutral filtering of the new materials.
    nf_materials = filtering.neutrality_reduction(new_materials, group)

    # Set the new materials for each prototype that have the same group
    # and label.
    prototypes = iodb.get_prototype(groups_csv, group, label)
    print([group, label, len(new_materials), len(nf_materials)])
    size_generation.append([group, label, len(new_materials),
                            len(nf_materials)])
    iodb.save_csv(nf_materials, prototypes, group,
                  file_dir=path_results)
