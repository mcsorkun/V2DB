#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 19:15:42 2019

Project: V2DB (Virtual 2D Material Database)
Content: Main file for prediction of properties and storing materials into SQLite db

@author: severin astruc, murat cihan sorkun
"""
import sqlite3
import pandas as pd
import numpy as np
import iodb
from sklearn.externals import joblib
from os import listdir
import time


print('#####DATABASE CREATION######')
print('Importing data')
# Initialization of the differents directories.
path = '../../'
path_results = path +'results/'
path_models = path_results + 'models/'
path_encoders = path_results + 'encoders/'
path_materials = path_results + 'generation/'
path_data = path + 'data/'

# Get all the CSV file of generated materials
files_list = listdir(path_materials)

# Import the ML models for 8 properties. 
# CBM, ismetal, and ismagnetic  do not need ML model because there are 
# directly calculated with the other properties.
bandgap_model = joblib.load(path_models + "bandgap_model.sav")
dirgap_model = joblib.load(path_models + "dirbandgap_model.sav")
ehull_model = joblib.load(path_models + "ehull_model.sav")
formenergy_model = joblib.load(path_models + "heatform_model.sav")
magnstate_model = joblib.load(path_models + "magnstate_model.sav")
stability_model = joblib.load(path_models + "stability_model.sav")
vbm_model = joblib.load(path_models + "vbm_model.sav")
workfunc_model = joblib.load(path_models + "workfunc_model.sav")


# Set the list of element used for all bandgap, positive bandgap and zero bandgap.
elements_df = pd.read_csv(path_data+"elements.csv")
elements_list = elements_df['elements'].tolist()

nonzero_elements_df = pd.read_csv(path_data+"elements.csv")
nonzero_elements_list = elements_df['elements'].tolist()


print('Predicting properties')


totaltime = 0
for file in files_list:
    start = time.time()
    print("Prediction for " + file + " has been started.")
    materials_df = pd.read_csv(path_materials + file, header=None)
    material_list = materials_df[0].tolist()
    prototype_list = materials_df[1].tolist()

    input_data_en = iodb.get_input_data(materials_df, elements_list, EN=True)
    input_data_na = iodb.get_input_data(materials_df, elements_list, NA=True)
    input_data_en_na = iodb.get_input_data(materials_df, elements_list, EN=True, NA=True)
    
    nonzero_input_data_na = iodb.get_input_data(materials_df, nonzero_elements_list, NA=True)
    nonzero_input_data_en = iodb.get_input_data(materials_df, nonzero_elements_list, EN=True)


    print('####Properties prediction####')

    print('    Stability')
    stability_list = stability_model.predict(input_data_na)

    print('    Gap')
    bandgap_list = bandgap_model.predict(input_data_en_na)
    bandgap_list_n0 = np.zeros(len(bandgap_list))
    for i, y in enumerate(bandgap_list):
        bandgap_list_n0[i] = max([0, y])
     
    print('    Is metallic')
    upper, lower = 1, 0
    ismetal_list = np.where(bandgap_list_n0 == 0, upper, lower )    
    
    print('    VBM')
    VBM_list = vbm_model.predict(nonzero_input_data_na)
    
    print('    CBM')
    CBM_list = np.array(VBM_list) + np.array(bandgap_list_n0)
    
    print('    Work function')
    work_function_list = workfunc_model.predict(input_data_en)

    print('    Heat of formation')
    formation_energy_list = formenergy_model.predict(input_data_na)

    print('    Energy above convex hull')
    ehull_list = ehull_model.predict(input_data_en_na)
    
    print('    Has a direct band gap')
    has_direct_bandgap_list = dirgap_model.predict(nonzero_input_data_en)

    print('    Magnetic State')
    magnstate_list = magnstate_model.predict(input_data_na)
    
    print('    Is magnetic')
    upper, lower = 1, 0
    ismagn_list = np.where(magnstate_list == "NM", lower, upper)
    
    stacked_data = list(zip(stability_list,
                                    material_list,
                                    prototype_list,
                                    bandgap_list_n0,
                                    ismetal_list,
                                    formation_energy_list,
                                    VBM_list,
                                    ehull_list,
                                    CBM_list,
                                    has_direct_bandgap_list,
                                    magnstate_list,
                                    ismagn_list,
                                    work_function_list))    
    
    df_data=pd.DataFrame(stacked_data)


#   Filter unstable materials (based on Stability, heat of formation, and convex hull)    
    df_data=df_data[df_data[0]==1]
    df_data=df_data[df_data[5]<0.2]
    df_data=df_data[df_data[7]<0.2]

    df_data=df_data.drop(df_data.columns[0], axis=1)
    stacked_data=df_data.values
    
    
#   Write results into sqlite DB  
    conn = sqlite3.connect(path_results+'v2db.db')
    c = conn.cursor()
    try:
        sql = '''INSERT INTO materials (Material,Prototype,Band_Gap,Material_is_metallic,Heat_of_formation,VBM,Energy_above_convex_hull,CBM,Material_has_direct_gap,Magnetic_state,Material_is_magnetic,Work_function) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        c.executemany(sql, stacked_data)
    except sqlite3.IntegrityError as e:
        print('sqlite error: ', e.args[0])
    conn.commit()

    conn.close()

    print("Generation for " + file + " has been completed.\n")
    print("Total: " + str(len(df_data)) + " materials.")
    stop=time.time()
    time_elapse=stop-start
    totaltime += time_elapse
    print("Total time %.3f minutes.\n" % (time_elapse/60) )

print("Total time all script %.3f minutes.\n" % (totaltime/60) )

