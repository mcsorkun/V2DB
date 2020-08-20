# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:31:17 2019

Project: V2DB (Virtual 2D Material Database)
Content: Main file which trains and exports the machine learning models for property prediction

@author: severin astruc, murat cihan sorkun
"""
import preprocessing
import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


print('#####MODEL TRAINING#####')
path_models = '../../results/models/'
path_encoder = '../../results/encoders/'


# turn it into True to see cross_val results (time consuming)
make_cross_val=False
# turn it into True to see cross_val plots (requires make_cross_val=True)
make_plot=False 



# =============================================================================
# Stability
# =============================================================================
print("\nStability Clasification")
name = 'stability'
file_path = path_models + name + '_model.sav'
prototypes, formulas, features, x_stab = preprocessing.data_import(target='Stab')
feature_keys = ['xnumbat', 'onehot_xproto', 'fractional_xnumbat']

mlp = MLPClassifier(activation='relu', alpha=0.1,
                    hidden_layer_sizes=400, max_iter=400,
                    random_state=0, solver='adam')

X, y = preprocessing.get_data(features, feature_keys, x_stab)

true_values_stability=[]
pred_values_stability=[]


if make_cross_val:
    
    cross_val = KFold(n_splits=20,shuffle=True,random_state=0)   
    accuracies=[]
    f1_scores=[]
    for train_index, test_index in cross_val.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
    
        accuracy,f1=metrics.score_classification(y_test, y_pred, binary=True, verbose=0)
        accuracies.append(accuracy)
        f1_scores.append(f1)
        true_values_stability.append(y_test)
        pred_values_stability.append(y_pred)
              
    true_values_stability=np.concatenate(true_values_stability)
    pred_values_stability=np.concatenate(pred_values_stability)  
    
        
    # CV Results    
    accuracy,f1=metrics.score_classification(true_values_stability, pred_values_stability, binary=True, verbose=0)
    print('CV Test score (ACC):%.3f' % accuracy)
    print('CV Test score (F1):%.3f\n' % f1)
     

#export the model by training with full data
mlp = MLPClassifier(activation='relu', alpha=0.1,
                    hidden_layer_sizes=400, max_iter=400,
                    random_state=0, solver='adam')

mlp.fit(X, y)
joblib.dump(mlp, file_path)


## =============================================================================
## Egap
## =============================================================================
print("\nBandgap Regression")   
name = 'bandgap'
file_path = path_models + name + '_model.sav'
prototypes, formulas, features, x_gap = preprocessing.data_import(save=True)
feature_keys = ['xnumbat','en_AB','onehot_xproto', 'fractional_xnumbat']

mlp=MLPRegressor(activation='relu', alpha=0.01,
                    hidden_layer_sizes=1200, max_iter=400,
                    random_state=0, solver='adam')

X, y = preprocessing.get_data(features, feature_keys, x_gap)


true_values_gap=[]
pred_values_gap=[]

if make_cross_val:

    cross_val = KFold(n_splits=20,shuffle=True,random_state=0)

    r2_scores=[]
    mae_scores=[]
    rmse_scores=[]
    
    for train_index, test_index in cross_val.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        
        y_predicted_n0 = np.zeros(len(y_pred))
        for i, j in enumerate(y_pred):
            y_predicted_n0[i] = max([0, j])
            
    
        r2,mae,rmse=metrics.score_regression(y_test, y_predicted_n0, verbose=0)
        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        true_values_gap.append(y_test)
        pred_values_gap.append(y_predicted_n0)
        
        
    true_values_gap=np.concatenate(true_values_gap)
    pred_values_gap=np.concatenate(pred_values_gap)  
    r2,mae,rmse=metrics.score_regression(true_values_gap, pred_values_gap, verbose=0)

    
    # CV Results
    print('CV Test score (R2):%.3f' % r2)
    print('CV Test score (MAE):%.3f' % mae)
    print('CV Test score (RMSE):%.3f\n' % rmse)


#export the model by training with full data
mlp=MLPRegressor(activation='relu', alpha=0.01,
                    hidden_layer_sizes=1200, max_iter=400,
                    random_state=0, solver='adam')
mlp.fit(X, y)
joblib.dump(mlp, file_path)


# =============================================================================
# Import the encoder created with bandgap model training. We use for the each
# property these encoder to have the same preprocessing step. Indeed, some
# properties are defined only for positive or negative bandgap. Thus, some
# columns from the features can disappear because not used. With the saved
# encoders, this issue is resolved.
# =============================================================================
prototype_label_encoder_all = joblib.load(path_encoder + "label_encoder_xproto_all.sav")
prototype_onehot_encoder_all = joblib.load(path_encoder + "onehot_encoder_xproto_all.sav")
elements_list = joblib.load(path_encoder + "frac_xnumbat_all.sav")

# The 3 following features are not fitted (False) but only transformed with
# the encoders and onehot encoders.
# {feature: [label encoder, onehot encoder, fit_transform or not (transform)]}
encoders = {'xproto': [prototype_label_encoder_all,
                   prototype_onehot_encoder_all, False],
            'frac_xnumbat': [elements_list, False]}

## =============================================================================
## VBM
## =============================================================================
print("\nVBM Regression")
name = 'vbm'
file_path = path_models + name + '_model.sav'
feature_keys = ['xnumbat', 'onehot_xproto', 'fractional_xnumbat']
prototypes, formulas, features, x_vbm = preprocessing.data_import(select='gap>0', target='VBM', encoders=encoders)
X, y = preprocessing.get_data(features, feature_keys, x_vbm)

mlp = MLPRegressor(activation='relu', alpha=0.025,
                             hidden_layer_sizes=(400), max_iter=400,
                             solver='adam', random_state=0)

true_values_vbm=[]
pred_values_vbm=[]


if make_cross_val:

    cross_val = KFold(n_splits=20,shuffle=True,random_state=0)

    r2_scores=[]
    mae_scores=[]
    rmse_scores=[]
    
    for train_index, test_index in cross_val.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
    
        r2,mae,rmse=metrics.score_regression(y_test, y_pred, verbose=0)
        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        true_values_vbm.append(y_test)
        pred_values_vbm.append(y_pred)


    true_values_vbm=np.concatenate(true_values_vbm)
    pred_values_vbm=np.concatenate(pred_values_vbm)  
    r2,mae,rmse=metrics.score_regression(true_values_vbm, pred_values_vbm, verbose=0)
    
        
    # CV Results
    print('CV Test score (R2):%.3f' % r2)
    print('CV Test score (MAE):%.3f' % mae)
    print('CV Test score (RMSE):%.3f\n' % rmse)
    

#export the model by training with full data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=1,
                                                    test_size=0.05)
mlp.fit(X_train, y_train)
joblib.dump(mlp, file_path)


## =============================================================================
## CBM 
## The model is not saved because it is calculated as CBM=bandgap+VBM
## =============================================================================
print("\nCBM Regression")
prototypes, formulas, features, x_cbm = preprocessing.data_import(select='gap>0', target='CBM', encoders=encoders)
feature_keys = ['xnumbat','en_AB','onehot_xproto', 'fractional_xnumbat']
X, y_cbm = preprocessing.get_data(features, feature_keys, x_cbm)

#import bandgap model
bandgap_model = joblib.load(path_models + "bandgap_model.sav")

pred_values_gap_for_cbm=[]
true_values_cbm=[]

if make_cross_val:

    cross_val = KFold(n_splits=20,shuffle=True,random_state=0)

    r2_scores=[]
    mae_scores=[]
    rmse_scores=[]
    
    for train_index, test_index in cross_val.split(X):
        
        ## Predict the band gap
        
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y_cbm[train_index], y_cbm[test_index]
        y_predicted = bandgap_model.predict(X_test)
        ## Set the negative value to 0
        y_predicted_n0 = np.zeros(len(y_predicted))
        for i, y in enumerate(y_predicted):
            y_predicted_n0[i] = max([0, y])
            
        pred_values_gap_for_cbm.append(y_predicted_n0)
        true_values_cbm.append(y_test)
        

    pred_values_gap_for_cbm=np.concatenate(pred_values_gap_for_cbm)
    true_values_cbm=np.concatenate(true_values_cbm)
    pred_values_cbm=pred_values_vbm + pred_values_gap_for_cbm
    r2,mae,rmse=metrics.score_regression(true_values_cbm, pred_values_cbm, verbose=0)
    
        
    # CV Results
    print('CV Test score (R2):%.3f' % r2)
    print('CV Test score (MAE):%.3f' % mae)
    print('CV Test score (RMSE):%.3f\n' % rmse)
    
    #no model exported since it is calculated as CBM=bandgap+VBM


## =============================================================================
## Work function
## =============================================================================
print("\nWorkFunction Regression")
name = 'workfunc'
file_path = path_models + name + '_model.sav'
prototypes, formulas, features, x_wf = preprocessing.data_import(select='gap=0', target='WF', encoders=encoders)
feature_keys = ['en_AB','onehot_xproto', 'fractional_xnumbat']

mlp=MLPRegressor(activation='relu', alpha=0.01,
                    hidden_layer_sizes=1000, max_iter=200,
                    random_state=0, solver='adam')

X, y = preprocessing.get_data(features, feature_keys, x_wf)


true_values_work=[]
pred_values_work=[]


if make_cross_val:

    cross_val = KFold(n_splits=20,shuffle=True,random_state=0)
    
    r2_scores=[]
    mae_scores=[]
    rmse_scores=[]
    
    for train_index, test_index in cross_val.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
    
        r2,mae,rmse=metrics.score_regression(y_test, y_pred, verbose=0)
        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        true_values_work.append(y_test)
        pred_values_work.append(y_pred)       
        
    true_values_work=np.concatenate(true_values_work)
    pred_values_work=np.concatenate(pred_values_work)  
    r2,mae,rmse=metrics.score_regression(true_values_work, pred_values_work, verbose=0)
    
    # CV Results
    print('CV Test score (R2):%.3f' % r2)
    print('CV Test score (MAE):%.3f' % mae)
    print('CV Test score (RMSE):%.3f\n' % rmse)
          

#export the model by training with full data
mlp=MLPRegressor(activation='relu', alpha=0.01,
                    hidden_layer_sizes=1000, max_iter=200,
                    random_state=0, solver='adam')
mlp.fit(X, y)
joblib.dump(mlp, file_path)


## =============================================================================
## Heat of formation
## =============================================================================
print("\nHeat of Formation Regression")
name = 'heatform'
file_path = path_models + name + '_model.sav'
prototypes, formulas, features, x_hf = preprocessing.data_import(target='HF', encoders=encoders)
feature_keys = ['xnumbat', 'onehot_xproto', 'fractional_xnumbat']

mlp=MLPRegressor(activation='relu', alpha=0.025,
                    hidden_layer_sizes=1200, max_iter=100,
                    random_state=0, solver='adam')

X, y = preprocessing.get_data(features, feature_keys, x_hf)


true_values_hof=[]
pred_values_hof=[]

if make_cross_val:
    
    cross_val = KFold(n_splits=20,shuffle=True,random_state=0)
    
    r2_scores=[]
    mae_scores=[]
    rmse_scores=[]
    
    for train_index, test_index in cross_val.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
    
        r2,mae,rmse=metrics.score_regression(y_test, y_pred, verbose=0)
        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        true_values_hof.append(y_test)
        pred_values_hof.append(y_pred)       
        
    true_values_hof=np.concatenate(true_values_hof)
    pred_values_hof=np.concatenate(pred_values_hof)  
    r2,mae,rmse=metrics.score_regression(true_values_hof, pred_values_hof, verbose=0)
    
    # CV Results
    print('CV Test score (R2):%.3f' % r2)
    print('CV Test score (MAE):%.3f' % mae)
    print('CV Test score (RMSE):%.3f\n' % rmse)


#export the model by training with full data
mlp=MLPRegressor(activation='relu', alpha=0.025,
                    hidden_layer_sizes=1200, max_iter=100,
                    random_state=0, solver='adam')
mlp.fit(X, y)
joblib.dump(mlp, file_path)



## =============================================================================
## Energy above convex Hull
## =============================================================================
print("\nEnergy above convex Hull Regression")
name = 'ehull'
file_path = path_models + name + '_model.sav'
prototypes, formulas, features, x_eh = preprocessing.data_import(target='EHull', encoders=encoders)
feature_keys = ['xnumbat','en_AB','onehot_xproto', 'fractional_xnumbat']

mlp=MLPRegressor(activation='relu', alpha=0.01,
                    hidden_layer_sizes=1200, max_iter=100,
                    random_state=0, solver='adam')

X, y = preprocessing.get_data(features, feature_keys, x_eh)

true_values_hull=[]
pred_values_hull=[]

if make_cross_val:

    cross_val = KFold(n_splits=20,shuffle=True,random_state=0)
    
    r2_scores=[]
    mae_scores=[]
    rmse_scores=[]
    
    for train_index, test_index in cross_val.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
    
        r2,mae,rmse=metrics.score_regression(y_test, y_pred, verbose=0)
        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        true_values_hull.append(y_test)
        pred_values_hull.append(y_pred)  

    true_values_hull=np.concatenate(true_values_hull)
    pred_values_hull=np.concatenate(pred_values_hull)  
    r2,mae,rmse=metrics.score_regression(true_values_hull, pred_values_hull, verbose=0)
        
    # CV Results
    print('CV Test score (R2):%.3f' % r2)
    print('CV Test score (MAE):%.3f' % mae)
    print('CV Test score (RMSE):%.3f\n' % rmse)
    

#export the model by training with full data
mlp=MLPRegressor(activation='relu', alpha=0.01,
                    hidden_layer_sizes=1200, max_iter=100,
                    random_state=0, solver='adam')
mlp.fit(X, y)
joblib.dump(mlp, file_path)


## =============================================================================
## Direct band gap
## =============================================================================
print("\nDirect band gap Clasification")
name = 'dirbandgap'
file_path = path_models + name + '_model.sav'
prototypes, formulas, features, x_d = preprocessing.data_import(select='gap>0', target='D', encoders=encoders)
feature_keys = ['en_AB','onehot_xproto', 'fractional_xnumbat']

true_values_direct=[]
pred_values_direct=[]


mlp = MLPClassifier(activation='relu', alpha=0.001,
                    hidden_layer_sizes=800, max_iter=400,
                    random_state=0, solver='adam')

X, y = preprocessing.get_data(features, feature_keys, x_d)

if make_cross_val:

    cross_val = KFold(n_splits=20,shuffle=True,random_state=0)
    
    accuracies=[]
    f1_scores=[]
    for train_index, test_index in cross_val.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
    
        accuracy,f1=metrics.score_classification(y_test, y_pred, binary=True, verbose=0)
        accuracies.append(accuracy)
        f1_scores.append(f1)
        true_values_direct.append(y_test)
        pred_values_direct.append(y_pred)
        
        
    true_values_direct=np.concatenate(true_values_direct)
    pred_values_direct=np.concatenate(pred_values_direct)  
    
   
    # CV Results    
    accuracy,f1=metrics.score_classification(true_values_direct, pred_values_direct, binary=True, verbose=0)
    print('CV Test score (ACC):%.3f' % accuracy)
    print('CV Test score (F1):%.3f\n' % f1)
    
   
#export the model by training with full data   
mlp = MLPClassifier(activation='relu', alpha=0.001,
                    hidden_layer_sizes=800, max_iter=400,
                    random_state=0, solver='adam')
mlp.fit(X, y)
joblib.dump(mlp, file_path)


## =============================================================================
## Magnetic State
## =============================================================================
print("\nMagnetic State Clasification")
name = 'magnstate'
file_path = path_models + name + '_model.sav'
prototypes, formulas, features, x_MS = preprocessing.data_import(target='MS',
                                           encoders=encoders)
feature_keys = ['xnumbat', 'onehot_xproto', 'fractional_xnumbat']

true_values_magnetic=[]
pred_values_magnetic=[]

mlp = MLPClassifier(activation='relu', alpha= 0.1,
                    hidden_layer_sizes=400, max_iter=100,
                    random_state=0, solver='adam')

X, y = preprocessing.get_data(features, feature_keys, x_MS)

if make_cross_val:

    cross_val = KFold(n_splits=20,shuffle=True,random_state=0)
    
    accuracies=[]
    f1_scores=[]
    for train_index, test_index in cross_val.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
    
        accuracy,f1=metrics.score_classification(y_test, y_pred, verbose=0)
        accuracies.append(accuracy)
        f1_scores.append(f1)
        true_values_magnetic.append(y_test)
        pred_values_magnetic.append(y_pred)
        
        
    true_values_magnetic=np.concatenate(true_values_magnetic)
    pred_values_magnetic=np.concatenate(pred_values_magnetic)
     
    
    # CV Results    
    accuracy,f1=metrics.score_classification(true_values_magnetic, pred_values_magnetic, binary=False, verbose=0)
    print('CV Test score (ACC):%.3f' % accuracy)
    print('CV Test score (F1):%.3f\n' % f1)
    

#export the model by training with full data    
mlp = MLPClassifier(activation='relu', alpha= 0.1,
                    hidden_layer_sizes=400, max_iter=100,
                    random_state=0, solver='adam')
mlp.fit(X, y)
joblib.dump(mlp, file_path)



if make_plot:

    label_size=18
    font_size=24
 
    
    metrics.plot_regression(pred_values_hof,true_values_hof,"ML heat of formation (eV/atom)","DFT heat of formation (eV/atom)"
                            , label_size=label_size, font_size=font_size, line_start=-4, line_end=2)    
    metrics.plot_regression(pred_values_hull,true_values_hull,"ML energy above convex hull (eV/atom)","DFT energy above convex hull (eV/atom)"
                            , label_size=label_size, font_size=font_size, line_start=-2, line_end=3)
    metrics.plot_regression(pred_values_work,true_values_work,"ML work function (eV)","DFT work function (eV)"                
                            , label_size=label_size, font_size=font_size, line_start=2, line_end=8)
    metrics.plot_regression(pred_values_gap,true_values_gap,"ML band gap (eV)","DFT band gap (eV)"
                            , label_size=label_size, font_size=font_size, line_start=-0, line_end=6)
    metrics.plot_regression(pred_values_vbm,true_values_vbm,"ML VBM (eV)","DFT VBM (eV)"
                            , label_size=label_size, font_size=font_size, line_start=-9, line_end=-2)
    metrics.plot_regression(pred_values_cbm,true_values_cbm,"ML CBM (eV)","DFT CBM (eV)"
                            , label_size=label_size, font_size=font_size, line_start=-9, line_end=-2)


    label_size=18 #labels andright bar values
    font_size=24 #titles of lable
    number_size=30

    metrics.plot_confusion_matrix(true_values_stability,pred_values_stability,["Unstable","Stable"],title="Stability (Accuracy=0.881)", cmap=plt.cm.Greens
                                  , label_size=label_size, font_size=font_size, number_size=number_size)   
    metrics.plot_confusion_matrix(true_values_direct,pred_values_direct,["No","Yes"],title="Has direct band gap (Accuracy=0.883)", cmap=plt.cm.Reds
                                  , label_size=label_size, font_size=font_size, number_size=number_size)
    metrics.plot_confusion_matrix(true_values_magnetic,pred_values_magnetic,["AFM","FM","NM"],title="Magnetic state (Accuracy=0.787)", cmap=plt.cm.Purples
                                  , label_size=label_size, font_size=font_size, number_size=number_size)    








