# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:29:16 2019

Project: V2DB (Virtual 2D Material Database)
Content: prints and plots the metrics for regression and classification

@author: severin astruc, murat cihan sorkun
"""
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score,
                             roc_auc_score, r2_score,
                             mean_absolute_error, mean_squared_error)
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np



def score_classification(y_true, y_pred, binary=False, verbose=1):
    """
    This function prints and returns confusion matrix and the metrics of the classifier.
    """
    if verbose>0:
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        print('Confusion matrix')
    
        print(cm)
        print('Accuracy : {:.3f}'
              .format(accuracy_score(y_true, y_pred)))
        if binary:
            print('ROC AUC Score: {:.3f}'
                  .format(roc_auc_score(y_true, y_pred)))
            print('f1 Score: {:.3f}'
                  .format(f1_score(y_true, y_pred)))
        else:
            print('f1 Score: {:.3f}'
                  .format(f1_score(y_true, y_pred, average='micro')))
    
    if binary:               
        return accuracy_score(y_true, y_pred),f1_score(y_true, y_pred)
    else:
        return accuracy_score(y_true, y_pred),f1_score(y_true, y_pred, average='micro')


def get_accuracy(y_true, y_pred):
    """
    This function returns accuracy.
    """     
    return accuracy_score(y_true, y_pred)


def score_regression(y_true, y_pred, verbose=1):
    """
    This function prints and returns the metrics for a regressor.
    """
    if verbose>0:
        print('R2 score set: {:.3f}'
              .format(r2_score(y_true, y_pred)))
        print('Mean absolute error: {:.3f}'
              .format(mean_absolute_error(y_true, y_pred)))
        print('Mean squared error: {:.3f}'
              .format(mean_squared_error(y_true, y_pred)))
        print('RMSE: {:.3f}'.format(sqrt(mean_squared_error(y_true, y_pred))))
    
    return r2_score(y_true, y_pred),mean_absolute_error(y_true, y_pred),sqrt(mean_squared_error(y_true, y_pred))


def plot_regression(y_true, y_pred, x_label="x_label", y_label="y_label", label_size=14, font_size=20, line_start=-0.5, line_end=1.5):
       
    # Plot Results
    x = np.linspace(line_start, line_end, 10)
    fig, ax = plt.subplots(1, 1,figsize=(10, 8))
    ax.grid()
    ax.plot(y_true, y_pred, color='#2874A6', linestyle= 'None',  marker='o',markersize=3 )
    ax.tick_params(axis='both', which='major', labelsize=label_size)
    plt.plot(x, x, linestyle='solid',color='black')
    plt.xlabel(x_label,fontsize=font_size)
    plt.ylabel(y_label,fontsize=font_size)
    plt.show()        
    
    return ax    

  #Color list for the regression  
  #943126  
  #CB4335
  #EC7063    
  #1D8348
  #28B463  
  #2874A6
    
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Greys, label_size=14, font_size=16, number_size=20):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
     
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cbar=ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=label_size)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes)
    
    ax.set_title(title, fontsize=font_size)
    ax.set_xlabel('ML label', fontsize=font_size)
    ax.set_ylabel('DFT label', fontsize=font_size)

    ax.tick_params(axis='both', which='major', labelsize=label_size)


    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0,
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center", fontsize=number_size,
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
