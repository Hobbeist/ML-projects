# Define a confusion matrix plot function

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
from sklearn.metrics import (accuracy_score, 
                             f1_score, 
                             recall_score, 
                             precision_score, 
                             roc_auc_score, 
                             roc_curve, 
                             precision_recall_curve, 
                             plot_roc_curve, 
                             auc, 
                             cohen_kappa_score)


def plotConfMatrix(y_pred,
                   y_test,
                  target_names,
                  title = 'Confusion matrix',
                  normalize = True):
    
    '''Function to plot a confusion matrix based on true and predicted outcome
    
    Parameters
    ---------------------------------------
    y_pred: Predicted outcome from ML model
    
    y_test: Classes from the test data set
    
    target_names: Classes in case the outcome should be represented as words
    
    title: Title of the plot
    
    normalize: Represent the classes as percentage; default is True'''
    
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    
    
    
    
    
def plotCurves(y_test_mlr, 
               y_test_mnb, 
               y_pred_proba_mlr, 
               y_pred_proba_mnb): 
    
    '''Function to plot a ROC and PRC curve for Naive Bayes and Logistic Regression model results
    
    Parameters
    ---------------------------------------
    y_test_mlr: Test classes from Logistic Regression model
    
    y_test_mnb: Test classes from Naive Bayes model
    
    y_pred_proba_mlr: Predicted class probabilities from the Logistic Regression model
    
    y_pred_proba_mnb: Predicted class probabilities from the Naive Bayes model'''
    
    
    
    fig, ax = plt.subplots(1,2,figsize=(32,10))

    mlr_fpr, mlr_tpr, thresholds = roc_curve(y_test_mlr,y_pred_proba_mlr[:,1])
    mnb_fpr, mnb_tpr, thresholds = roc_curve(y_test_mnb,y_pred_proba_mnb[:,1])

    ax[0].plot(mlr_fpr,mlr_tpr,label='MLR')
    ax[0].plot(mnb_fpr,mnb_tpr,label='MNB')

    ax[0].set_title('ROC Plot for imbalanced Naive Bayes with SMOTE')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate');
    ax[0].legend();


    mlr_precision, mlr_recall, _ = precision_recall_curve(y_test_mlr,y_pred_proba_mlr[:,1])
    mnb_precision, mnb_recall, _ = precision_recall_curve(y_test_mnb,y_pred_proba_mnb[:,1])
    
    # plot the precision-recall curves
    ax[1].plot(mlr_recall, mlr_precision, label='Multinomial Logistic Regression')
    ax[1].plot(mnb_recall, mnb_precision, label='Multinomial Naive Bayes')

    ax[1].set_title('Precision Recall Curve for Multinomial NaiveBayes with SMOTE')
    # axis labels
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    # show the legend
    ax[1].legend()