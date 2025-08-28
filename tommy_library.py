import os, sys
import numpy as np
import pandas as pd
import scipy.stats as st
import re
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics


xs = np.linspace(0, 1, 30)  # this needs to be global


def parse_str(string, ix):
    n_pt = '([\.0-9\+e\-]+)'
    pattern = '\['+n_pt+'\s+'+n_pt+'\s+'+n_pt+'\s+'+n_pt+'[.\\n\s]+'+n_pt+'[.\\n\s]+'+n_pt
    m = re.match(pattern, string)
    if m is None:
        print('error', string)
        
    CBT_pb, CBTA_pb, CBTP_pb, CBT3_pb, CBTPA_pb, CBTP3_pb = m.groups()
    CBT_pb = float(CBT_pb)
    CBTA_pb = float(CBTA_pb) 
    CBTP_pb = float(CBTP_pb)
    CBT3_pb = float(CBT3_pb)
    CBTPA_pb = float(CBTPA_pb) 
    CBTP3_pb = float(CBTP3_pb)
    all_pbs = [CBT_pb, CBTA_pb, CBTP_pb, CBT3_pb, CBTPA_pb, CBTP3_pb]
    return all_pbs[ix]


def get_roc_mean_err(dataframe):
    """
        Get mean ROC and confidence intervals by bootstrapping.
        `dataframe` needs to have two colmns: ground truth a
        nd predictions.
    """
    global xs
    
    dataframe = dataframe.copy()
    dataframe.columns = ['true', 'pred']
    
    roc_ys = []
    for _ in range(5000):
        df = dataframe.sample(frac=1., replace=True)
        fpr, tpr, thr = roc_curve(df.true, df.pred)
        a1_coefs = np.polyfit(fpr, tpr, 3)
        new_ys = np.polyval(a1_coefs, xs)
        roc_ys.append(new_ys)
    roc_ys = np.asarray(roc_ys)
    roc_y_mean = np.mean(roc_ys, axis=0)
    roc_y_qnt = np.quantile(roc_ys, q=.95, axis=0)
    roc_y_err = np.abs(roc_y_mean - roc_y_qnt)
    return roc_y_mean, roc_y_err


def get_roc_xy(dataframe):
    """
    """
    global xs
    
    df = dataframe.copy()
    df.columns = ['true', 'pred']
    fpr, tpr, thr = roc_curve(df.true, df.pred)
    return fpr, tpr


def get_roc_mean_err_rand(dataframe):
    """
    """
    assert len(dataframe.columns) == 1  # only ground truth
    df = dataframe.copy()
    df.columns = ['true']
    roc_ys = []
    for _ in range(500):
        df['pred'] = np.random.rand(len(df))  # generate rnd
        fpr, tpr, thr = roc_curve(df.true, df.pred)
        a1_coefs = np.polyfit(fpr, tpr, 3)
        new_ys = np.polyval(a1_coefs, xs)
        roc_ys.append(new_ys)
    roc_ys = np.asarray(roc_ys)
    roc_y_mean = np.mean(roc_ys, axis=0)
    roc_y_qnt = np.quantile(roc_ys, q=.95, axis=0)
    roc_y_err = np.abs(roc_y_mean - roc_y_qnt)
    return roc_y_mean, roc_y_err


def auc(X, Y):
    return 1/(len(X)*len(Y)) * sum([kernel(x, y) for x in X for y in Y])


def kernel(X, Y):
    return .5 if Y == X else int(Y < X)


def structural_components(X, Y):
    V10 = [1/len(Y) * sum([kernel(x, y) for y in Y]) for x in X]
    V01 = [1/len(X) * sum([kernel(x, y) for x in X]) for y in Y]
    return V10, V01


def get_S_entry(V_A, V_B, auc_A, auc_B):
    return 1/(len(V_A)-1) * sum([(a-auc_A)*(b-auc_B) for a,b in zip(V_A, V_B)])


def z_score(var_A, var_B, covar_AB, auc_A, auc_B):
    return (auc_A - auc_B)/((var_A + var_B - 2*covar_AB)**(.5))


def group_preds_by_label(preds, actual):
    X = [p for (p, a) in zip(preds, actual) if a]
    Y = [p for (p, a) in zip(preds, actual) if not a]
    return X, Y


def get_delong_pvalue_zscore(preds_A, preds_B, actual):
    """
        pred_A: prediction from a random classifier
        pred_B: prediction from our model
        actual: ground truth
    
    """
    X_A, Y_A = group_preds_by_label(preds_A, actual)
    X_B, Y_B = group_preds_by_label(preds_B, actual)
    V_A10, V_A01 = structural_components(X_A, Y_A)
    V_B10, V_B01 = structural_components(X_B, Y_B)
    auc_A = auc(X_A, Y_A)
    auc_B = auc(X_B, Y_B)

    # Compute entries of covariance matrix S (covar_AB = covar_BA)
    var_A = (get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1/len(V_A10)
             + get_S_entry(V_A01, V_A01, auc_A, auc_A) * 1/len(V_A01))
    var_B = (get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1/len(V_B10)
             + get_S_entry(V_B01, V_B01, auc_B, auc_B) * 1/len(V_B01))
    covar_AB = (get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1/len(V_A10)
                + get_S_entry(V_A01, V_B01, auc_A, auc_B) * 1/len(V_A01))
    # Two tailed test
    z = z_score(var_A, var_B, covar_AB, auc_A, auc_B)
    p = st.norm.sf(abs(z))*2
    return p, z