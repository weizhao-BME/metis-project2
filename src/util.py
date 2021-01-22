#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for data analysis

@author: Wei Zhao @ Metis, 01/17/2021
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt

def convert_cat_to_num(X, feat_name):
    """
    This function converts categorical variables
    to numerical variables, e.g. fuel type, drive type,
    transmission, and engine type.
    """
    X_cat = X[[feat_name]]
    ohe = OneHotEncoder(sparse=False, drop='first')
        
    ohe.fit(X_cat)
    X_cat_tform = ohe.transform(X_cat)
    columns = ohe.get_feature_names(["X" + feat_name])
    X_cat_tform_df = pd.DataFrame(X_cat_tform, columns=columns, index=X_cat.index)
    
    return X_cat_tform_df

def add_interaction_features(feat_a, feat_b, X_train_combined, X_test_combined):
    """
    This function adds interactin features to a pandas data frame

    Parameters
    ----------
    feat_a : str
        feature a.
    feat_b : str
        feature a.
    X_train_combined : pandas Dataframe
        A training dataset that combines interaction features.
    X_test_combined : pandas Dataframe
        A testing dataset that combines interaction features.

    Returns
    -------
    X_train_combined : pandas Dataframe
        A training dataset that combines interaction features.
    X_test_combined : pandas Dataframe
        A testing dataset that combines interaction features.

    """
    new_feat = feat_a + '*' + feat_b
    if new_feat not in X_train_combined.columns:
        X_train_combined.insert(X_train_combined.shape[1], new_feat,
                                X_train_combined[feat_a] * X_train_combined[feat_b])
    
        X_test_combined.insert(X_test_combined.shape[1], new_feat,
                               X_test_combined[feat_a] * X_test_combined[feat_b])
    return X_train_combined, X_test_combined

def k_fold_split(num_of_fold, X_train_cont, X_train_cat, y_train):
    """
    This function split continuous and categorical traning data into k folds.
    Parameters
    ----------
    num_of_fold : int
        DESCRIPTION.
    X_train_cont : pandas Dataframe
        Continuous variables.
    X_train_cat : pandas Dataframe
        Categorical variables.
    y_train : pandas Dataframe
        Lables.

    Returns
    -------
    dict_kf : dictionary
        dict_kf includes each pair of traning and validation datasets.

    """

    kf = KFold(n_splits=num_of_fold, shuffle=True, random_state = 71)
    
    dict_kf = defaultdict(list)
    
    for train_ind, val_ind in kf.split(X_train_cont, y_train):
        t_x_train_count, t_x_train_cat, t_y_train = (X_train_cont.iloc[train_ind, :],
                                                     X_train_cat.iloc[train_ind, :],
                                                     y_train.iloc[train_ind]
                                                    )
        
        dict_kf["x_train_cont_kf"].append(t_x_train_count)
        dict_kf["x_train_cat_kf"].append(t_x_train_cat)
        dict_kf["y_train_kf"].append(t_y_train)
        
        t_x_val_count, t_x_val_cat, t_y_val = (X_train_cont.iloc[val_ind, :],
                                               X_train_cat.iloc[val_ind, :],
                                               y_train.iloc[val_ind]
                                              )
        
        dict_kf["x_val_cont_kf"].append(t_x_val_count)
        dict_kf["x_val_cat_kf"].append(t_x_val_cat)
        dict_kf["y_val_kf"].append(t_y_val)
        
    return dict_kf
    
def feature_tform_kf(dict_kf, num_of_fold):
    """
    This function performs oneHotEncoding to convert categorical values
    to numerical values and standardizes continuous varibles for k-fold 
    cross validation.
    
    Parameters
    ----------
    dict_kf : dictionary
        dict_kf includes each pair of traning and validation datasets.
    num_of_fold : int
        Number of fold.

    Returns
    -------
    X_train_vf : dictionary
        K training datasets- features.
    X_val_vf : dictionary
        K validation datasets - features.
    y_train_vf : dictionary
        K training datasets- labels.
    y_val_vf : dictionary
        K validation datasets- labels.

    """
    dict_kf_tform = defaultdict(list)

    for k in dict_kf.keys():
        if "cat" in k:
            for i in dict_kf[k]:
                col_name = i.columns
                t_df = pd.DataFrame()
                for cn in col_name:
                    t_df = pd.concat([t_df, convert_cat_to_num(i, cn)], axis=1)
                col_name_cat = t_df.columns
                dict_kf_tform[k].append(np.array(t_df))
                
        elif "cont" in k:
            for i in dict_kf[k]:
                col_name_cont = i.columns
                std = StandardScaler()
                std.fit(i)
                t_std = std.transform(i.values)
                dict_kf_tform[k].append(t_std)
    # combine cat and num        
    X_train_vf = {}
    X_val_vf = {}
    y_train_vf = {}
    y_val_vf = {}
    
    col_name = list(col_name_cont) + list(col_name_cat)
    # col_name = ['year', 'mileage', 'engine_size', 'cty_mpg', 'hwy_mpg', 
    #             'X_fuel_gas', 'X_fuel__hybrid',
    #             'X_drive_fwd','X_drive_rwd',
    #             'X_engine_turbo']
    
    for i in range(0, num_of_fold):
        t = np.hstack((dict_kf_tform["x_train_cont_kf"][i],
                                   dict_kf_tform["x_train_cat_kf"][i]))
        X_train_vf[i] = pd.DataFrame(t, columns=col_name)
        
        t = np.hstack([dict_kf_tform["x_val_cont_kf"][i],
                                 dict_kf_tform["x_val_cat_kf"][i]])
        X_val_vf[i] = pd.DataFrame(t, columns=col_name)
        
        y_train_vf[i] = pd.DataFrame(dict_kf["y_train_kf"][i], columns=["price"])
        y_val_vf[i] = pd.DataFrame(dict_kf["y_val_kf"][i], columns=["price"])
    
    return X_train_vf, y_train_vf, X_val_vf, y_val_vf

def feature_tform(dict_all):
    """
    This function performs oneHotEncoding to convert categorical values
    to numerical values and standardizes continuous varibles for the entire dataset.
    
    Parameters
    ----------
    dict_kf : dictionary
        dict_kf includes each pair of traning and validation datasets.
    
    Returns
    -------
    X_train_vf : dictionary
        K training datasets- features.
    X_val_vf : dictionary
        K validation datasets - features.
    y_train_vf : dictionary
        K training datasets- labels.
    y_val_vf : dictionary
        K validation datasets- labels.

    """
    dict_tform = defaultdict(list)

    for k in dict_all.keys():
        if "cat" in k:
            col_name = dict_all[k].columns
            t_df = pd.DataFrame()
            for cn in col_name:
                t_df = pd.concat([t_df, convert_cat_to_num(dict_all[k], cn)], axis=1)
            col_name_cat = t_df.columns
            dict_tform[k] = np.array(t_df)

        elif "cont" in k:
            col_name_cont = dict_all[k]
            std = StandardScaler()
            std.fit(dict_all[k])
            t_std = std.transform(dict_all[k].values)
            dict_tform[k] = t_std
    # combine cat and num        
    col_name = list(col_name_cont) + list(col_name_cat)
    # col_name = ['year', 'mileage', 'engine_size', 'cty_mpg', 'hwy_mpg', 
    #             'X_fuel_gas', 'X_fuel_hybrid',
    #             'X_drive_fwd','X_drive_rwd',
    #             'X_engine_turbo']
        
    t = np.hstack((dict_tform["X_train_cont"],
                               dict_tform["X_train_cat"]))
    X_train_vf = pd.DataFrame(t, columns=col_name)
    
    t = np.hstack((dict_tform["X_test_cont"],
                             dict_tform["X_test_cat"]))
    X_test_vf = pd.DataFrame(t, columns=col_name)
    
    y_train_vf = pd.DataFrame(dict_all["y_train"], columns=["price"])
    y_test_vf = pd.DataFrame(dict_all["y_test"], columns=["price"])
    
    return X_train_vf, y_train_vf, X_test_vf, y_test_vf

def lin_regres(X_train, y_train, X_val, y_val, alpha_list, method="ridge"):
    """
    This function performs linear regressin with
    ridge regularization. 

    Parameters
    ----------
    X_train : list
        Training data.
    y_train : list
        Training labels.
    X_val : list
        Validation or testing data.
    y_val : list
        Validation or testing lables.
    alpha_list : list
        A list of alpha values for ridge regularization.

    Returns
    -------
    Linear model and predicted labels 
    if alpha_list only has 1 value.
    else
    R^2, mean absolute error, and mean squared errors

    """
    cv_rsq, cv_mae, cv_mse = [], [], []
    for a in alpha_list:
        t_cv_rsq, t_cv_mae, t_cv_mse = [], [], []
        for i in range(0, len(X_train)):
            t_x_train = X_train[i]
            t_y_train = y_train[i]
            t_x_val = X_val[i]
            t_y_val = y_val[i]
            if method == "ridge":
                lm = Ridge(alpha = a)
            elif method == "lasso":
                lm = Lasso(alpha = a)
            lm.fit(t_x_train,t_y_train)
            t_y_pred = lm.predict(t_x_val)
            
            t_cv_rsq.append(r2_score(t_y_val, t_y_pred))
            t_cv_mae.append(mean_absolute_error(t_y_val, t_y_pred))
            t_cv_mse.append(mean_squared_error(t_y_val, t_y_pred))
        
        cv_rsq.append(np.mean(t_cv_rsq))
        cv_mae.append(np.mean(t_cv_mae))
        cv_mse.append(np.mean(t_cv_mse))
        
    if len(alpha_list) == 1:
        return lm, t_y_pred, cv_rsq, cv_mae, cv_mse
    else:
        return cv_rsq, cv_mae, cv_mse

def adjusted_r2(ord_r2, sample_size, n_of_feat):
    """
    This function calcualted the adjusted R-squared value from the ordinary
    R-squared value reported from sklearn
    """
    adj_r2 = 1-(1-ord_r2)*(sample_size-1)/(sample_size-n_of_feat-1)
    return adj_r2

def calculate_p_val(lm, X_train_all, y_test_all, y_pred_all):
    """
    This function calculates p-value for each conefficient
    determined from regression model, "lm"
    X_train_all and y_test_all: pandas Dataframe
    y_pred_all: numpy array as generated
    ref: "https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression"
    """
    params = np.array(lm.coef_)

    new_x = np.array(X_train_all)
    mse = mean_squared_error(np.array(y_test_all), y_pred_all)
    
    var_b = mse*(np.linalg.inv(np.dot(new_x.T,new_x)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b
    
    p_values =[2*(1-stats.t.cdf(np.abs(i),
                                (len(new_x)))) for i in ts_b]
    sd_b = np.round(sd_b,3)
    ts_b = np.round(ts_b,3)
    p_values = np.round(p_values,4)
    # params = np.round(params,5)
    
    params = params.reshape(sd_b.shape)
    ts_b = ts_b.reshape(sd_b.shape)
    p_values = p_values.reshape(sd_b.shape)
    
    stats_df = pd.DataFrame()
    stats_df["Coefficients"], stats_df["Standard Errors"], stats_df["t values"], stats_df["p-values"] = [params, sd_b, ts_b, p_values]
    return stats_df 

def calculate_f_stats(r2, dfn, dfd):
    """
    ref:"http://facweb.cs.depaul.edu/sjost/csc423/documents/f-test-reg.htm"
    """
    F = (r2/dfn)/((1-r2)/dfd)
    p_val = 1 - stats.f.cdf(F, dfn, dfd)
    return p_val, F

def diagnose_plot(y_pred, y_test):
    residuals = y_pred - y_test
    plt.figure(figsize=(16, 4))
    plt.subplots_adjust(wspace = 0.6)
    
    plt.subplot(1,3,1)
    ax = plt.scatter(y_test/1000,y_pred/1000, 2);
    ax.set_facecolor([0.2, 0.4, 0.6])
    plt.xticks((range(0, 51, 10)))
    
    plt.plot([12, 12],[-100, 100],
              color='k', 
              linewidth=1.5,
              linestyle='--');
    plt.plot([30, 30],[-100, 100],
              color='k',
              linewidth=1.5,
              linestyle='--');
    plt.plot([0, 60],[0, 60],
              color='r',
              linewidth=1.5,
              linestyle='-');
    
    plt.xlabel("Actual price (x1000)")
    plt.ylabel("Predicted price (x1000)")
    plt.xlim([0, 50])
    plt.ylim([0, 50])
    plt.text(6, 40, '$12k', horizontalalignment='center',
             verticalalignment='center')
    plt.text(24, 40, '$30k', horizontalalignment='center',
             verticalalignment='center')
    plt.text(39, 45, 'y=x', horizontalalignment='center',
             verticalalignment='center', color='r')
    plt.title("Predicted vs Actual")
    
    
    plt.subplot(1,3,2)
    ax = plt.scatter(y_pred/1000, residuals/1000, 2)
    ax.set_facecolor([0.2, 0.4, 0.6])
    plt.title("Residual plot")
    plt.xlabel("Predicted price (x1000)")
    plt.ylabel("Residuals (x1000)")
    plt.plot([0, 50],[0, 0],
              color='k',
              linewidth=1.5,
              linestyle='--');
    
    plt.xlim([0, 50])
    plt.ylim([-15, 10])
    
    ax = plt.subplot(1,3,3)
    stats.probplot(residuals/1000, dist="norm", plot=plt)
    ax.get_lines()[0].set_markersize(4.0)
    ax.get_lines()[0].set_markerfacecolor([0.2, 0.4, 0.6])
    ax.get_lines()[0].set_markeredgecolor([0.2, 0.4, 0.6])
    
    plt.title("Normal Q-Q plot")
    plt.ylabel("Ordered values (x1000)");
    

