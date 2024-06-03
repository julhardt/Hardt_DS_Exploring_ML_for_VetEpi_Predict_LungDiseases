###########################################################
### Final Project for a Certification as Data Scientist ###
### Thesis: Prediction of lung diseases in pigs         ###
###########################################################
### Juliane Hardt, Essen, 29.05.2024                    ###
###########################################################
###########################################################
### Complete Program for Reprocibility of Analyses      ###
###########################################################
#
###########################################################
### A. Path definitions                                 ###
###########################################################

# location of data
EXT_PATH = (r"C:/Users/jhardt/Documents/PYTHON/Dig_DS_Projektarbeit/")  #change path here to write out figures
ROOT_PATH = "../"
ROOT_DATA_PATH = ROOT_PATH + "data/"
# DATA_PATH_RESULTS = ROOT_PATH + "results/"
DATA_PATH_FIGURES = ROOT_PATH + "figures/"
FIGURES_PATH_ext  = EXT_PATH  + "3_plots/"                              # adjust path here to write out figures
# DATA_PATH_MODELS  = ROOT_PATH + "models/"
# DATA_PATH_SCRIPTS = ROOT_PATH + "programs/"

print(ROOT_PATH)
print(ROOT_DATA_PATH)
print(DATA_PATH_FIGURES)
# print(DATA_PATH_MODELS) 
print(EXT_PATH)
print(FIGURES_PATH_ext)

###########################################################
### B. Import general PYTHON modules needed             ###
###########################################################

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#################################################################
### 1. Read preprocessed data set and define variable objects ###
#################################################################

simdata200_id = pd.read_excel(os.path.join("data", "Simdata200formatted.xlsx"))  # wenn PADS_1_EDA unter programs verschoben, dann in (()) zuerst "..", ergänzen
print(simdata200_id)

if isinstance(simdata200_id, pd.DataFrame):
   print("yes")

print(type(simdata200_id))

# Define predictor variables (= features) as standalone objects (arrays)
numpigs_farm    = np.array(simdata200_id['numpigs_farm'])
space_ppig      = np.array(simdata200_id['space_ppig'])
ADG             = np.array(simdata200_id['ADG'])
coughing        = np.array(simdata200_id['coughing'])              
antibiotics_TF  = np.array(simdata200_id['antibiotics_TF'])        

# Outcome variable (= target) as array
prevlungdis     = np.array(simdata200_id['prevlungdis']) 

###############################################
## 2. Exploratory Data Analyses             ###
##    Descriptive Analyses and Correlations ### 
###############################################

###############################
## 2.1. Descriptive Analyses ##
############################### 

# 2.1.a) Table: Descriptive Analyses
simdata200_id.describe()      

# 2.1. b) Plot histograms of DataFrame features
pd.DataFrame.hist(simdata200_id[['numpigs_farm']])
plt.ylabel('count')
plt.savefig('supplement/figures/hist_numpigs.png', transparent = True, bbox_inches='tight')
plt.savefig(FIGURES_PATH_ext + 'hist_numpigs.png', transparent = True, bbox_inches='tight')
# plt.show()

pd.DataFrame.hist(simdata200_id[['space_ppig']])
plt.ylabel('count')
plt.savefig('supplement/figures/hist_space_ppig.png', transparent = True, bbox_inches='tight')
plt.savefig(FIGURES_PATH_ext + 'hist_space_ppig.png', transparent = True, bbox_inches='tight')
# plt.show()

pd.DataFrame.hist(simdata200_id[['ADG']])
plt.ylabel('count')
plt.savefig('supplement/figures/hist_ADG.png', transparent = True, bbox_inches='tight')
plt.savefig(FIGURES_PATH_ext + 'hist_ADG.png', transparent = True, bbox_inches='tight')
# plt.show()

pd.DataFrame.hist(simdata200_id[['coughing']])
plt.ylabel('count')
plt.savefig('supplement/figures/hist_coughing.png', transparent = True, bbox_inches='tight')
plt.savefig(FIGURES_PATH_ext + 'hist_coughing.png', transparent = True, bbox_inches='tight')
# plt.show()

pd.DataFrame.hist(simdata200_id[['antibiotics_TF']])
plt.ylabel('count')
plt.savefig('supplement/figures/hist_antibiotics.png', transparent = True, bbox_inches='tight')
plt.savefig(FIGURES_PATH_ext + 'hist_antibiotics.png', transparent = True, bbox_inches='tight')
# plt.show()

pd.DataFrame.hist(simdata200_id[['prevlungdis']])
plt.ylabel('count')
plt.savefig('supplement/figures/hist_prevlungdis.png', transparent = True, bbox_inches='tight')
plt.savefig(FIGURES_PATH_ext + 'hist_prevlungdis.png', transparent = True, bbox_inches='tight')
# plt.show()

# Prepare dataset for correlations and train-test-split of data

print(simdata200_id.columns)

# Pop the columns of technical variables of simulation:
to_pop = ["farm_id"]

for col in to_pop:
    simdata200_id.pop(col)

simdata200 = simdata200_id.copy()
print(simdata200.columns)

#######################
## 2.2. Correlations ##
####################### 

## 2.2.a) Tabular correlation matrix ##

simdata200.corr()
print(simdata200.corr())

#                numpigs_farm  space_ppig       ADG  coughing  antibiotics_TF  \
# numpigs_farm        1.000000   -0.110643  0.034834 -0.038527        0.188770   
# space_ppig         -0.110643    1.000000 -0.034934 -0.073912       -0.015002   
# ADG                 0.034834   -0.034934  1.000000 -0.290193       -0.102363   
# coughing           -0.038527   -0.073912 -0.290193  1.000000        0.167913   
# antibiotics_TF      0.188770   -0.015002 -0.102363  0.167913        1.000000   
# prevlungdis         0.002303    0.055925 -0.206561  0.133292        0.216841   

#                prevlungdis  
# numpigs_farm       0.002303  
# space_ppig         0.055925  
# ADG               -0.206561  
# coughing           0.133292  
# antibiotics_TF     0.216841  
# prevlungdis        1.000000  

## 2.2.b) Correlation heatmap matrix ##

import seaborn as sns
sns.heatmap(simdata200.corr())
plt.title("Correlation matrix heatmap")
plt.savefig('figures/corr_heatmap.png', bbox_inches='tight')
plt.savefig(FIGURES_PATH_ext + 'corr_heatmap.png', transparent = True, bbox_inches='tight')
# plt.show()

## 2.2.c) Scatter plots for correlations ##

#%%
simdata200_id = pd.read_excel(os.path.join("data", "Simdata200formatted.xlsx"))

plt.scatter(simdata200_id['numpigs_farm'],simdata200_id['prevlungdis'])
plt.title("Scatterplot for Number of pigs per farm")
plt.xlabel('number of pigs per farm')
plt.ylabel('prevalence of lung disease')
plt.savefig('supplement/figures/scatter_numpigs_prevlungdis.png', bbox_inches='tight')
plt.savefig(FIGURES_PATH_ext + 'scatter_numpigs_prevlungdis.png', transparent = True, bbox_inches='tight')
# plt.show()

#%%

plt.scatter(simdata200_id['space_ppig'],simdata200_id['prevlungdis'])
plt.title("Scatterplot for Space per pig [m²]")
plt.xlabel('space per pig [m²]')
plt.ylabel('prevalence of lung disease')
plt.savefig('supplement/figures/scatter_spaceppig_prevlungdis.png', bbox_inches='tight')
plt.savefig(FIGURES_PATH_ext + 'scatter_spaceppig_prevlungdis.png', transparent = True, bbox_inches='tight')
# plt.show()

#%%

plt.scatter(simdata200_id['ADG'],simdata200_id['prevlungdis'])
plt.title("Scatterplot for ADG per pig in box")
plt.xlabel('Average daily weight gain')
plt.ylabel('prevalence of lung disease')
plt.savefig('supplement/figures/scatter_ADG_prevlungdis.png', bbox_inches='tight')
plt.savefig(FIGURES_PATH_ext + 'scatter_ADG_prevlungdis.png', transparent = True, bbox_inches='tight')
# plt.show()

#%%

plt.scatter(simdata200_id['coughing'],simdata200_id['prevlungdis'])
plt.title("Scatterplot for coughing index")
plt.xlabel('coughing index per pig in box')
plt.ylabel('prevalence of lung disease')
plt.savefig('supplement/figures/scatter_coughing_prevlungdis.png', bbox_inches='tight')
plt.savefig(FIGURES_PATH_ext + 'scatter_coughing_prevlungdis.png', transparent = True, bbox_inches='tight')
# plt.show()

#%%

plt.scatter(simdata200_id['antibiotics_TF'],simdata200_id['prevlungdis'])
plt.title("Scatterplot for Treatment freq with antibiotics")
plt.xlabel('treatment frequency with antibiotics')
plt.ylabel('prevalence of lung disease')
plt.savefig('supplement/figures/scatter_antibiotics_prevlungdis.png', bbox_inches='tight')
plt.savefig(FIGURES_PATH_ext + 'scatter_antibiotics_prevlungdis.png', transparent = True, bbox_inches='tight')
# plt.show()

#%%

###############################################
### 3. Split Data in Training and Test data ###
###############################################

#%%

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

X = simdata200.drop(["prevlungdis"],axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, simdata200["prevlungdis"])

minmax = MinMaxScaler()
#                                                                    
x_train = pd.DataFrame(minmax.fit_transform(x_train), columns=X.columns)    
#                                        # here: code - however unusual - only works identically to course notebook DS04
x_test = pd.DataFrame(minmax.transform(x_test))    # here dataframes used due to error messages below

print('X Train: {}'.format(x_train.shape)) 
print('Y Train: {}'.format(y_train.shape)) 
print('X Test: {}'.format(x_test.shape)) 
print('Y Test: {}'.format(y_test.shape))

# Check datasets 
# x_train.describe()
# x_test.describe()
# y_train.describe()
# y_test.describe()

#%%

#####################################################
### 4 Methods comparison: define and train models ###
#####################################################

#%%
# import warnings
# warnings.filterwarnings('ignore')

# import time              # if running time of models of interest
from sklearn.linear_model import LinearRegression
# from sklearn import svm
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
# from libsvm.svmutil import *

def mse(y_pred,y):
    return np.mean((y_pred-y)**2)
print(mse)

# set seed for model runs:
np.random.seed(1234)
# for method in methods:          # variant B for loop command 
def train_evaluate_time(method):
    # start_time = time.time()                          # optional if running time of interest
    methods[method].fit(x_train, y_train)
    # running_time = round(time.time() - start_time,2)  # optional if running time of interest
    model_predict = methods[method].predict(x_test)
    model_mse = mse(model_predict,y_test)
    print("{}: {}".format(method, model_mse))                                                       # variant A to provide data for figures
    # print("Mean squared error of {}: {}; running time: {}".format(method, model_mse, running_time)) # variant B to compare running times
methods = {"linregr": LinearRegression(),           
           "svm_lin": LinearSVR(), 
           "svm_nu":  NuSVR(),
           "svm_eps": SVR(),  
           "grad_boost": GradientBoostingRegressor(),   # here several hyperparameters can be tested
           "ada_boost": AdaBoostRegressor()}            # here several hyperparameters can be tested
for method in methods:           # variant A for loop command for cases of error messages 
    train_evaluate_time(method)

#%%

### visualization of MSE (values after setting seed inserted): 
#                                                          # can be extended to other errors functions
methods_mse = {"linregr": round(0.9264739917022772,3),           
               "svm_lin": round(0.9245213942880319,3), 
               "svm_nu":  round(0.8433280063343814,3),  
               "svm_eps": round(0.8786628129414654,3),
               "grad_boost": round(0.9069524855631259,3),
               "ada_boost": round(0.8737169445193704,3)}                 
print(methods_mse)

### bar plot for comparison of Mean squared Error (MSE) for 6 methods ###

plt.bar(range(len(methods_mse)), list(methods_mse.values()), align='center')
plt.xticks(range(len(methods_mse)), list(methods_mse.keys()))
plt.title("Comparison of methods: Mean Squared Error")
plt.xlabel("Regression modelling methods")
plt.ylabel("Mean Squared Error")
plt.savefig('figures/models_comp_mse.png', transparent = True, bbox_inches='tight')
plt.savefig(FIGURES_PATH_ext + 'models_comp_mse.png', transparent = True, bbox_inches='tight')
# plt.show()

#%%

#############################################################
### 5. Ensemble Learning for chosen methods in comparison ###
#############################################################

from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error 

#%%

# Model fit of ensemble learner for methods comparison of 6 regression methods #

np.random.seed(1234)
ensemble = VotingRegressor([("linregr",LinearRegression()),
                            ("svm_lin",LinearSVR()),("svm_nu",NuSVR()),("svm_eps",SVR()),
                            ("grad_boost",GradientBoostingRegressor()),
                            ("ada_boost",AdaBoostRegressor())])
ensemble.fit(x_train,y_train)

# compute error functions for Ensemble Super-Learner #
np.random.seed(1234)
mse_ensemble      = mean_squared_error(ensemble.predict(x_test),y_test)
rsme_ensemble     = root_mean_squared_error(ensemble.predict(x_test),y_test)

print("MSE for ensemble learner", mse_ensemble)
print("RMSE for ensemble learner", rsme_ensemble)

#%%

#################################################################
### 6. Visualization of model results for interpretation ###
#################################################################

### 6.1. Beeswarm plots of shap values for SVM regression methods ###

import shap

# 6.1. a) Shap value beeswarm plots of Linear Support Vector regression #

explainer_svmlin = shap.KernelExplainer(methods["svm_lin"].predict, x_train)
shapvalues_svmlin = explainer_svmlin.shap_values(x_train)
plt.figure(figsize=(10,20))
plt.title("SHAP Values: {}".format(methods["svm_lin"]))
shap.summary_plot(shapvalues_svmlin, x_train)
plt.savefig('figures/shap_svmlin.png', transparent = True, bbox_inches='tight')
plt.savefig(FIGURES_PATH_ext + 'shap_svmlin.png', transparent = True, bbox_inches='tight')
# plt.show()
#%%

# 6.1. b) Shap value beeswarm plots of Nu-SV regression  #
#%%
explainer_svmnu = shap.KernelExplainer(methods["svm_nu"].predict, x_train)
shapvalues_svmnu = explainer_svmnu.shap_values(x_train)
plt.figure(figsize=(10,20))
plt.title("SHAP Values: {}".format(methods["svm_nu"]))
shap.summary_plot(shapvalues_svmnu, x_train)
plt.savefig('figures/shap_svmnu.png', transparent = True, bbox_inches='tight')
plt.savefig(FIGURES_PATH_ext + 'shap_svmnu.png', transparent = True, bbox_inches='tight')
# plt.show()
#%%

# 6.1. c) Shap value beeswarm plots of Epsilon-SV Regression  #
#%%
explainer_svmeps = shap.KernelExplainer(methods["svm_eps"].predict, x_train)
shapvalues_svmeps = explainer_svmeps.shap_values(x_train)
plt.figure(figsize=(10,20))
plt.title("SHAP Values: Epsilon-SVM Regression {}".format(methods["svm_eps"]))
shap.summary_plot(shapvalues_svmeps, x_train)
plt.savefig('figures/shap_svm-eps.png', transparent = True, bbox_inches='tight')
plt.savefig(FIGURES_PATH_ext + 'shap_svm-eps.png', transparent = True, bbox_inches='tight')
# plt.show()
#%%

### 6.2. Mosaic plot of feature importance for Linear and Boosting Regression methods ###

#%%
def plot_one_method(ax, title, x):
    ax.set_title(title)
    ticks = np.arange(1, len(x) + 1)
    ax.bar(ticks, x)
    ax.set_xticks(ticks=ticks, labels=list(x_train.columns), rotation=90)
    np.random.seed(2345)

fig, ax = plt.subplots(1,3, figsize=(15,5))    
plot_one_method(ax[0], "Coefficients of Linear Regression", methods["linregr"].coef_)
plot_one_method(ax[1], "Feature Importance Gradient Boosting Regressor", methods["grad_boost"].feature_importances_)
plot_one_method(ax[2], "Feature Importance AdaBoost Regressor", methods["ada_boost"].feature_importances_)
plt.savefig('figures/mosaic_plot_featureimportance.png', transparent = True, bbox_inches='tight') 
plt.savefig(FIGURES_PATH_ext + 'mosaic_plot_featureimportance.png', transparent = True, bbox_inches='tight')
plt.show() 

#%%


