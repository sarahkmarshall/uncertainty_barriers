# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 12:39:47 2019

@author: mars0451

This script is for post-processing the data from the models.

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import flopy
import pandas as pd
import os
import sys
import string
import pickle


print(sys.version)
print('numpy version: {}'.format(np.__version__))
print('matplotlib version: {}'.format(mpl.__version__))
print('pandas version: {}'.format(pd.__version__))
print('flopy version: {}'.format(flopy.__version__))

savedata_question = 'NO' # 'YES' or 'NO'

#==== DIRECTORIES =============================================================

MODFLOW_folder      = r'C:\workspace\modflow_dbl\mfnwtdbl.exe' # double precision MODFLOW NWT
proj_3_folder       = r'C:\workspace\Proj3_PilbaraBarriers'
MT3D_USGS_folder      = r'C:\workspace\modflow_dbl\mt3dusgsdbl.exe' # Double precision MT3D-USGS
exe_name = r'C:\workspace\Proj2_TracersBarriers\MT3D_Reaction_sngl' 
mt3d_version = 'mt3d-usgs' 

overallName = "UB_47"

n_runs = 18

modelNames_list = []

for run in range(n_runs):
    modelNames_list.append(overallName + "_" + string.ascii_lowercase[run])
 
ppc_folder = os.path.join(proj_3_folder, (overallName + "_PPC")) # Folder to put all of these "post processing combined" figures.
if not os.path.exists(ppc_folder):
    os.makedirs(ppc_folder)
    
figureDirectory = os.path.join(ppc_folder, 'Figures')
if not os.path.exists(figureDirectory):
    os.makedirs(figureDirectory)
       
## Lists for comparing all models -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

all_phantom_dfs  = []
metric_list      = []
metric_mean_list = []
number_of_key_f  = []
rmse_h_all_f_all = []
rmse_a_all_f_all = []
rmse_c_all_f_all = []

# RMSE with the key faults only

rmse_h_key_f_all = []
rmse_a_key_f_all = []
rmse_c_key_f_all = []

#=====PROPERTIES OF MODEL OUTPUTS==============================================
length_real_barrier = 5515 #m
   
#=====ANALYSIS OF MODEL OUTPUTS================================================
mdl = 1
for mdl in range(len(modelNames_list)):
    modelname = modelNames_list[mdl]
    print("Model name: %s" % modelname)
    modelname_mt = modelname + "_MT3D" 
    print("MT3D name: %s" % modelname_mt)

    this_model_folder = os.path.join(proj_3_folder, modelname)
    dataDirectory = os.path.join(this_model_folder, 'Data')
    final_pest_files_folder = os.path.join(this_model_folder, "final_from_pest")

    overarching_model_folder = os.path.join(r'C:\workspace\Proj3_PilbaraBarriers', modelname[:-2])
    pst_name_reg = str(modelname) + '_reg.pst'
    
    # PARAMETERS
    os.chdir(overarching_model_folder)
    
    with open("param_dict.p", "rb") as fp:
        param_dict = pickle.load(fp)
        
    os.chdir(this_model_folder)
     
    # -------------------------------------------------------------------------
    # Download summary table of the post processing analysis
    phantom_df_f = os.path.join(dataDirectory, "key_dataframe.csv") # or phantom_df for all faults, key is just those identified >0.8 and low K
    
    phantom_df = pd.read_csv(phantom_df_f)
    
    phantom_df["length_diff"] = abs(length_real_barrier - phantom_df["length"])
    
    all_phantom_dfs.append(phantom_df)
    
    n_key_f = len(phantom_df.index)
    number_of_key_f.append(n_key_f)
    
    #--------------------------------------------------------------------------
    # Download RMSE data
    
    # Head
    with open (os.path.join(dataDirectory, 'rmse_h_cum_inv'), 'rb') as fp:
        rmse_h_cum_inv = pickle.load(fp)
    with open (os.path.join(dataDirectory, 'rmse_h_cum_K_p'), 'rb') as fp:
        rmse_h_cum_K_p = pickle.load(fp)
    rmse_h_all_f = rmse_h_cum_inv[-1]
    rmse_h_all_f_all.append(rmse_h_all_f) 
    
    rmse_h_key_f = rmse_h_cum_inv[-2] # RMSE with key faults only
    rmse_h_key_f_all.append(rmse_h_key_f) 
    
    # Age
    with open (os.path.join(dataDirectory, 'rmse_a_cum_inv'), 'rb') as fp:
        rmse_a_cum_inv = pickle.load(fp)
    with open (os.path.join(dataDirectory, 'rmse_a_cum_K_p'), 'rb') as fp:
        rmse_a_cum_K_p = pickle.load(fp)
    rmse_a_all_f = rmse_a_cum_inv[-1]
    rmse_a_all_f_all.append(rmse_a_all_f) 
    
    rmse_a_key_f = rmse_a_cum_inv[-2] # RMSE with key faults only
    rmse_a_key_f_all.append(rmse_a_key_f) 

    # Combined
    with open (os.path.join(dataDirectory, 'rmse_c_cum_inv'), 'rb') as fp:
        rmse_c_cum_inv = pickle.load(fp)
    with open (os.path.join(dataDirectory, 'rmse_c_cum_K_p'), 'rb') as fp:
        rmse_c_cum_K_p = pickle.load(fp)
    rmse_c_all_f = rmse_c_cum_inv[-1]
    rmse_c_all_f_all.append(rmse_c_all_f) 
    
    rmse_c_key_f = rmse_c_cum_inv[-2] # RMSE with key faults only
    rmse_c_key_f_all.append(rmse_c_key_f) 
    
#=====RMSE CUT OFF=============================================================
    
    rmse_to_use = rmse_c_cum_inv # rmse_c_cum_inv
    
    drop_between = []
    for i in range(len(rmse_to_use)):
        print(i)
        if i != len(rmse_to_use)-1:
            drop_between.append((rmse_to_use[i] - rmse_to_use[i+1])*100/rmse_to_use[i])


#=====PLOTTING=================================================================

    # Alpha values (orientation) ----------------------------------------------
    plt.figure()
    plt.title(modelname)
    plt.plot(phantom_df["orientation"], phantom_df["K_posterior"], "m*", label="Phantom Faults")

    plt.yscale("log")
    plt.xlabel("Angle of barrier (degr)", fontsize=14)
    plt.ylabel("Estimated K value", fontsize=14) 
    
    axes = plt.gca()
    axes.axhline(y = 1e-3 , xmin=0, xmax=1, ls='--', color='gray')
    axes.axhline(y = 1., xmin=0, xmax=1, ls='--', color='goldenrod')
    plt.ylim([10**-5, 10**5])
    plt.legend()
    plt.savefig(os.path.join(figureDirectory, (modelname + "_alpha")))   
    
    # Difference in alpha values (orientation) --------------------------------
    plt.figure()
    plt.title(modelname)
    plt.plot(phantom_df["diff_in_orientations"], phantom_df["K_posterior"], "m*", label="Phantom Faults")
    plt.yscale("log")
    plt.xlabel("Difference in angle of barrier", fontsize=14)
    plt.ylabel("Estimated K value", fontsize=14)  
    axes = plt.gca()
    axes.axhline(y = 1e-3 , xmin=0, xmax=1, ls='--', color='gray')
    axes.axhline(y = 1., xmin=0, xmax=1, ls='--', color='goldenrod')
    plt.ylim([10**-5, 10**5])
    plt.legend()
    plt.savefig(os.path.join(figureDirectory, (modelname + "_diffalpha")))    
    
    # Distance to real barrier ------------------------------------------------
    plt.figure()
    plt.title(modelname)
    plt.plot(phantom_df["distance"], phantom_df["K_posterior"], "m*", label="Phantom Faults")
    plt.yscale("log")
    plt.xlabel("Distance to real barrier (m)", fontsize=14)
    plt.ylabel("Estimated K value", fontsize=14)  
    axes = plt.gca()
    axes.axhline(y = 1e-3 , xmin=0, xmax=1, ls='--', color='gray')
    axes.axhline(y = 1., xmin=0, xmax=1, ls='--', color='goldenrod')
    plt.ylim([10**-5, 10**5])
    plt.legend()
    plt.savefig(os.path.join(figureDirectory, (modelname + "_distreal")))   
    
    # Length of phantom barrier -----------------------------------------------
    plt.figure()
    plt.title(modelname)
    plt.plot(phantom_df["length"], phantom_df["K_posterior"], "m*", label="Phantom Faults")
    plt.yscale("log")
    plt.xlabel("Length of barrier (m)", fontsize=14)
    plt.ylabel("Estimated K value", fontsize=14)  
    axes = plt.gca()
    axes.axhline(y = 1e-3, xmin=0, xmax=1, ls='--', color='gray')
    axes.axhline(y = 1., xmin=0, xmax=1, ls='--', color='goldenrod')
    plt.ylim([10**-5, 10**5])
    plt.legend()
    plt.savefig(os.path.join(figureDirectory, (modelname + "_len"))) 
    
    # Diff in length of phantom barrier ---------------------------------------
    plt.figure()
    plt.title(modelname)
    plt.plot(phantom_df["length_diff"], phantom_df["K_posterior"], "m*", label="Phantom Faults")
    plt.yscale("log")
    plt.xlabel("Diff in length of barrier (m)", fontsize=14)
    plt.ylabel("Estimated K value", fontsize=14)  
    axes = plt.gca()
    axes.axhline(y = 1e-3, xmin=0, xmax=1, ls='--', color='gray')
    axes.axhline(y = 1., xmin=0, xmax=1, ls='--', color='goldenrod')
    plt.ylim([10**-5, 10**5])
    plt.legend()
    plt.savefig(os.path.join(figureDirectory, (modelname + "_lendiff"))) 
    
#=====OVERALL METRIC TO COMPARE MODEL RESULTS==================================

    w_dist = .001 # Weight for distance
    w_orient = .1 # Weight for orientation/angle/alpha
    w_len = .001 # Weight got length
    w_K = 1 # Weight for K value

    metric_sum = 0
    for idx in phantom_df.index:
        metric = ((w_dist*phantom_df.loc[idx, "distance"] + 
                   w_orient*phantom_df.loc[idx, "diff_in_orientations"] + 
                  w_len*phantom_df.loc[idx, "length_diff"] + 
                  w_K*abs(phantom_df.loc[idx, "hk_real_diff_log"])
                  )*abs(phantom_df.loc[idx, "hk_real_diff_log_aq"])*w_len*(
                  phantom_df.loc[idx, "length"]))
        metric_sum = metric_sum + metric
                
    metric_list.append(metric_sum)   
    if len(phantom_df.index) > 0:
        metric_mean_list.append(metric_sum/len(phantom_df.index))
    else:
        metric_mean_list.append(0)
#=====OVERALL METRIC TO COMPARE MODEL RESULTS==================================
      
    phantom_df = phantom_df.sort_values(by=['inv_identif'], axis=0)
    
    # END OF LOOP
# -=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+

### Double check all of the plots above --> make sure my metrics are correct
### Perhaps I should make a large plot where I combine all of the model runs together.
        
# Sorting out the metric for comparing faults, based on what Peter was talking about
    
plt.close("all")

c_list = ["r", "r", "g", "g", "b", "b", 
          "r", "r", "g", "g", "b", "b", 
          "r", "r", "g", "g", "b", "b"]

# Metric mean list

plt.figure(figsize=(12, 5))
a = np.asarray(metric_mean_list)
b = list(range(n_runs))
plt.bar(b, a, align="center", color=c_list)

plt.xticks(b, modelNames_list, rotation=45)
plt.ylabel('Metric value')
plt.title('Metric mean list')
plt.grid()
plt.savefig(os.path.join(figureDirectory, (modelname + "_meanmet"))) 

# Metric list

plt.figure(figsize=(12, 5))
a = np.asarray(metric_list)
b = list(range(n_runs))
plt.bar(b, a, align="center", color=c_list)

plt.xticks(b, modelNames_list, rotation=45)
plt.ylabel('Metric value')
plt.title('Metric list')
plt.grid()
plt.savefig(os.path.join(figureDirectory, (modelname + "_met"))) 

#------------------------------------------------------------------------------
# Save the lists
dataDirectory = os.path.join(ppc_folder, 'Data')
if not os.path.exists(dataDirectory):
    os.makedirs(dataDirectory)
    
with open(os.path.join(dataDirectory, 'metric_mean_list'), 'wb') as fp:
    pickle.dump(metric_mean_list, fp)
    
with open(os.path.join(dataDirectory, 'metric_list'), 'wb') as fp:
    pickle.dump(metric_list, fp)

with open(os.path.join(dataDirectory, 'rmse_h_all_f_all'), 'wb') as fp:
    pickle.dump(rmse_h_all_f_all, fp)

with open(os.path.join(dataDirectory, 'rmse_a_all_f_all'), 'wb') as fp:
    pickle.dump(rmse_a_all_f_all, fp)
    
with open(os.path.join(dataDirectory, 'rmse_c_all_f_all'), 'wb') as fp:
    pickle.dump(rmse_c_all_f_all, fp)
    
with open(os.path.join(dataDirectory, 'number_of_key_f'), 'wb') as fp:
    pickle.dump(number_of_key_f, fp)
    
with open(os.path.join(dataDirectory, 'drop_rmse_between_fs'), 'wb') as fp:
    pickle.dump(drop_between, fp)
    
# RMSE with key faults only
with open(os.path.join(dataDirectory, 'rmse_h_key_f_all'), 'wb') as fp:
    pickle.dump(rmse_h_key_f_all, fp)

with open(os.path.join(dataDirectory, 'rmse_a_key_f_all'), 'wb') as fp:
    pickle.dump(rmse_a_key_f_all, fp)
    
with open(os.path.join(dataDirectory, 'rmse_c_key_f_all'), 'wb') as fp:
    pickle.dump(rmse_c_key_f_all, fp)
    

# -----------------------------------------------------------------------------
# Plot all K posterior data

# Make a list of dataframes that doesn't include the non-regularisation runs
all_phantom_dfs_reg = all_phantom_dfs[::2]
c_list_reg = c_list[::2]

fig = plt.figure()
ax = plt.subplot(1,1,1)


for mdl in range(len(all_phantom_dfs_reg)):
    for flt in all_phantom_dfs_reg[mdl].index:
        print(flt)
        plt.plot(all_phantom_dfs_reg[mdl].loc[flt, "K_posterior"], mdl, color = c_list_reg[mdl],
                 marker='o')

axes = plt.gca()
axes.set_xscale("log")
