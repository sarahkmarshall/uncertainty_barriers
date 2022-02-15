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
from flopy import mt3d
import pandas as pd
import os
import sys
import string
import pickle
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import flopy.utils.binaryfile as bf
from matplotlib import ticker
import subprocess


functions_path = "C:\workspace\sarahs_functions"

if any(functions_path in s for s in sys.path):
    print("Functions path already in list")
else:
    sys.path.append(r"C:\workspace\sarahs_functions")

import functions_proj_3

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

pprfigureDirectory = r"C:\workspace\Proj3_PilbaraBarriers\Paper3Figs_postrev"

all_well_dens_key_dfs       = []
all_well_dens_ph_dfs        = []

all_well_dens_rmse_h        = []
all_well_dens_rmse_a        = []
all_well_dens_rmse_c        = []

all_well_dens_mmetric       = []
all_well_dens_metric        = []

all_well_dens_filt_stats    = []

all_well_dens_mdl_rmse_f_uf = []

overall_names_list = ["UB_29", "UB_37", "UB_36", "UB_33", "UB_38", "UB_39", "UB_40"]
well_densities = [500, 250, 100, 50, 25, 10, 5]
config_list = ["b", "d", "f", "h", "j", "l", "n", "p", "r"]


# Model set up ===============================================================

nrow = 100
ncol = 200
nlay = 1
delr = 100
delc = 100
ztop = 300.  
zbot = 0.
botm = np.linspace(ztop, zbot, nlay + 1)
hk_aquifer = 1. 
vka = hk_aquifer/10.
sy = 0.1
ss = sy/ztop
laytyp = 1
recharge_flux = 1.3699e-5
prsity = sy
sHead = 300
length_simulation = 1
real_fault_K = hk_aquifer/1000

# Download data ===============================================================

for overallName in overall_names_list:
    
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
    
    all_key_dfs         = []
    all_phantom_dfs     = []
    metric_list         = []
    metric_mean_list    = []
    number_of_key_f     = []
    
    rmse_h_all_mdls     = []
    rmse_h_all_f_all    = []
    
    rmse_a_all_mdls     = []
    rmse_a_all_f_all    = []
    
    rmse_c_all_mdls     = []
    rmse_c_all_f_all    = []
    
    filt_stats_df_all   = []
    
    rmse_whole_mdl_all  = []
    #=====PROPERTIES OF MODEL OUTPUTS==============================================
    length_real_barrier = 5515 #m
       
    #=====ANALYSIS OF MODEL OUTPUTS================================================
    mdl = 0
    for mdl in range(1, len(modelNames_list), 2): # Every second (regularised only)
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
        key_df_f = os.path.join(dataDirectory, "key_dataframe.csv") # or key_df for all faults, key is just those identified >0.8 and low K
        key_df = pd.read_csv(key_df_f)
        
        key_df["length_diff"] = abs(length_real_barrier - key_df["length"])
        
        all_key_dfs.append(key_df)
        
        n_key_f = len(key_df.index)
        number_of_key_f.append(n_key_f)
        
        # Also download all faults
        phantom_df_f = os.path.join(dataDirectory, "phantom_df.csv")
        phantom_df = pd.read_csv(phantom_df_f)
        all_phantom_dfs.append(phantom_df)
        
        #--------------------------------------------------------------------------
        # Download RMSE data - for observation data
        
        # Head
        with open (os.path.join(dataDirectory, 'rmse_h_cum_inv'), 'rb') as fp:
            rmse_h_cum_inv = pickle.load(fp)
        with open (os.path.join(dataDirectory, 'rmse_h_cum_K_p'), 'rb') as fp:
            rmse_h_cum_K_p = pickle.load(fp)
        
        rmse_h_all_f = rmse_h_cum_inv[-1]
        rmse_h_all_f_all.append(rmse_h_all_f) 
        
        rmse_h_all_mdls.append(rmse_h_cum_inv)
        
        # Age
        with open (os.path.join(dataDirectory, 'rmse_a_cum_inv'), 'rb') as fp:
            rmse_a_cum_inv = pickle.load(fp)
        with open (os.path.join(dataDirectory, 'rmse_a_cum_K_p'), 'rb') as fp:
            rmse_a_cum_K_p = pickle.load(fp)
            
        rmse_a_all_f = rmse_a_cum_inv[-1]
        rmse_a_all_f_all.append(rmse_a_all_f) 
        
        rmse_a_all_mdls.append(rmse_a_cum_inv)
    
        # Combined
        with open (os.path.join(dataDirectory, 'rmse_c_cum_inv'), 'rb') as fp:
            rmse_c_cum_inv = pickle.load(fp)
        with open (os.path.join(dataDirectory, 'rmse_c_cum_K_p'), 'rb') as fp:
            rmse_c_cum_K_p = pickle.load(fp)
            
        rmse_c_all_f = rmse_c_cum_inv[-1]
        rmse_c_all_f_all.append(rmse_c_all_f) 
        
        rmse_c_all_mdls.append(rmse_c_cum_inv)
        
        # Download RMSE data - for all model cells
        
        # filtered and unfiltered
        with open (os.path.join(dataDirectory, 'rmse_f_uf_whole_mdl'), 'rb') as fp:
            rmse_f_uf_whole_mdl = pickle.load(fp)
            
        rmse_whole_mdl_all.append(rmse_f_uf_whole_mdl)
        
    #=====RMSE CUT OFF=============================================================
        
        rmse_to_use = rmse_c_cum_inv # rmse_c_cum_inv
        
        drop_between = []
        for i in range(len(rmse_to_use)):
            print(i)
            if i != len(rmse_to_use)-1:
                drop_between.append((rmse_to_use[i] - rmse_to_use[i+1])*100/rmse_to_use[i])
            
    #=====OVERALL METRIC TO COMPARE MODEL RESULTS==================================
    
        w_dist = .001 # Weight for distance
        w_orient = .1 # Weight for orientation/angle/alpha
        w_len = .001 # Weight got length
        w_K = 1 # Weight for K value
    
        metric_sum = 0
        for idx in key_df.index:
            metric = ((w_dist*key_df.loc[idx, "distance"] + 
                       w_orient*key_df.loc[idx, "diff_in_orientations"] + 
                      w_len*key_df.loc[idx, "length_diff"] + 
                      w_K*abs(key_df.loc[idx, "hk_real_diff_log"])
                      )) #*abs(key_df.loc[idx, "hk_real_diff_log_aq"])*w_len*(
                      #key_df.loc[idx, "length"]))
            metric_sum = metric_sum + metric
                    
        metric_list.append(metric_sum)   
        if len(key_df.index) > 0:
            metric_mean_list.append(metric_sum/len(key_df.index))
        else:
            metric_mean_list.append(0)
            
    # -------------------------------------------------------------------------
        # Download summary table of the difference between filtered arrays
        filtered_stats_df_f = os.path.join(dataDirectory, "filtered_arrays_stats.csv") # or key_df for all faults, key is just those identified >0.8 and low K
        filtered_stats_df = pd.read_csv(filtered_stats_df_f)
    
        filt_stats_df_all.append(filtered_stats_df)
            
            
        # END OF FIRST LOOP
        
    all_well_dens_key_dfs.append(all_key_dfs)
    all_well_dens_ph_dfs.append(all_phantom_dfs)
    
    all_well_dens_rmse_h.append(rmse_h_all_mdls)
    all_well_dens_rmse_a.append(rmse_a_all_mdls)
    all_well_dens_rmse_c.append(rmse_c_all_mdls)
    
    all_well_dens_mmetric.append(metric_mean_list)
    all_well_dens_metric.append(metric_list)
    
    all_well_dens_filt_stats.append(filt_stats_df_all)
    
    all_well_dens_mdl_rmse_f_uf.append(rmse_whole_mdl_all)
    
    # END OF SECOND LOOP 
    
# -=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+-=+
# SETTING UP FIGURE PROPERTIES
       
plt.close("all")

fontSize = 14

def cm2inch(value):
    return value/2.54
    
mpl.rc("axes", lw=0.5, edgecolor='grey', labelsize=fontSize)
mpl.rc('lines', lw=1, c='k')
#mpl.rc("grid", c='gray', ls='-', lw='0.5')
#plt.rc('xtick', direction='out', color='0.1')
#plt.rc('ytick', direction='out', color='0.1')

# A column in groundwater is 8.2 cm wide

mpl.rc('font', size=fontSize)
mpl.rc('legend', labelspacing=0.1, frameon=False, fontsize=fontSize)
mpl.rc("figure", figsize=(cm2inch(8.25), cm2inch(8.25*1.25)), titlesize='medium')
#mpl.rc('font', **{'sans-serif' : 'Helvetica','family' : 'sans-serif'}) 
mpl.rc('font',family='Arial')

dpi_value = 300
plt_format = "pdf"

K_colourmap     = "terrain" # "hot" "inferno" "cividis"
sens_colourmap  = "cool"
ph_1_c          = "r"
ph_2_c          = "g"
ph_3_c          = "b"
freq_colourmap  = ""
head_colourmap  = "coolwarm"
age_colourmap   = "jet"
ident_colourmap = "winter"
metr_colourmap  = "nipy_spectral"
n_wells_cmap    = "gnuplot"

c_list = [ph_1_c, ph_1_c, ph_2_c, ph_2_c, ph_3_c, ph_3_c, 
          ph_1_c, ph_1_c, ph_2_c, ph_2_c, ph_3_c, ph_3_c, 
          ph_1_c, ph_1_c, ph_2_c, ph_2_c, ph_3_c, ph_3_c]

c_list_reg = c_list[::2]

#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
#------------------------------------------------------------------------------    
# PLOTTING
# -----------------------------------------------------------------------------
#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

###############################################################################
###############################################################################
# FIGURE ONE: LOCATION OF PHANTOM FAULTS
###############################################################################
###############################################################################

# Need to plot the phantom fault locations
dataDirectory1 = r'C:\workspace\Proj3_PilbaraBarriers\UB_29_b\Data'
dataDirectory2 = r'C:\workspace\Proj3_PilbaraBarriers\UB_29_d\Data'
dataDirectory3 = r'C:\workspace\Proj3_PilbaraBarriers\UB_29_f\Data'

# Real Barrier
re_fault_coordinates = np.load(os.path.join(dataDirectory1, "re_fault_coords.npy"))

# Set up real barriers at different sizes and locations
re_fa_rows_list, re_fa_cols_list, re_fa_length_list = (functions_proj_3.barriers.makeFaultRowCols(
                            re_fault_coordinates, 
                            param_dict["nrow"], 
                            param_dict["ncol"]))

# Remove any faults that turned out not on the grid
for i in range(len(re_fa_length_list)):
    if re_fa_length_list[i] == 0:
        print("Length of barrier is 0 --> no barrier on grid")
        while re_fa_length_list[i] == 0:
            fault_coordinate_redone = functions_proj_3.barriers.makeFaultCoords(1, 
                                                    param_dict["nrow"], 
                                                    param_dict["ncol"])    
            re_fault_coordinates[i] = fault_coordinate_redone[0]
            re_fa_rows_list, re_fa_cols_list, re_fa_length_list = (
                    functions_proj_3.barriers.makeFaultRowCols(re_fault_coordinates, 
                                                               param_dict["nrow"], 
                                                               param_dict["ncol"]))
            re_fa_length_list[0]

    else:
        print("Length of barrier is fine, there is a barrier on the grid")

# Phantom Fault Config 1
ph_fault_coordinates1 = np.load(os.path.join(dataDirectory1, "ph_fault_coords.npy")) 

ph_fa_rows_list1, ph_fa_cols_list1, ph_fa_length_list1 = (functions_proj_3.barriers.makeFaultRowCols(
                                ph_fault_coordinates1, 
                                param_dict["nrow"], 
                                param_dict["ncol"]))

# Remove any faults that turned out not on the grid
for i in range(len(ph_fa_length_list1)):
    if ph_fa_length_list1[i] == 0:
        print("Length of barrier is 0 --> no barrier on grid")
        while ph_fa_length_list1[i] == 0:
            fault_coordinate_redone = functions_proj_3.barriers.makeFaultCoords(1, 
                                                    param_dict["nrow"], 
                                                    param_dict["ncol"])    
            ph_fault_coordinates1[i] = fault_coordinate_redone[0]
            ph_fa_rows_list1, ph_fa_cols_list1, ph_fa_length_list1 = (
                    functions_proj_3.barriers.makeFaultRowCols(ph_fault_coordinates1, 
                                                               param_dict["nrow"], 
                                                               param_dict["ncol"]))
            ph_fa_length_list1[0]

    else:
        print("Length of barrier is fine, there is a barrier on the grid")
        
ph_fault_coordinates2 = np.load(os.path.join(dataDirectory2, "ph_fault_coords.npy"))        

# Phantom Fault Config 2
ph_fa_rows_list2, ph_fa_cols_list2, ph_fa_length_list2 = (functions_proj_3.barriers.makeFaultRowCols(
                                ph_fault_coordinates2, 
                                param_dict["nrow"], 
                                param_dict["ncol"]))
        
for i in range(len(ph_fa_length_list2)):
    if ph_fa_length_list2[i] == 0:
        print("Length of barrier is 0 --> no barrier on grid")
        while ph_fa_length_list2[i] == 0:
            fault_coordinate_redone = functions_proj_3.barriers.makeFaultCoords(1, 
                                                    param_dict["nrow"], 
                                                    param_dict["ncol"])    
            ph_fault_coordinates2[i] = fault_coordinate_redone[0]
            ph_fa_rows_list2, ph_fa_cols_list2, ph_fa_length_list2 = (
                    functions_proj_3.barriers.makeFaultRowCols(ph_fault_coordinates2, 
                                                               param_dict["nrow"], 
                                                               param_dict["ncol"]))
            ph_fa_length_list2[0]

    else:
        print("Length of barrier is fine, there is a barrier on the grid")
        
# Phantom Fault Config 3
ph_fault_coordinates3 = np.load(os.path.join(dataDirectory3, "ph_fault_coords.npy")) 

ph_fa_rows_list3, ph_fa_cols_list3, ph_fa_length_list3 = (functions_proj_3.barriers.makeFaultRowCols(
                        ph_fault_coordinates3, 
                        param_dict["nrow"], 
                        param_dict["ncol"]))

for i in range(len(ph_fa_length_list3)):
    if ph_fa_length_list3[i] == 0:
        print("Length of barrier is 0 --> no barrier on grid")
        while ph_fa_length_list3[i] == 0:
            fault_coordinate_redone = functions_proj_3.barriers.makeFaultCoords(1, 
                                                    param_dict["nrow"], 
                                                    param_dict["ncol"])    
            ph_fault_coordinates3[i] = fault_coordinate_redone[0]
            ph_fa_rows_list3, ph_fa_cols_list3, ph_fa_length_list3 = (
                    functions_proj_3.barriers.makeFaultRowCols(ph_fault_coordinates3, 
                                                               param_dict["nrow"], 
                                                               param_dict["ncol"]))
            ph_fa_length_list3[0]

    else:
        print("Length of barrier is fine, there is a barrier on the grid")

# -------------
# Plot all phantom fault locations

def plot_phantom_faults(ph_fa_cols_list, ph_fa_rows_list, 
                        re_fa_cols_list, re_fa_rows_list):
    for i in range(len(ph_fa_rows_list)):
        plt.plot(ph_fa_cols_list[i], ph_fa_rows_list[i], lw = 3, 
                 color = 'k')
    plt.xlabel("Columns", fontsize=14)
    plt.ylabel("Rows", fontsize=14)
      
    axes = plt.gca()
    axes.set_xlim([0, param_dict["ncol"]])                
    axes.set_ylim([0, param_dict["nrow"]])                
    
    # Now add the real fault
    for i in range(len(re_fa_rows_list)):
        plt.plot(re_fa_cols_list[i], re_fa_rows_list[i], lw = 4, 
                 color = "m")
        
#==========================    
plt.figure(figsize=(8, 30))
#==========================    

plt.subplot(3, 1, 1)
plot_phantom_faults(ph_fa_cols_list1, ph_fa_rows_list1, 
                    re_fa_cols_list, re_fa_rows_list)
plt.subplot(3, 1, 2)
plot_phantom_faults(ph_fa_cols_list2, ph_fa_rows_list2, 
                    re_fa_cols_list, re_fa_rows_list)
plt.subplot(3, 1, 3)
plot_phantom_faults(ph_fa_cols_list3, ph_fa_rows_list3, 
                    re_fa_cols_list, re_fa_rows_list)

plt.savefig(os.path.join(pprfigureDirectory, "all_ph_fts.pdf"), dpi=dpi_value, format=plt_format)
plt.savefig(os.path.join(pprfigureDirectory, "all_ph_fts"), dpi=dpi_value)


###############################################################################
###############################################################################
# FIGURE ONE: LOCATION OF PHANTOM FAULTS - BUT IN UNITS SAME AS MODEL
###############################################################################
###############################################################################

ph_fa_cols_list1
ph_fa_rows_list1
ph_fa_cols_list2
ph_fa_rows_list2
ph_fa_cols_list3
ph_fa_rows_list3

ph_fa_cols_list1_m = []
ph_fa_rows_list1_m = []
ph_fa_cols_list2_m = []
ph_fa_rows_list2_m = []
ph_fa_cols_list3_m = []
ph_fa_rows_list3_m = []

re_fa_cols_list_m = []
re_fa_rows_list_m = []

for fault in range(len(ph_fa_cols_list1)):
    faultcols_new = []
    faultrows_new = []
    for cell in range(len(ph_fa_cols_list1[fault])):
        faultcols_new.append(ph_fa_cols_list1[fault][cell]*delr)
        faultrows_new.append(ph_fa_rows_list1[fault][cell]*delc)
    ph_fa_cols_list1_m.append(faultcols_new)
    ph_fa_rows_list1_m.append(faultrows_new)
        
for fault in range(len(ph_fa_cols_list2)):
    faultcols_new = []
    faultrows_new = []
    for cell in range(len(ph_fa_cols_list2[fault])):
        faultcols_new.append(ph_fa_cols_list2[fault][cell]*delr)
        faultrows_new.append(ph_fa_rows_list2[fault][cell]*delc)
    ph_fa_cols_list2_m.append(faultcols_new)
    ph_fa_rows_list2_m.append(faultrows_new)
        
for fault in range(len(ph_fa_cols_list3)):
    faultcols_new = []
    faultrows_new = []
    for cell in range(len(ph_fa_cols_list3[fault])):
        faultcols_new.append(ph_fa_cols_list3[fault][cell]*delr)
        faultrows_new.append(ph_fa_rows_list3[fault][cell]*delc)
    ph_fa_cols_list3_m.append(faultcols_new)
    ph_fa_rows_list3_m.append(faultrows_new)

for fault in range(len(re_fa_cols_list)):
    faultcols_new = []
    faultrows_new = []
    for cell in range(len(re_fa_cols_list[fault])):
        faultcols_new.append(re_fa_cols_list[fault][cell]*delr)
        faultrows_new.append(re_fa_rows_list[fault][cell]*delc)
    re_fa_cols_list_m.append(faultcols_new)
    re_fa_rows_list_m.append(faultrows_new)

# -------------
# Plot all phantom fault locations

def plot_phantom_faults_m(ph_fa_cols_list, ph_fa_rows_list, 
                        re_fa_cols_list, re_fa_rows_list):
    for i in range(len(ph_fa_rows_list)):
        plt.plot(ph_fa_cols_list[i], ph_fa_rows_list[i], lw = 3, 
                 color = 'k', alpha=0.7)
    # plt.xlabel("$x axis$", fontsize=14)
    # plt.ylabel("$y axis$", fontsize=14)
      
    axes = plt.gca()
    axes.set_xlim([0, param_dict["Lx"]])                
    axes.set_ylim([0, param_dict["Ly"]])                
    
    # Now add the real fault
    for i in range(len(re_fa_rows_list)):
        plt.plot(re_fa_cols_list[i], re_fa_rows_list[i], lw = 4, 
                 color = "m")
        
#==========================    
plt.figure(figsize=(8, 12))
#==========================    

plt.subplot(3, 1, 1)
plot_phantom_faults_m(ph_fa_cols_list1_m, ph_fa_rows_list1_m, 
                    re_fa_cols_list_m, re_fa_rows_list_m)
axes1 = plt.gca()
axes1.axes.get_xaxis().set_visible(False)
axes1.text(0.5, 10500, "(a) Phantom fault configuration 1")


plt.subplot(3, 1, 2)
plot_phantom_faults_m(ph_fa_cols_list2_m, ph_fa_rows_list2_m, 
                    re_fa_cols_list_m, re_fa_rows_list_m)
axes2 = plt.gca()
axes2.axes.get_xaxis().set_visible(False)
axes2.text(0.5, 10500, "(b) Phantom fault configuration 2")

plt.ylabel("$y axis$")

plt.subplot(3, 1, 3)
plot_phantom_faults_m(ph_fa_cols_list3_m, ph_fa_rows_list3_m, 
                    re_fa_cols_list_m, re_fa_rows_list_m)
axes3 = plt.gca()
axes3.text(0.5, 10500, "(c) Phantom fault configuration 3")

plt.xlabel("$x axis$")

plt.savefig(os.path.join(pprfigureDirectory, "all_ph_fts_model.pdf"), dpi=dpi_value, format=plt_format)
plt.savefig(os.path.join(pprfigureDirectory, "all_ph_fts_model"), dpi=dpi_value)

      
# -----------------------------------------------------------------------------
###############################################################################
###############################################################################
# FIGURE TWO: LOCATION OF KEY PHANTOM FAULTS
###############################################################################
###############################################################################
   
# ------ 
# Which phantom faults are "key" for each phantom fault configuration?
# Ph 1 = b, h, n
# Ph 2 = d, j, p
# Ph 3 = f, l, r

# Use "all_well_dens_key_dfs"

def make_comp_df(df_whole_or_key, K_or_i):

# Set up a dataframe to confirm the final hks for each configuration
    
    comp_df = all_phantom_dfs[0].copy() # Just to get the same indexes and set up
    comp_df = comp_df.drop(labels=['scale', 'offset', 'K_prior', 'row_start',
           'col_start', 'row_end', 'col_end', 'hk_log10', 'hk_real_diff',
           'hk_real_diff_log', 'hk_real_diff_aq', 'hk_real_diff_log_aq', 'identif',
           'orientation', 'length', 'mid_row', 'y_mid_loc', 'mid_col', 'x_mid_loc',
           'orientation_norm', 'diff_in_orientations', 'mid_value_r',
           'mid_value_c', 'distance', 'length_diff', 'metric'], axis=1)
    
    for letter in config_list:    
        comp_df[letter] = ['na']*len(comp_df.index)
    
    comp_df = comp_df.drop("K_posterior", axis=1)
    comp_df.index = comp_df.param_name
    
    # Set up empty dictionary
    comp_dict = {}
    for welldensoptn in well_densities:
        comp_dict[welldensoptn] = comp_df
        
    # Now get the value from the results to assign to the dict
    for welldensoptn in range(len(df_whole_or_key)):
        print("Number of wells is: %d" %well_densities[welldensoptn])
        
        for cnfg in range(len(df_whole_or_key[welldensoptn])):
            for idx in df_whole_or_key[welldensoptn][cnfg].index:
                indexname = df_whole_or_key[welldensoptn][cnfg].loc[idx, "param_name"]
                K_post = df_whole_or_key[welldensoptn][cnfg].loc[idx, K_or_i]
                
                # Now assign this value to my new dictionary
                comp_dict[well_densities[welldensoptn]].loc[indexname, 
                                                config_list[cnfg]] = K_post


    return comp_dict

comp_dict_K_all = make_comp_df(all_well_dens_ph_dfs,  "K_posterior")
comp_dict_K_key = make_comp_df(all_well_dens_key_dfs, "K_posterior")
comp_dict_i_all = make_comp_df(all_well_dens_ph_dfs,  "identif")
comp_dict_i_key = make_comp_df(all_well_dens_key_dfs, "identif")

# Find out which faults are identified in total for each phantom fault configuration

b_list = []
d_list = []
f_list = []
h_list = []
j_list = []
l_list = []
n_list = []
p_list = []
r_list = []

for welldensoptn in comp_dict_K_key:
    print(welldensoptn)
    for k, v in comp_dict_K_key[welldensoptn].iterrows():
        
        if v['b'] == 'na':
            pass
        else:
            b_list.append(k)
            
        if v['d'] == 'na':
            pass
        else:
            d_list.append(k)
            
        if v['f'] == 'na':
            pass
        else:
            f_list.append(k)
            
        if v['h'] == 'na':
            pass
        else:
            h_list.append(k)
            
        if v['j'] == 'na':
            pass
        else:
            j_list.append(k)
            
        if v['l'] == 'na':
            pass
        else:
            l_list.append(k)
            
        if v['n'] == 'na':
            pass
        else:
            n_list.append(k)
        
        if v['p'] == 'na':
            pass
        else:
            p_list.append(k)
            
        if v['r'] == 'na':
            pass
        else:
            r_list.append(k)
            
# Ph 1 = b, h, n
ph_1_identified = b_list + h_list + n_list
ph_1_identified_nodupl = list(dict.fromkeys(ph_1_identified)) # This removed duplicates
# Determine indexes of these
f_idx_1 = []
for flt in ph_1_identified_nodupl:
    f_idx_1.append(int(flt[-3:])-1)

# Ph 2 = d, j, p
ph_2_identified = d_list + j_list + p_list
ph_2_identified_nodupl = list(dict.fromkeys(ph_2_identified)) # This removed duplicates
f_idx_2 = []
for flt in ph_2_identified_nodupl:
    f_idx_2.append(int(flt[-3:])-1)

# Ph 3 = f, l, r
ph_3_identified = f_list + l_list + r_list
ph_3_identified_nodupl = list(dict.fromkeys(ph_3_identified)) # This removed duplicates
f_idx_3 = []
for flt in ph_3_identified_nodupl:
    f_idx_3.append(int(flt[-3:])-1)

# Make function to only plot the faults that were identified and are indexed in these lists

def plot_phantom_faults_ident(index_list,
                        ph_fa_cols_list, ph_fa_rows_list, 
                        re_fa_cols_list, re_fa_rows_list):
    for i in index_list:
        plt.plot(ph_fa_cols_list[i], ph_fa_rows_list[i], lw = 3, 
                 color = 'k')
    plt.xlabel("Columns", fontsize=14)
    plt.ylabel("Rows", fontsize=14)
      
    axes = plt.gca()
    axes.set_xlim([0, param_dict["ncol"]])                
    axes.set_ylim([0, param_dict["nrow"]])                
    
    # Now add the real fault
    for i in range(len(re_fa_rows_list)):
        plt.plot(re_fa_cols_list[i], re_fa_rows_list[i], lw = 4, 
                 color = "m")
#==========================          
plt.figure(figsize=(8, 30))
#==========================    

plt.subplot(3, 1, 1)
plot_phantom_faults_ident(f_idx_1, ph_fa_cols_list1, ph_fa_rows_list1, 
                    re_fa_cols_list, re_fa_rows_list)
plt.subplot(3, 1, 2)
plot_phantom_faults_ident(f_idx_2, ph_fa_cols_list2, ph_fa_rows_list2, 
                    re_fa_cols_list, re_fa_rows_list)
plt.subplot(3, 1, 3)
plot_phantom_faults_ident(f_idx_3, ph_fa_cols_list3, ph_fa_rows_list3, 
                    re_fa_cols_list, re_fa_rows_list)

plt.savefig(os.path.join(pprfigureDirectory, "ident_ph_fts.pdf"), dpi=dpi_value, format=plt_format)        
plt.savefig(os.path.join(pprfigureDirectory, "ident_ph_fts"), dpi=dpi_value)        

###############################################################################
###############################################################################
# FIGURE THREE: LOCATION OF KEY PHANTOM FAULTS - WITH FREQUENCY
###############################################################################
###############################################################################

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
# Now I want to show some sort of colour-scale, how many times did each fault appear?
      
# Let's see how many duplicates of each there are in the lists for each phantom fault
def getDuplicatesWithCount(listOfElems):
    ''' Get frequency count of duplicate elements in the given list '''
    dictOfElems = dict()
    # Iterate over each element in list
    for elem in listOfElems:
        print(elem)
        # If element exists in dict then increment its value else add it in dict
        if elem in dictOfElems:
            dictOfElems[elem] += 1
        else:
            dictOfElems[elem] = 1    
        # Filter key-value pairs in dictionary. Keep pairs whose value is greater 
        # than 1 i.e. only duplicate elements from list.
    dictOfElems = { key:value for key, value in dictOfElems.items() if value > 1}
    # Returns a dict of duplicate elements and thier frequency count
    return dictOfElems
    
dictOfPhNumbers_1 = getDuplicatesWithCount(ph_1_identified)
dictOfPhNumbers_2 = getDuplicatesWithCount(ph_2_identified)
dictOfPhNumbers_3 = getDuplicatesWithCount(ph_3_identified)

#Test that the dictionary counts worked
for key, value in dictOfPhNumbers_3.items():
    print(key , ' :: ', value)

# Get max value
max_freq = max(zip(dictOfPhNumbers_1.values(), dictOfPhNumbers_1.keys()))
max(zip(dictOfPhNumbers_2.values(), dictOfPhNumbers_2.keys()))
max(zip(dictOfPhNumbers_3.values(), dictOfPhNumbers_3.keys()))

# It is 21.

# Make a colourscale to plot the frequency of occurrence of faults
colour = cm.Greys(np.linspace(0, 1, max_freq[0]+1))

# Make a function to plot the faults based on their frequency
def plot_phantom_faults_ident_col(freq_dict,
                        ph_fa_cols_list, ph_fa_rows_list, 
                        re_fa_cols_list, re_fa_rows_list, colour):
    
    for key, value in freq_dict.items():
        i = int(key[-3:])-1
        c = colour[value]
        plt.plot(ph_fa_cols_list[i], ph_fa_rows_list[i], lw = 3, 
                 color = c)
    plt.xlabel("Columns", fontsize=14)
    plt.ylabel("Rows", fontsize=14)
      
    axes = plt.gca()
    axes.set_xlim([0, param_dict["ncol"]])                
    axes.set_ylim([0, param_dict["nrow"]])                
    
    # Now add the real fault
    for i in range(len(re_fa_rows_list)):
        plt.plot(re_fa_cols_list[i], re_fa_rows_list[i], lw = 4, 
                 color = "m")
#==========================            
plt.figure(figsize=(8, 30))
#==========================  

plt.subplot(3, 1, 1)
plot_phantom_faults_ident_col(dictOfPhNumbers_1, ph_fa_cols_list1, ph_fa_rows_list1, 
                    re_fa_cols_list, re_fa_rows_list, colour)
plt.subplot(3, 1, 2)
plot_phantom_faults_ident_col(dictOfPhNumbers_2, ph_fa_cols_list2, ph_fa_rows_list2, 
                    re_fa_cols_list, re_fa_rows_list, colour)
plt.subplot(3, 1, 3)
plot_phantom_faults_ident_col(dictOfPhNumbers_3, ph_fa_cols_list3, ph_fa_rows_list3, 
                    re_fa_cols_list, re_fa_rows_list, colour)

plt.savefig(os.path.join(pprfigureDirectory, "ident_ph_fts_freq.pdf"), dpi=dpi_value, format=plt_format)        
plt.savefig(os.path.join(pprfigureDirectory, "ident_ph_fts_freq"), dpi=dpi_value)        

###############################################################################
###############################################################################
# FIGURE FOUR: PLOTTING RMSE FOR ALL SIMULATIONS
###############################################################################
###############################################################################

pattern_list = ["--"]*3 + ["-"]*3 + [":"]*3 # For the observation wells

#==========================  
plt.figure(figsize=(8, 12))
#==========================  

welldensoptn = 0
for welldensoptn in range(len(all_well_dens_rmse_c)):
    ax = plt.subplot(7, 1, welldensoptn+1)
    
    for mdl_n in range(len(all_well_dens_rmse_c[welldensoptn])):
        x_axis = list(range(len(all_well_dens_rmse_c[welldensoptn][mdl_n])-1)) 
        ax.plot(x_axis, all_well_dens_rmse_c[welldensoptn][mdl_n][:-1], c=c_list_reg[mdl_n],
                ls=pattern_list[mdl_n], alpha=0.8)
        
    axes = plt.gca()
    axes.set_ylim(0, 300)
    axes.set_xlim(0, 8)
    plt.grid()
    
# Making legend
plt.plot([-2, -1], [0, 0], c="r", ls="-", label="PF 1")
plt.plot([-2, -1], [0, 0], c="g", ls="-", label="PF 2")
plt.plot([-2, -1], [0, 0], c="b", ls="-", label="PF 3")

plt.plot([-2, -1], [0, 0], c="k", ls="--", label="OW 1")
plt.plot([-2, -1], [0, 0], c="k", ls="-", label="OW 2")
plt.plot([-2, -1], [0, 0], c="k", ls=":", label="OW 3")

axes = plt.gca()
axes.set_ylim(0, 300)
axes.set_xlim(0, 8)
# plt.grid()

plt.legend()
    
plt.savefig(os.path.join(pprfigureDirectory, "RMSE_all_no100.pdf"), dpi=dpi_value, format=plt_format)        
plt.savefig(os.path.join(pprfigureDirectory, "RMSE_all_no100"), dpi=dpi_value)        

###############################################################################
###############################################################################
# FIGURE FIVE: PLOTTING RMSE AVERAGED OUT - average obs wells & ph faults 
###############################################################################
###############################################################################

well_dens_mean_rmse_lists = []
well_dens_all_rmse_lists = []

welldensoptn = 0
for welldensoptn in range(len(all_well_dens_rmse_c)):
    
    rmse1 = all_well_dens_rmse_c[welldensoptn][0][:-1]
    rmse2 = all_well_dens_rmse_c[welldensoptn][1][:-1]
    rmse3 = all_well_dens_rmse_c[welldensoptn][2][:-1]
    rmse4 = all_well_dens_rmse_c[welldensoptn][3][:-1]
    rmse5 = all_well_dens_rmse_c[welldensoptn][4][:-1]
    rmse6 = all_well_dens_rmse_c[welldensoptn][5][:-1]
    rmse7 = all_well_dens_rmse_c[welldensoptn][6][:-1]
    rmse8 = all_well_dens_rmse_c[welldensoptn][7][:-1]
    rmse9 = all_well_dens_rmse_c[welldensoptn][8][:-1]
    
    # The longest is 10 faults, so I want to add in N/A values so I can calculate means
    # when they are all the same length
    rmse1l = []
    rmse2l = []
    rmse3l = []
    rmse4l = []
    rmse5l = []
    rmse6l = []
    rmse7l = []
    rmse8l = []
    rmse9l = []
    for i in range(10):
        print("i = %d" %i)
        if len(rmse1) > i:
            print("1")
            ival = rmse1[i]
            rmse1l.append(ival)
        else:
            print("2")
            rmse1l.append(np.nan)
            
        if len(rmse2) > i:
            ival = rmse2[i]
            rmse2l.append(ival)
        else:
            rmse2l.append(np.nan)
            
        if len(rmse3) > i:
            ival = rmse3[i]
            rmse3l.append(ival)
        else:
            rmse3l.append(np.nan)
            
        if len(rmse4) > i:
            ival = rmse4[i]
            rmse4l.append(ival)
        else:
            rmse4l.append(np.nan)
            
        if len(rmse5) > i:
            ival = rmse5[i]
            rmse5l.append(ival)
        else:
            rmse5l.append(np.nan)
            
        if len(rmse6) > i:
            ival = rmse6[i]
            rmse6l.append(ival)
        else:
            rmse6l.append(np.nan)
            
        if len(rmse7) > i:
            ival = rmse7[i]
            rmse7l.append(ival)
        else:
            rmse7l.append(np.nan)
            
        if len(rmse8) > i:
            ival = rmse8[i]
            rmse8l.append(ival)
        else:
            rmse8l.append(np.nan)
            
        if len(rmse9) > i:
            ival = rmse9[i]
            rmse9l.append(ival)
        else:
            rmse9l.append(np.nan)
        
    rmse_lists =     [np.array(rmse1l), np.array(rmse2l), np.array(rmse3l),
                      np.array(rmse4l), np.array(rmse5l), np.array(rmse6l),
                      np.array(rmse7l), np.array(rmse8l), np.array(rmse9l)]
    
    well_dens_all_rmse_lists.append(rmse_lists)
    
    mean_rmse_list = []
    
    for i in range(len(rmse_lists[0])):
        mean_val = np.nanmean(np.asarray([rmse_lists[0][i], rmse_lists[1][i], 
                                         rmse_lists[2][i], rmse_lists[3][i], 
                                         rmse_lists[4][i], rmse_lists[5][i], 
                                         rmse_lists[6][i], rmse_lists[7][i], 
                                         rmse_lists[8][i]]))
        mean_rmse_list.append(mean_val)
            
    well_dens_mean_rmse_lists.append(mean_rmse_list)                  


colour = cm.brg(np.linspace(0, 1, len(well_densities))) # gnuplot

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#==========================  
plt.figure(figsize=(7, 4))
#==========================  

ax = plt.subplot(1, 1, 1)
x_axis = list(range(0, 10)) 
for welldensoptn in range(len(well_dens_mean_rmse_lists)):
    lbl = well_densities[welldensoptn]
    ax.plot(x_axis, well_dens_mean_rmse_lists[welldensoptn], 
            marker="o", label=lbl, c=colour[welldensoptn])
    
plt.legend(title="Number of wells")
plt.xlabel("Number of phantom faults")
plt.ylabel(r"$RMSE_c$")
plt.grid()
plt.tight_layout()

plt.savefig(os.path.join(pprfigureDirectory, "RMSE_all_mean.pdf"), dpi=dpi_value, format=plt_format)        
plt.savefig(os.path.join(pprfigureDirectory, "RMSE_all_mean"), dpi=dpi_value)        


###############################################################################
###############################################################################
# FIGURE FIVE: PLOTTING RMSE AVERAGED OUT - average obs wells 
###############################################################################
###############################################################################

captions = ["(a) 500 wells", "(b) 250 wells", "(c) 100 wells", "(d) 50 wells",
            "(e) 25 wells", "(f) 10 wells", "(g) 5 wells"]

#==========================  
plt.figure(figsize=(8, 12))
#==========================  

welldensoptn = 0
for welldensoptn in range(len(well_dens_all_rmse_lists)):

    arr_1 = np.asarray([well_dens_all_rmse_lists[welldensoptn][0][:-1],  
             well_dens_all_rmse_lists[welldensoptn][3][:-1], 
             well_dens_all_rmse_lists[welldensoptn][6][:-1]])
    arr_2 = np.asarray([well_dens_all_rmse_lists[welldensoptn][1][:-1],  
             well_dens_all_rmse_lists[welldensoptn][4][:-1],  
             well_dens_all_rmse_lists[welldensoptn][7][:-1]])
    arr_3 = np.asarray([well_dens_all_rmse_lists[welldensoptn][2][:-1],  
             well_dens_all_rmse_lists[welldensoptn][5][:-1],  
             well_dens_all_rmse_lists[welldensoptn][8][:-1]])
    
    avr_1 = np.nanmean(arr_1, axis=0)
    avr_2 = np.nanmean(arr_2, axis=0)
    avr_3 = np.nanmean(arr_3, axis=0)
    
    ax = plt.subplot(7, 1, welldensoptn+1)
    x_axis_1 = list(range(len(avr_1))) 
    ax.plot(x_axis_1, avr_1, c=ph_1_c, ls="-", alpha=0.8, label="Phantom 1")
    
    x_axis_2 = list(range(len(avr_2)))
    ax.plot(x_axis_2, avr_2, c=ph_2_c, ls="-", alpha=0.8, label="Phantom 2")
    
    x_axis_3 = list(range(len(avr_3))) 
    ax.plot(x_axis_3, avr_3, c=ph_3_c, ls="-", alpha=0.8, label="Phantom 3")
        
    axes = plt.gca()
    axes.set_ylim(0, 300)
    axes.set_xlim(0, 8)
    plt.grid()
    plt.ylabel("RMSE")
    
    if welldensoptn != 6:
        # ax.axes.get_xaxis().set_ticks([]) # This removes grid lines also.
        ax.xaxis.set_ticklabels([]) # This should just remove the tick labels
        # ax.xaxis.set_ticks_position('none') # Do this if I want to also remove the ticks
    else:
        pass
    
    plt.text(0.5, 220, captions[welldensoptn])
    
plt.legend()
plt.xlabel("Number of phantom structures")
plt.savefig(os.path.join(pprfigureDirectory, "RMSE_obs_mean.pdf"), dpi=dpi_value, format=plt_format)        
plt.savefig(os.path.join(pprfigureDirectory, "RMSE_obs_mean"), dpi=dpi_value)        

###############################################################################
###############################################################################
# FIGURE SIX: PLOTTING K AVERAGED OUT 
###############################################################################
###############################################################################

# Plotting the mean K value of all results

list_of_K_dicts_1 = [] # One for each well dens optn
list_of_K_dicts_2 = [] # One for each well dens optn
list_of_K_dicts_3 = [] # One for each well dens optn

    
all_well_dens_key_dfs

list_optns = ["b", "d", "f", "h", "j", "l", "n", "p", "r"]

# Which phantom faults are "key" for each phantom fault configuration?
# Ph 1 = b, h, n # 0, 3, 6
# Ph 2 = d, j, p # 1, 4, 7
# Ph 3 = f, l, r # 2, 5, 8

welldensoptn = 6    # Number of wells

for welldensoptn in range(7):
    print(welldensoptn)
        
    # Making fresh new empty dictionaries to save the data in for each round
    
    dictOfKValues_1 = dictOfPhNumbers_1.copy() # To get the same keys
    # But I want to "erase" the previous entries
    for key, value in dictOfKValues_1.items():
        print(key , ' :: ', value)
        dictOfKValues_1[key] = []
        
    dictOfKValues_2 = dictOfPhNumbers_2.copy() # To get the same keys
    # But I want to "erase" the previous entries
    for key, value in dictOfKValues_2.items():
        print(key , ' :: ', value)
        dictOfKValues_2[key] = []
        
    dictOfKValues_3 = dictOfPhNumbers_3.copy() # To get the same keys
    # But I want to "erase" the previous entries
    for key, value in dictOfKValues_3.items():
        print(key , ' :: ', value)
        dictOfKValues_3[key] = []    
        
    for optn in [0, 3, 6]:            # Phantom fault config 1, avr all obs well configs
        
        df = all_well_dens_key_dfs[welldensoptn][optn]
        
        df.index = df.param_name
        idx = df.index 
        
        for key in dictOfKValues_1:
            print(key)
            if key in df.index:
                K_value = df.loc[key, "K_posterior"]
                dictOfKValues_1[key].append(K_value)
            else:
                pass
            
    for optn in [1, 4, 7]:            # Phantom fault config 2, avr all obs well configs
        
        df = all_well_dens_key_dfs[welldensoptn][optn]
        
        df.index = df.param_name
        idx = df.index 
        
        for key in dictOfKValues_2:
            print(key)
            if key in df.index:
                K_value = df.loc[key, "K_posterior"]
                dictOfKValues_2[key].append(K_value)
            else:
                pass
            
    for optn in [2, 5, 8]:            # Phantom fault config 3, avr all obs well configs
        
        df = all_well_dens_key_dfs[welldensoptn][optn]
        
        df.index = df.param_name
        idx = df.index 
        
        for key in dictOfKValues_3:
            print(key)
            if key in df.index:
                K_value = df.loc[key, "K_posterior"]
                dictOfKValues_3[key].append(K_value)
            else:
                pass
            
    # Save one dictionary for each ph fault config per well density option        
    list_of_K_dicts_1.append(dictOfKValues_1)
    list_of_K_dicts_2.append(dictOfKValues_2)
    list_of_K_dicts_3.append(dictOfKValues_3)      


# Getting max and min value in entire dictionaries
mins_dicts = []
maxs_dicts = []
for dict in range(len(list_of_K_dicts_1)):
    dictionary1 = list_of_K_dicts_1[dict]
    for key, value in dictionary1.items():
        print(value)
        if len(value)>0:
            mins_dicts.append(min(value))
            maxs_dicts.append(max(value))
        else:
            pass
    dictionary2 = list_of_K_dicts_2[dict]
    for key, value in dictionary2.items():
        print(value)
        if len(value)>0:
            mins_dicts.append(min(value))
            maxs_dicts.append(max(value))
        else:
            pass
    dictionary3 = list_of_K_dicts_3[dict]  
    for key, value in dictionary3.items():
        print(value)
        if len(value)>0:
            mins_dicts.append(min(value))
            maxs_dicts.append(max(value))
        else:
            pass
        
min_of_mins = min(mins_dicts) # 3.4730468e-09
max_of_maxs = max(maxs_dicts) # 0.99999842
         
# -------------Plotting------------------#
lwr = np.log10(3e-9)
upr = np.log10(1)
rangecol = upr - lwr
colour = cm.terrain(np.linspace(0, 1, int(rangecol+1)*100)) # coolwarm_r
real_plot_c = "k"
real_plot_ls = (0, (1,.6)) #":"
xcaptn = 10*100
ycaptn = 80*100
ax_positions = []
alphaval = 0.7
captn_wells = ["500 wells", "250 wells", "100 wells", "50 wells",
               "25 wells", "10 wells", "5 wells"]

# captn_wells = ["$y$ [m] 500 wells", "$y$ [m] 250 wells", "$y$ [m] 100 wells", "$y$ [m] 50 wells",
#                "$y$ [m] 25 wells", "$y$ [m] 10 wells", "$y$ [m] 5 wells"]

welldensoptn = 0

#==========================  
fig = plt.figure(figsize=(8, 11)) # (8, 13)
#==========================  

pltcount = 1
welldensoptn = 0
for welldensoptn in range(len(list_of_K_dicts_1)):
    
    # Phantom fault config 1
    ax = plt.subplot(7, 3, pltcount)
    dict1 = list_of_K_dicts_1[welldensoptn]
    for key, value in dict1.items():
        print(key , ' :: ', value)
        n = len(value)
        print(n)
        if n == 0:
            pass
        elif n == 1:
            Kval = value[0]
            logKval = np.log10(Kval)
        elif n > 1:
            Kval = np.mean(value)
            logKval = np.log10(Kval)
        else:
            print("Error, the length seems spuriuos - check")
            
        indx_f = int(key[-3:])-1
        # Plot values
        if n > 0:
            c = colour[int((logKval - lwr)*100)]
            ax.plot(ph_fa_cols_list1_m[indx_f], ph_fa_rows_list1_m[indx_f], lw = 3, 
                 color = c, alpha=alphaval)
            axes = plt.gca()
            axes.set_xlim([0, param_dict["Lx"]])                
            axes.set_ylim([0, param_dict["Ly"]])   
            
        else: # Not faults to plot
            pass
        
    # Now add the real fault
    for i in range(len(re_fa_rows_list)):
        plt.plot(re_fa_cols_list_m[i], re_fa_rows_list_m[i], lw = 4, 
                 color = real_plot_c, ls=real_plot_ls)
    
    if pltcount == 19:
        plt.xticks(fontsize=12, rotation=45)
        plt.xticks([10000, 20000]) # , ["", "2 km"]
        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))


    else:
        ax.axes.get_xaxis().set_ticks([])
    
    plt.text(xcaptn, ycaptn, "(%s)" % string.ascii_lowercase[pltcount-1],
             fontweight="bold") 
    ax_positions.append(axes.get_position())
    
    plt.yticks(fontsize=12, rotation=45)
    # plt.yticks([0, 5000, 10000], ["", "", "2 km"])
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))


    # Label the well number
    plt.ylabel(captn_wells[welldensoptn], fontweight="bold", labelpad=-2.2) # On the side
    # plt.text(2.125, 102.125, captn_wells[welldensoptn]) # On the top

     ##### End of subplot
    pltcount = pltcount+1 
        
    # Phantom fault config 2
    ax = plt.subplot(7, 3, pltcount)
    dict2 = list_of_K_dicts_2[welldensoptn]
    for key, value in dict2.items():
        print(key , ' :: ', value)
        n = len(value)
        print(n)
        if n == 0:
            pass
        elif n == 1:
            Kval = value[0]
            logKval = np.log10(Kval)
        elif n > 1:
            Kval = np.mean(value)
            logKval = np.log10(Kval)
        else:
            print("Error, the length seems spuriuos - check")
            
        indx_f = int(key[-3:])-1
        # Plot values
        if n > 0:
            c = colour[int((logKval - lwr)*100)]
            ax.plot(ph_fa_cols_list2_m[indx_f], ph_fa_rows_list2_m[indx_f], lw = 3, 
                 color = c, alpha=alphaval)
            axes = plt.gca()
            axes.set_xlim([0, param_dict["Lx"]])                
            axes.set_ylim([0, param_dict["Ly"]])   
            
        else: # Not faults to plot
            pass
        
    # Now add the real fault
    for i in range(len(re_fa_rows_list)):
        plt.plot(re_fa_cols_list_m[i], re_fa_rows_list_m[i], lw = 4, 
                 color = real_plot_c, ls=real_plot_ls)
        
    if pltcount == 20:
        plt.xticks(fontsize=12, rotation=45)
        plt.xticks([10000, 20000]) # , ["", "2 km"]
        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))


    else:
        ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    plt.text(xcaptn, ycaptn, "(%s)" % string.ascii_lowercase[pltcount-1],
             fontweight="bold")     
     ##### End of subplot
    pltcount = pltcount+1
        
    # Phantom fault config 3
    ax = plt.subplot(7, 3, pltcount)
    dict3 = list_of_K_dicts_3[welldensoptn]
    for key, value in dict3.items():
        print(key , ' :: ', value)
        n = len(value)
        print(n)
        if n == 0:
            pass
        elif n == 1:
            Kval = value[0]
            logKval = np.log10(Kval)
        elif n > 1:
            Kval = np.mean(value)
            logKval = np.log10(Kval)
        else:
            print("Error, the length seems spuriuos - check")
            
        indx_f = int(key[-3:])-1
        # Plot values
        if n > 0:
            c = colour[int((logKval - lwr)*100)]
            ax.plot(ph_fa_cols_list3_m[indx_f], ph_fa_rows_list3_m[indx_f], lw = 3, 
                 color = c, alpha=alphaval)
            axes = plt.gca()
            axes.set_xlim([0, param_dict["Lx"]])                
            axes.set_ylim([0, param_dict["Ly"]])   
            
        else: # Not faults to plot
            pass
        
    # Now add the real fault
    for i in range(len(re_fa_rows_list)):
        plt.plot(re_fa_cols_list_m[i], re_fa_rows_list_m[i], lw = 4, 
                 color = real_plot_c, ls=real_plot_ls)
        
    if pltcount == 21:
        plt.xticks(fontsize=12, rotation=45)
        plt.xticks([10000, 20000]) # , ["", "20 km"]
        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    else:
        ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    
    plt.text(xcaptn, ycaptn, "(%s)" % string.ascii_lowercase[pltcount-1],
             fontweight="bold")        
     ##### End of subplot
    pltcount = pltcount+1        
    # End plot     
    
# Adding common x and y labels
fig.text(0.5, 0.052, '$x$ [m]', ha='center', va='center')
fig.text(0.013, 0.52, '$y$ [m]', ha='center', va='center', rotation='vertical')

ax = plt.subplot(7, 3, 3)
pos1 = ax.get_position() 
x1t = 0.125 +0.2279411764705883*0.5#+ 0.5*0.35294117647058826
x2t = 0.3985294117647059 + 0.2279411764705883*0.5 #+ 0.5*0.6264705882352941
x3t = 0.6720588235294118 + 0.2279411764705883*0.5#+ 0.5*0.9000000000000001
# Adding phantom fault labels
fig.text(x1t, 0.94, 'PS1', fontweight="bold", ha='center', va='center')
fig.text(x2t, 0.94, 'PS2', fontweight="bold", ha='center', va='center')
fig.text(x3t, 0.94, 'PS3', fontweight="bold", ha='center', va='center')


# Adding number of wells labels
plt.subplots_adjust(top = 0.93, hspace=0.15, wspace=0.1) # , bottom=0.01, 

# New method for making colourbar
cbaxes1 = fig.add_axes([0.24, 0.025, 0.5, 0.015]) 

# Answer here : https://stackoverflow.com/questions/28801803/matplotlib-scalarmappable-why-need-to-set-array-if-norm-set/49036899
cmap = mpl.cm.ScalarMappable(
      norm = mcolors.Normalize(lwr, upr), 
      cmap = plt.get_cmap('terrain'))

cmap.set_array([])
cb = fig.colorbar(cmap,  cax=cbaxes1, orientation="horizontal")
cb.set_label("log $K_b$ [m/d]", labelpad=-30, x=1.15, y=0.6, fontsize=14)

plt.savefig(os.path.join(pprfigureDirectory, "AllKVals.pdf"), dpi=dpi_value,format=plt_format)  
plt.savefig(os.path.join(pprfigureDirectory, "AllKVals"), dpi=dpi_value)  

###############################################################################
###############################################################################
# FIGURE SEVEN: PLOTTING FREQUENCY OF FAULT 
###############################################################################
###############################################################################      

list_of_freq_dicts_1 = [] # One for each well dens optn
list_of_freq_dicts_2 = [] # One for each well dens optn
list_of_freq_dicts_3 = [] # One for each well dens optn


welldensoptn = 0    # Number of wells

for welldensoptn in range(7):
    print(welldensoptn)
        
    # Making fresh new empty dictionaries to save the data in for each round
    
    dictOfFreqValues_1 = dictOfPhNumbers_1.copy() # To get the same keys
    # But I want to "erase" the previous entries
    for key, value in dictOfFreqValues_1.items():
        print(key , ' :: ', value)
        dictOfFreqValues_1[key] = []
        
    dictOfFreqValues_2 = dictOfPhNumbers_2.copy() # To get the same keys
    # But I want to "erase" the previous entries
    for key, value in dictOfFreqValues_2.items():
        print(key , ' :: ', value)
        dictOfFreqValues_2[key] = []
        
    dictOfFreqValues_3 = dictOfPhNumbers_3.copy() # To get the same keys
    # But I want to "erase" the previous entries
    for key, value in dictOfFreqValues_3.items():
        print(key , ' :: ', value)
        dictOfFreqValues_3[key] = []    
        
    for optn in [0, 3, 6]:            # Phantom fault config 1, avr all obs well configs
        
        df = all_well_dens_key_dfs[welldensoptn][optn]
        
        df.index = df.param_name
        idx = df.index 
        
        for key in dictOfFreqValues_1:
            print(key)
            if key in df.index:
                current_val = dictOfFreqValues_1[key]
                if len(current_val) > 0:
                    dictOfFreqValues_1[key] = [current_val[0]+1]
                else:
                    dictOfFreqValues_1[key].append(1)
            else:
                pass
            
    for optn in [1, 4, 7]:            # Phantom fault config 2, avr all obs well configs
        
        df = all_well_dens_key_dfs[welldensoptn][optn]
        
        df.index = df.param_name
        idx = df.index 
        
        for key in dictOfFreqValues_2:
            print(key)
            if key in df.index:
                current_val = dictOfFreqValues_2[key]
                if len(current_val) > 0:
                    dictOfFreqValues_2[key] = [current_val[0]+1]
                else:
                    dictOfFreqValues_2[key].append(1)
            else:
                pass
            
    for optn in [2, 5, 8]:            # Phantom fault config 3, avr all obs well configs
        
        df = all_well_dens_key_dfs[welldensoptn][optn]
        
        df.index = df.param_name
        idx = df.index 
        
        for key in dictOfFreqValues_3:
            print(key)
            if key in df.index:
                current_val = dictOfFreqValues_3[key]
                if len(current_val) > 0:
                    dictOfFreqValues_3[key] = [current_val[0]+1]
                else:
                    dictOfFreqValues_3[key].append(1)
            else:
                pass
            
    # Save one dictionary for each ph fault config per well density option        
    list_of_freq_dicts_1.append(dictOfFreqValues_1)
    list_of_freq_dicts_2.append(dictOfFreqValues_2)
    list_of_freq_dicts_3.append(dictOfFreqValues_3)      


# -------------Plotting------------------#
lwr = 0
upr = 3
alphaval = 0.7
rangecol = upr - lwr
colour = cm.viridis_r(np.linspace(0, 1, int(rangecol+1)))
real_plot_c = "k"
real_plot_ls = (0, (1,.6)) #":"
xcaptn = 10*100
ycaptn = 80*100
ax_positions = []
captn_wells = ["500 wells", "250 wells", "100 wells", "50 wells",
               "25 wells", "10 wells", "5 wells"]

welldensoptn = 0

#==========================  
fig = plt.figure(figsize=(8, 13))
#==========================  

pltcount = 1
for welldensoptn in range(len(list_of_freq_dicts_1)):
    
    # Phantom fault config 1
    ax = plt.subplot(7, 3, pltcount)
    dict1 = list_of_freq_dicts_1[welldensoptn]
    for key, value in dict1.items():
        print(key , ' :: ', value)
        
        n = len(value)
        print(n)
        if n > 0:
            freqValue = value[0]
        else:
            freqValue = 0
            
        indx_f = int(key[-3:])-1
        # Plot values
        # if n > 0: # I got rid of this - maybe I will plot those with colour as 0
        c = colour[freqValue]
        ax.plot(ph_fa_cols_list1_m[indx_f], ph_fa_rows_list1_m[indx_f], lw = 3, 
             color = c, alpha=alphaval)
        axes = plt.gca()
        axes.set_xlim([0, param_dict["Lx"]])                
        axes.set_ylim([0, param_dict["Ly"]])   
            
        # else: # Not faults to plot
        #     pass
        
    # Now add the real fault
    for i in range(len(re_fa_rows_list)):
        plt.plot(re_fa_cols_list_m[i], re_fa_rows_list_m[i], lw = 4, 
                 color = real_plot_c, ls=real_plot_ls)
    
    if pltcount == 19:
        plt.xticks(fontsize=12, rotation=45)
        plt.xticks([10000, 20000], ["", "2 km"])
        # ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    else:
        ax.axes.get_xaxis().set_ticks([])
    
    plt.text(xcaptn, ycaptn, "(%s)" % string.ascii_lowercase[pltcount-1],
             fontweight="bold") 
    ax_positions.append(axes.get_position())
    
    plt.yticks(fontsize=12, rotation=45)
    plt.yticks([5000, 10000], ["", "2 km"])
    
        # Label the well number
    plt.ylabel(captn_wells[welldensoptn], labelpad=-2) # On the side
    # plt.text(2.125, 102.125, captn_wells[welldensoptn]) # On the top
    
    ##### End of subplot
    pltcount = pltcount+1 
        
    # Phantom fault config 2
    ax = plt.subplot(7, 3, pltcount)
    dict2 = list_of_freq_dicts_2[welldensoptn]
    for key, value in dict2.items():
        print(key , ' :: ', value)
        
        n = len(value)
        print(n)
        if n > 0:
            freqValue = value[0]
        else:
            freqValue = 0
            
        indx_f = int(key[-3:])-1
        # Plot values
        # if n > 0: # I got rid of this - maybe I will plot those with colour as 0
        c = colour[freqValue]
        ax.plot(ph_fa_cols_list2_m[indx_f], ph_fa_rows_list2_m[indx_f], lw = 3, 
             color = c, alpha=alphaval)
        axes = plt.gca()
        axes.set_xlim([0, param_dict["Lx"]])                
        axes.set_ylim([0, param_dict["Ly"]])   
            
        # else: # Not faults to plot
        #     pass
        
    # Now add the real fault
    for i in range(len(re_fa_rows_list)):
        plt.plot(re_fa_cols_list_m[i], re_fa_rows_list_m[i], lw = 4, 
                 color = real_plot_c, ls=real_plot_ls)
        
    if pltcount == 20:
        plt.xticks(fontsize=12, rotation=45)
        plt.xticks([10000, 20000], ["", "2 km"])
        # ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    else:
        ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    plt.text(xcaptn, ycaptn, "(%s)" % string.ascii_lowercase[pltcount-1],
             fontweight="bold")     
     ##### End of subplot
    pltcount = pltcount+1
        
    # Phantom fault config 3
    ax = plt.subplot(7, 3, pltcount)
    dict3 = list_of_freq_dicts_3[welldensoptn]
    for key, value in dict3.items():
        print(key , ' :: ', value)
        
        n = len(value)
        print(n)
        if n > 0:
            freqValue = value[0]
        else:
            freqValue = 0
            
        indx_f = int(key[-3:])-1
        # Plot values
        # if n > 0: # I got rid of this - maybe I will plot those with colour as 0
        c = colour[freqValue]
        ax.plot(ph_fa_cols_list3_m[indx_f], ph_fa_rows_list3_m[indx_f], lw = 3, 
             color = c, alpha=alphaval)
        axes = plt.gca()
        axes.set_xlim([0, param_dict["Lx"]])                
        axes.set_ylim([0, param_dict["Ly"]])   
            
        # else: # Not faults to plot
        #     pass
        
    # Now add the real fault
    for i in range(len(re_fa_rows_list)):
        plt.plot(re_fa_cols_list_m[i], re_fa_rows_list_m[i], lw = 4, 
                 color = real_plot_c, ls=real_plot_ls)
        
    if pltcount == 21:
        plt.xticks(fontsize=12, rotation=45)
        plt.xticks([10000, 20000], ["", "2 km"])
        # ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    else:
        ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    
    plt.text(xcaptn, ycaptn, "(%s)" % string.ascii_lowercase[pltcount-1],
             fontweight="bold")        
     ##### End of subplot
    pltcount = pltcount+1        
    # End plot     
    
# Adding common x and y labels
# fig.text(0.5, 0.058, '$x$ [m]', ha='center', va='center')
# fig.text(0.013, 0.5, '$y$ [m]', ha='center', va='center', rotation='vertical')

# Adding phantom fault labels
fig.text(x1t, 0.94, 'PS 1', ha='center', va='center')
fig.text(x2t, 0.94, 'PS 2', ha='center', va='center')
fig.text(x3t, 0.94, 'PS 3', ha='center', va='center')

# Adding number of wells labels
plt.subplots_adjust(top = 0.93, hspace=0.3, wspace=0.1) # , bottom=0.01, 

# New method for making colourbar
cbaxes1 = fig.add_axes([0.35, 0.03, 0.35, 0.01]) 

# Answer here : https://stackoverflow.com/questions/28801803/matplotlib-scalarmappable-why-need-to-set-array-if-norm-set/49036899
cmap = mpl.cm.ScalarMappable(
      norm = mcolors.Normalize(lwr, upr), 
      cmap = plt.get_cmap('viridis_r'))

cmap.set_array([])
cb = fig.colorbar(cmap,  cax=cbaxes1, orientation="horizontal")
cb.set_label("Frequency", labelpad=-30, x=1.2, y=0.5, fontsize=14)


plt.savefig(os.path.join(pprfigureDirectory, "FreqStructure.pdf"), dpi=dpi_value,format=plt_format)  
plt.savefig(os.path.join(pprfigureDirectory, "FreqStructure"), dpi=dpi_value)  


###############################################################################
###############################################################################
# FIGURE EIGHT: PLOTTING ORIGINAL K AND AGE ARRAY
###############################################################################
###############################################################################

sngle_example_fldr = r'C:\workspace\Proj3_PilbaraBarriers\UB_29_b'
modelname = "UB_29_b"
modelname_mt = "UB_29_b_MT3D"

# CREATE HEADFILE OBJECTS  

# Load original age file
original_age_file = os.path.join(sngle_example_fldr, "Data", "original_age_file")
original_age_file = np.loadtxt(original_age_file)

# Load original head file
original_head_file = os.path.join(sngle_example_fldr, "Data", "original_head_file")
original_head_file = np.loadtxt(original_head_file)
    

# CONCENTRATION/AGE FILES - FROM TRANSPORT MODEL

# Plotting options
head_cmap = 'coolwarm'
age_cmap = 'jet'

levels_h = np.linspace(0, 400, 51)
levels_a = np.linspace(0, 1400, 15)
extent = (100/2., 20000 - 100/2., 100/2., 10000 - 100/2.)

#=====PLOTTING HEAD AND AGE DATA==========
extent1 = (50.0, 19950.0, 50.0, 9950.0)

#==================================
fig = plt.figure(figsize = (7, 7))
#==================================

# Plot steady-state hydraulic heads

ax1 = fig.add_subplot(2, 1, 1, aspect='equal')

img1 = ax1.imshow(np.flipud(original_head_file[:, :]), extent=extent, cmap='coolwarm_r', 
                     vmin = 200, vmax = 300) # vmax=np.amax(head_1[0, :, :]), vmin = np.amin(head_1[0, :, :])

cs = plt.contour((original_head_file[:, :]), levels=levels_h, extent=extent, 
                           colors = "k") 
plt.clabel(cs, inline=1, fontsize=14, fmt='%1i')

# PLot the real fault location
for i in range(len(re_fa_rows_list)):
        ax1.plot(re_fa_cols_list_m[i], re_fa_rows_list_m[i], lw = 4, 
                 color = real_plot_c, ls=real_plot_ls)
        
        
axes1 = plt.gca()
axes1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

plt.ylabel("$y$ [m]")
axes1.axes.get_xaxis().set_visible(False)


# # # # # # # # # # #
ax2 = fig.add_subplot(2, 1, 2, aspect='equal')

img2 = ax2.imshow(np.flipud(original_age_file[:, :]), extent=extent, cmap=age_cmap) # vmax=np.amax(head_1[0, :, :]), vmin = np.amin(head_1[0, :, :])

cs = plt.contour((original_age_file[:, :]), levels=levels_a, extent=extent, 
                           colors = "k") 
plt.clabel(cs, inline=1, fontsize=14, fmt='%1i')

# PLot the real fault location
for i in range(len(re_fa_rows_list)):
        ax2.plot(re_fa_cols_list_m[i], re_fa_rows_list_m[i], lw = 4, 
                 color = real_plot_c, ls=real_plot_ls)
        
axes2 = plt.gca()
axes2.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
axes2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

plt.xlabel("$x$ [m]")
plt.ylabel("$y$ [m]")

# ---------------------------- #
plt.subplots_adjust(bottom=0.21)

cbaxes1 = fig.add_axes([0.2, 0.1, 0.3, 0.03])  # x, y, length, width
cb1 = plt.colorbar(img1, cax = cbaxes1, orientation="horizontal", format='%1.0f')


cb1.set_label('Hydraulic head [m]')
tick_locator = ticker.MaxNLocator(nbins = 4)
cb1.locator = tick_locator
cb1.update_ticks()

cbaxes2 = fig.add_axes([0.55, 0.1, 0.3, 0.03]) 
cb2 = plt.colorbar(img2, cax = cbaxes2, orientation="horizontal", format='%1.0f')  
       
cb2.set_label('Groundwater age [yrs]')
tick_locator = ticker.MaxNLocator(nbins = 5)
cb2.locator = tick_locator
cb2.update_ticks()

# Add text
pos1 = axes1.get_position() # shows x, y, width, height
pos2 = axes2.get_position()

gap_amount = 0.315
plt.gcf().text(0.19513686463836777, (0.575+gap_amount), "(a) Hydraulic head")
plt.gcf().text(0.19513686463836777, (0.210+gap_amount), "(b) Groundwater age")

plt.savefig(os.path.join(pprfigureDirectory, "HeadAndAgeNoBarr"), dpi=dpi_value)  
plt.savefig(os.path.join(pprfigureDirectory, "HeadAndAgeNoBarr.pdf"), dpi=dpi_value,format=plt_format)  

# Get mean values
mean_h = np.mean(original_head_file[:, :])
mean_a = np.mean(original_age_file[:, :])

# Get max values
max_h = np.max(original_head_file[:, :])
max_a = np.max(original_age_file[:, :])

###############################################################################
###############################################################################
# FIGURE NINE: PLOTTING K AND IDENTIFIABILITY OF SINGLE EXAMPLE KEY FAULTS
# WITH MODEL SIZES
###############################################################################
###############################################################################

# Make a list of phantom fault names--------------------------
ph_fault_name_list = []

for i in range(100):
    ph_fault_name_list.append("hkph_%03d" % (i+1)) 
    
### Make a dictionary of the rows and cols
ph_fa_rows_list = ph_fa_rows_list1_m  # Phantom fault option 1
ph_fa_cols_list = ph_fa_cols_list1_m  # Phantom fault option 1

all_rows_dict = {}
all_cols_dict = {}

for i in range(len(ph_fault_name_list)):
    all_rows_dict.update({ph_fault_name_list[i]: ph_fa_rows_list[i]})
    all_cols_dict.update({ph_fault_name_list[i]: ph_fa_cols_list[i]})

### Set up the phantom and key dataframes for the single example
all_well_dens_key_dfs
all_well_dens_ph_dfs

phantom_df_single =    all_well_dens_ph_dfs[0][0].copy()    # UB_29_b
phantom_df_single.index = phantom_df.param_name
key_df_single =        all_well_dens_key_dfs[0][0].copy()   # UB_29_b
key_df_single.index = key_df_single.param_name

# First plot --------------------------------------------------
# Plot the hydraulic conductivity values of identifiable faults.
 
min_value = np.floor(np.log10(key_df_single["K_posterior"].min()))
max_value = np.ceil(np.log10(key_df_single["K_posterior"].max()))
range_values = max_value*100-min_value*100 #max_value-min_value 
ranges = np.linspace(int(min_value*100), int(max_value*100), int(range_values)+1)
        
K_leg_value = []
for i in ranges:
    K_leg_value.append("$\geqslant$ %d" %i)
                          
color = cm.terrain(np.linspace(0,1,len(ranges)))

#==========================  
fig = plt.figure(figsize=(8, 5))    
ax  = plt.subplot(1,1,1)
#==========================  

for i in range(len(ph_fa_rows_list)):
    
    # First just plot the faults that are not key
    if (phantom_df_single.index[i] in key_df_single.index) == True:
        pass
                            
    else:
        K_col = "0.95"
        
        ax.plot(all_cols_dict[phantom_df_single.index[i]], all_rows_dict[phantom_df_single.index[i]],  
             lw = 4, color = K_col)
        
    # I am going to re-plot the key faults so they are over the top of the grey ones
for i in range(len(ph_fa_rows_list)):  
    if (phantom_df_single.index[i] in key_df_single.index) == True:
        K_val = int(np.log10(phantom_df_single["K_posterior"][i])*100)
        # K_val = np.floor(np.log10((phantom_df_single["K_posterior"][i])))
        K_col = color[np.where(ranges==K_val)[0][0]]
        K_leg = K_leg_value[np.where(ranges==K_val)[0][0]]
        
        ax.plot(all_cols_dict[phantom_df_single.index[i]], all_rows_dict[phantom_df_single.index[i]],  
             lw = 4, color = K_col)
                        
    else:
        pass
        

    
# Plot the real fault location
for i in range(len(re_fa_rows_list)):
        ax.plot(re_fa_cols_list_m[i], re_fa_rows_list_m[i], lw = 4, 
                 color = real_plot_c, ls=real_plot_ls)
        

axes = plt.gca()
axes.set_xlim([0, param_dict["Lx"]])                
axes.set_ylim([0, param_dict["Ly"]]) 
# axes.axes.get_xaxis().set_visible(False)
axes.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
axes.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

# Add text

axes.text(8000, 4100, "1")  # Old 3
axes.text(10550, 6000, "2") # Old 2
axes.text(6200, 1800, "3")  # Old 4
axes.text(12400, 7000, "4") # Old 6


# New method for making colourbar
plt.subplots_adjust(bottom=0.21)

cbaxes1 = fig.add_axes([0.25, 0.07, 0.5, 0.03])  # x, y, length, width

cmap = mpl.cm.ScalarMappable(
      norm = mcolors.Normalize(min_value, max_value), 
      cmap = plt.get_cmap('terrain'))

cmap.set_array([])
cb = fig.colorbar(cmap,  cax=cbaxes1, orientation="horizontal")
cb.set_label("Log $K_b$ [m/d]", labelpad=-30, x=1.2, y=0.5, fontsize=14)

# plt.tight_layout()
fig.text(0.51, 0.13, '$x$ [m]', ha='center', va='center')
fig.text(0.03, 0.53, '$y$ [m]', ha='center', va='center', rotation='vertical')

plt.savefig(os.path.join(pprfigureDirectory, "SingleKandIdent_m.pdf"), dpi=dpi_value,format=plt_format)  
plt.savefig(os.path.join(pprfigureDirectory, "SingleKandIdent_m"), dpi=dpi_value)  

###############################################################################
###############################################################################
# FIGURE TEN: SINGLE RMSE PLOT
###############################################################################
###############################################################################

all_well_dens_rmse_h
all_well_dens_rmse_a
all_well_dens_rmse_c

rmse_h_list = all_well_dens_rmse_h[0][0] # UB_29_b
rmse_a_list = all_well_dens_rmse_a[0][0] # UB_29_b
rmse_c_list = all_well_dens_rmse_c[0][0] # UB_29_b

len(rmse_h_list)
len(rmse_a_list)
len(rmse_c_list)

n_faults_ranking_rmse = list(range(len(rmse_h_list)-1)) + [100] 
len(n_faults_ranking_rmse)

gap = 0.05
gapy1 = 0.05
gapy2 = 1.35
gapy3 = 2.65

#==========================  
plt.figure(figsize=(5,6))
#==========================  

ax1 = plt.subplot(3, 1, 1)
plt.axhline(y=rmse_h_list[-1], color="g", ls=":", lw=3)
plt.plot(n_faults_ranking_rmse[:-1], rmse_h_list[:-1], "go", ls="-", label="Head")

# For legend
plt.plot([0], [-1], "mo", label="Age")
plt.plot([0], [-1], "bo", label="Combined")

plt.ylabel("$RMSE_h$", fontsize = 14, labelpad=3) 
plt.grid(which="both")
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14, rotation=0)

axes = plt.gca()

ax1.set_xticklabels([])
axes.set_ylim([1.9,3.15])
axes.set_xlim([-0.2,4.2])

axes.text(3.6, 3.0, "(a)", fontweight="bold")

# --------------------------------------
ax2 = plt.subplot(3, 1, 2)

plt.axhline(y=rmse_a_list[-1], color="m", ls=":", lw=3)
plt.plot(n_faults_ranking_rmse[:-1], rmse_a_list[:-1], "mo", ls="-", label="Age")


plt.ylabel("$RMSE_a$", fontsize = 14, labelpad=7)
plt.grid(which="both")
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14, rotation=0)

axes = plt.gca()
axes.set_xticklabels([])
axes.set_xlim([-0.2,4.2])

axes.text(3.6, 80, "(b)", fontweight="bold")

# --------------------------------------  
ax3 = plt.subplot(3, 1, 3)
plt.axhline(y=rmse_c_list[-1], color="b", ls=":", lw=3)
plt.plot(n_faults_ranking_rmse[:-1], rmse_c_list[:-1], "bo", ls="-", label="Combined")

plt.ylabel("$RMSE_c$", fontsize = 14, labelpad=-1.2) # 
plt.xlabel("Number of phantom structures", fontsize=14)
plt.grid(which="both")
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14, rotation=0)
plt.text(12, rmse_c_list[0], "no phantom structures")

axes = plt.gca()
axes.set_xlim([-0.2,4.2])

axes.text(3.6, 154, "(c)", fontweight="bold")

# # # # # 
plt.savefig(os.path.join(pprfigureDirectory, "SingleRMSE_h_a_c.pdf"), dpi=dpi_value, format=plt_format)  
plt.savefig(os.path.join(pprfigureDirectory, "SingleRMSE_h_a_c"), dpi=dpi_value)  

# Getting values for the report
rmse_h_list[0]
rmse_a_list[0]
rmse_c_list[0]

rmse_h_list[2] - rmse_h_list[0]
rmse_a_list[2] - rmse_a_list[0]
rmse_c_list[2] - rmse_c_list[0]

rmse_h_list[4] - rmse_h_list[2]
rmse_a_list[4] - rmse_a_list[2]
rmse_c_list[4] - rmse_c_list[2]


# _____________________________________________________________________________
# FIGURE TEN WITH OTHER LINES

# Load the other lines

name = 'rmse_h_w_error.p'
with open(os.path.join(r'C:\workspace\Proj3_PilbaraBarriers\Paper3Figs_postrev1', name), 'rb') as fp:
            rmse_h_w_error = pickle.load(fp)
            
name = 'rmse_a_w_error.p'
with open(os.path.join(r'C:\workspace\Proj3_PilbaraBarriers\Paper3Figs_postrev1', name), 'rb') as fp:
            rmse_a_w_error = pickle.load(fp)
            
name = 'rmse_c_w_error.p'
with open(os.path.join(r'C:\workspace\Proj3_PilbaraBarriers\Paper3Figs_postrev1', name), 'rb') as fp:
            rmse_c_w_error = pickle.load(fp)
        
            
all_well_dens_rmse_h
all_well_dens_rmse_a
all_well_dens_rmse_c

rmse_h_list = all_well_dens_rmse_h[0][0] # UB_29_b
rmse_a_list = all_well_dens_rmse_a[0][0] # UB_29_b
rmse_c_list = all_well_dens_rmse_c[0][0] # UB_29_b

len(rmse_h_list)
len(rmse_a_list)
len(rmse_c_list)

n_faults_ranking_rmse = list(range(len(rmse_h_list)-1)) + [100] 
len(n_faults_ranking_rmse)

gap = 0.05
gapy1 = 0.05
gapy2 = 1.35
gapy3 = 2.65

#==========================  
plt.figure(figsize=(5,6))
#==========================  

ax1 = plt.subplot(3, 1, 1)
plt.axhline(y=rmse_h_list[-1], color="g", ls=":", lw=3)
plt.plot(n_faults_ranking_rmse[:-1], rmse_h_list[:-1], "go", ls="-", label="Head")

plt.plot(n_faults_ranking_rmse[:-1], rmse_h_w_error[:-2], "go", ls="--", label="Head with error")


# For legend
plt.plot([0], [-1], "mo", label="Age")
plt.plot([0], [-1], "bo", label="Combined")

plt.ylabel("$RMSE_h$", fontsize = 14, labelpad=3) 
plt.grid(which="both")
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14, rotation=0)

axes = plt.gca()

ax1.set_xticklabels([])
axes.set_ylim([1.9,3.15])
axes.set_xlim([-0.2,4.2])

axes.text(3.6, 3.0, "(a)", fontweight="bold")

# --------------------------------------
ax2 = plt.subplot(3, 1, 2)

plt.axhline(y=rmse_a_list[-1], color="m", ls=":", lw=3)
plt.plot(n_faults_ranking_rmse[:-1], rmse_a_list[:-1], "mo", ls="-", label="Age")


plt.ylabel("$RMSE_a$", fontsize = 14, labelpad=7)
plt.grid(which="both")
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14, rotation=0)

axes = plt.gca()
axes.set_xticklabels([])
axes.set_xlim([-0.2,4.2])

axes.text(3.6, 80, "(b)", fontweight="bold")

# --------------------------------------  
ax3 = plt.subplot(3, 1, 3)
plt.axhline(y=rmse_c_list[-1], color="b", ls=":", lw=3)
plt.plot(n_faults_ranking_rmse[:-1], rmse_c_list[:-1], "bo", ls="-", label="Combined")

plt.ylabel("$RMSE_c$", fontsize = 14, labelpad=-1.2) # 
plt.xlabel("Number of phantom structures", fontsize=14)
plt.grid(which="both")
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14, rotation=0)
plt.text(12, rmse_c_list[0], "no phantom structures")

axes = plt.gca()
axes.set_xlim([-0.2,4.2])

axes.text(3.6, 154, "(c)", fontweight="bold")

# # # # # 
plt.savefig(os.path.join(pprfigureDirectory, "SingleRMSE_h_a_c_all.pdf"), dpi=dpi_value, format=plt_format)  
plt.savefig(os.path.join(pprfigureDirectory, "SingleRMSE_h_a_c_all"), dpi=dpi_value)  


###############################################################################
###############################################################################
# FIGURE ELEVEN: SINGLE SENSITIVITIY DATA PLOT
###############################################################################
###############################################################################

os.chdir(r'C:\workspace\Proj3_PilbaraBarriers\UB_29_b')
modelname = "UB_29_b"
modelname_mt = "UB_29_b_MT3D"

# Upload observation sensitivity data
sens_obs_data_name = os.path.join(modelname + "_reg.seo")
  
sens_obs_file = pd.read_csv(sens_obs_data_name, delim_whitespace=True, encoding='utf-8-sig') 
sens_obs_file.index = sens_obs_file.Observation

# Get row and column from sensitivity data for plotting
for idx in sens_obs_file.index:
    if (sens_obs_file.loc[idx, "Group"] == "head" or 
        sens_obs_file.loc[idx, "Group"] == "age"):
        sens_obs_file.loc[idx, "Row"] = int(idx[1:4])*param_dict["delc"] # *delc so it's in model coords
        sens_obs_file.loc[idx, "Col"] = int(idx[5:8])*param_dict["delr"]
    else:
        pass
        
# Plot observation sensitivity data

#==========================  
fig = plt.figure(figsize=(7,7))
#==========================  

# plt.suptitle("PEST SENSITIVITY value of observation", fontsize=20)
ax = plt.subplot(2, 1, 1)
# ax.title.set_text("Head")
axes1 = plt.gca() 

# Plot observations sensitivity data
sens_obs_file[sens_obs_file["Group"]=="head"].plot.scatter(x="Col", y="Row", c="Sensitivity", s=30, 
                                               colormap="plasma_r", linewidths=0.0, ax=axes1)


# PLot the real fault location
for i in range(len(re_fa_rows_list)):
        axes1.plot(re_fa_cols_list_m[i], re_fa_rows_list_m[i], lw = 4, 
                 color = real_plot_c, ls=real_plot_ls)

axes1.set_xlim([0, param_dict["Lx"]])
axes1.set_ylim([0, param_dict["Ly"]])
axes1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

axes1.set_xticklabels([])
plt.ylabel("$y$ [m]", labelpad=-3)
plt.xlabel("")
axes1.text(100, 10300, "(a) Hydraulic head observation data", fontsize=14)

# ~ # ~ # ~ # ~    
ax = plt.subplot(2, 1, 2)
# ax.title.set_text("Age")
axes2 = plt.gca()
sens_obs_file[sens_obs_file["Group"]=="age"].plot.scatter(x="Col", y="Row", c="Sensitivity", s=30, 
                                                       colormap="plasma_r", linewidths=0.0, ax=axes2)

# PLot the real fault location
for i in range(len(re_fa_rows_list)):
        axes2.plot(re_fa_cols_list_m[i], re_fa_rows_list_m[i], lw = 4, 
                 color = real_plot_c, ls=real_plot_ls)
        
axes2.set_xlim([0, param_dict["Lx"]])
axes2.set_ylim([0, param_dict["Ly"]])
axes2.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
axes2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

plt.ylabel("$y$ [m]", labelpad=-3)
plt.xlabel("$x$ [m]")
axes2.text(100, 10300, "(b) Groundwater age observation data", fontsize=14)

# Change some of the colorbar properties

cax1 = fig.get_axes()[1]
cax2 = fig.get_axes()[3]

cax1.set_ylabel("Sensitivity [m per m/d $K_b$]", labelpad=6, fontsize=14)
cax2.set_ylabel("Sensitivity [yrs per m/d $K_b$]", labelpad=0, fontsize=14)

plt.savefig(os.path.join(pprfigureDirectory, "Single_Sens.pdf"), dpi=dpi_value, format=plt_format)  
plt.savefig(os.path.join(pprfigureDirectory, "Single_Sens"), dpi=dpi_value)  

###############################################################################
###############################################################################
# FIGURE TWELVE: COMBINED METRIC PLOT
###############################################################################
###############################################################################

# Plotting the mean metric

all_well_dens_mmetric
all_well_dens_metric

which_list = all_well_dens_mmetric

b_list = []
d_list = []
f_list = []
h_list = []
j_list = []
l_list = []
n_list = []
p_list = []
r_list = []

for welldensoptn in range(len(all_well_dens_mmetric)):
    b_list.append(which_list[welldensoptn][0]) #1
    d_list.append(which_list[welldensoptn][1]) #2
    f_list.append(which_list[welldensoptn][2]) #3
    h_list.append(which_list[welldensoptn][3]) #1
    j_list.append(which_list[welldensoptn][4]) #2
    l_list.append(which_list[welldensoptn][5]) #3
    n_list.append(which_list[welldensoptn][6]) #1
    p_list.append(which_list[welldensoptn][7]) #2
    r_list.append(which_list[welldensoptn][8]) #3

python_f_1_metric = (np.array(b_list) + np.array(h_list) + np.array(n_list))/3
python_f_2_metric = (np.array(d_list) + np.array(j_list) + np.array(p_list))/3
python_f_3_metric = (np.array(f_list) + np.array(l_list) + np.array(r_list))/3

all_mean_metric = (python_f_1_metric + python_f_2_metric + python_f_3_metric)/3

#==========================  
fig = plt.figure(figsize=(12,4.5))
#==========================  

plt.plot(well_densities, python_f_1_metric, color=ph_1_c, 
         marker="o", label = "PS1")
plt.plot(well_densities, python_f_2_metric, color=ph_2_c, 
         marker="o", label = "PS2")
plt.plot(well_densities, python_f_3_metric, color=ph_3_c, 
         marker="o", label = "PS3")

plt.plot(well_densities, all_mean_metric, color="k", ls="--", marker="o",
         label = "Mean")

plt.legend()
plt.grid() 

plt.xlabel("Number of wells")
plt.ylabel(r"$\rho $")

plt.savefig(os.path.join(pprfigureDirectory, "AllMetric_welldens.pdf"), dpi=dpi_value,format=plt_format)  
plt.savefig(os.path.join(pprfigureDirectory, "AllMetric_welldens"), dpi=dpi_value)  

###############################################################################
###############################################################################
# FIGURE THIRTEEN: FILTERED ARRAY STATS FOR HEAD AND AGE
###############################################################################
###############################################################################

wells_diff_h_mean_list = []
wells_diff_h_std_list = []

wells_diff_a_mean_list = []
wells_diff_a_std_list = []

for welldensoptn in range(len(all_well_dens_filt_stats)):
    diff_h_mean_list = []
    diff_h_std_list = []
    
    diff_a_mean_list = []
    diff_a_std_list = []
    for mdl in range(len(all_well_dens_filt_stats[welldensoptn])):
        df = all_well_dens_filt_stats[welldensoptn][mdl]
        df.index = df['Unnamed: 0']

        diff_h_mean_list.append(df.loc["diff_head", 'mean'])
        diff_h_std_list.append(df.loc["diff_head", 'st_dev'])

        diff_a_mean_list.append(df.loc["diff_age", 'mean'])
        diff_a_std_list.append(df.loc["diff_age", 'st_dev'])
        
    wells_diff_h_mean_list.append(diff_h_mean_list)
    wells_diff_h_std_list.append(diff_h_std_list)

    wells_diff_a_mean_list.append(diff_a_mean_list)
    wells_diff_a_std_list.append(diff_a_std_list)
    
# Plotting

b_h_mean_list = []
d_h_mean_list = []
f_h_mean_list = []
h_h_mean_list = []
j_h_mean_list = []
l_h_mean_list = []
n_h_mean_list = []
p_h_mean_list = []
r_h_mean_list = []

b_h_std_list = []
d_h_std_list = []
f_h_std_list = []
h_h_std_list = []
j_h_std_list = []
l_h_std_list = []
n_h_std_list = []
p_h_std_list = []
r_h_std_list = []

b_a_mean_list = []
d_a_mean_list = []
f_a_mean_list = []
h_a_mean_list = []
j_a_mean_list = []
l_a_mean_list = []
n_a_mean_list = []
p_a_mean_list = []
r_a_mean_list = []

b_a_std_list = []
d_a_std_list = []
f_a_std_list = []
h_a_std_list = []
j_a_std_list = []
l_a_std_list = []
n_a_std_list = []
p_a_std_list = []
r_a_std_list = []


for welldensoptn in range(len(all_well_dens_filt_stats)):

    b_h_mean_list.append(wells_diff_h_mean_list[welldensoptn][0])
    d_h_mean_list.append(wells_diff_h_mean_list[welldensoptn][1])
    f_h_mean_list.append(wells_diff_h_mean_list[welldensoptn][2])
    h_h_mean_list.append(wells_diff_h_mean_list[welldensoptn][3])
    j_h_mean_list.append(wells_diff_h_mean_list[welldensoptn][4])
    l_h_mean_list.append(wells_diff_h_mean_list[welldensoptn][5])
    n_h_mean_list.append(wells_diff_h_mean_list[welldensoptn][6])
    p_h_mean_list.append(wells_diff_h_mean_list[welldensoptn][7])
    r_h_mean_list.append(wells_diff_h_mean_list[welldensoptn][8])
    
    b_h_std_list.append(wells_diff_h_std_list[welldensoptn][0])
    d_h_std_list.append(wells_diff_h_std_list[welldensoptn][1])
    f_h_std_list.append(wells_diff_h_std_list[welldensoptn][2])
    h_h_std_list.append(wells_diff_h_std_list[welldensoptn][3])
    j_h_std_list.append(wells_diff_h_std_list[welldensoptn][4])
    l_h_std_list.append(wells_diff_h_std_list[welldensoptn][5])
    n_h_std_list.append(wells_diff_h_std_list[welldensoptn][6])
    p_h_std_list.append(wells_diff_h_std_list[welldensoptn][7])
    r_h_std_list.append(wells_diff_h_std_list[welldensoptn][8])
    
    b_a_mean_list.append(wells_diff_a_mean_list[welldensoptn][0])
    d_a_mean_list.append(wells_diff_a_mean_list[welldensoptn][1])
    f_a_mean_list.append(wells_diff_a_mean_list[welldensoptn][2])
    h_a_mean_list.append(wells_diff_a_mean_list[welldensoptn][3])
    j_a_mean_list.append(wells_diff_a_mean_list[welldensoptn][4])
    l_a_mean_list.append(wells_diff_a_mean_list[welldensoptn][5])
    n_a_mean_list.append(wells_diff_a_mean_list[welldensoptn][6])
    p_a_mean_list.append(wells_diff_a_mean_list[welldensoptn][7])
    r_a_mean_list.append(wells_diff_a_mean_list[welldensoptn][8])
    
    b_a_std_list.append(wells_diff_a_std_list[welldensoptn][0])
    d_a_std_list.append(wells_diff_a_std_list[welldensoptn][1])
    f_a_std_list.append(wells_diff_a_std_list[welldensoptn][2])
    h_a_std_list.append(wells_diff_a_std_list[welldensoptn][3])
    j_a_std_list.append(wells_diff_a_std_list[welldensoptn][4])
    l_a_std_list.append(wells_diff_a_std_list[welldensoptn][5])
    n_a_std_list.append(wells_diff_a_std_list[welldensoptn][6])
    p_a_std_list.append(wells_diff_a_std_list[welldensoptn][7])
    r_a_std_list.append(wells_diff_a_std_list[welldensoptn][8])

ph_1_h_mean = (np.array(b_h_mean_list) + np.array(h_h_mean_list) + 
               np.array(n_h_mean_list))/3
ph_2_h_mean = (np.array(d_h_mean_list) + np.array(j_h_mean_list) + 
               np.array(p_h_mean_list))/3
ph_3_h_mean = (np.array(f_h_mean_list) + np.array(l_h_mean_list) + 
               np.array(r_h_mean_list))/3

ph_1_h_std = (np.array(b_h_std_list) + np.array(h_h_std_list) + 
               np.array(n_h_std_list))/3
ph_2_h_std = (np.array(d_h_std_list) + np.array(j_h_std_list) + 
               np.array(p_h_std_list))/3
ph_3_h_std = (np.array(f_h_std_list) + np.array(l_h_std_list) + 
               np.array(r_h_std_list))/3

ph_1_a_mean = (np.array(b_a_mean_list) + np.array(h_a_mean_list) + 
               np.array(n_a_mean_list))/3
ph_2_a_mean = (np.array(d_a_mean_list) + np.array(j_a_mean_list) + 
               np.array(p_a_mean_list))/3
ph_3_a_mean = (np.array(f_a_mean_list) + np.array(l_a_mean_list) + 
               np.array(r_a_mean_list))/3

ph_1_a_std = (np.array(b_a_std_list) + np.array(h_a_std_list) + 
               np.array(n_a_std_list))/3
ph_2_a_std = (np.array(d_a_std_list) + np.array(j_a_std_list) + 
               np.array(p_a_std_list))/3
ph_3_a_std = (np.array(f_a_std_list) + np.array(l_a_std_list) + 
               np.array(r_a_std_list))/3

# Creating error bars
ph_1_h_mean_lwr = ph_1_h_mean + ph_1_h_std
ph_1_h_mean_upr = ph_1_h_mean - ph_1_h_std

ph_2_h_mean_lwr = ph_2_h_mean + ph_2_h_std
ph_2_h_mean_upr = ph_2_h_mean - ph_2_h_std

ph_3_h_mean_lwr = ph_3_h_mean + ph_3_h_std
ph_3_h_mean_upr = ph_3_h_mean - ph_3_h_std


#=================================
fig = plt.figure(figsize=(12,6))
#=================================

ax1 = plt.subplot(2, 1, 1)

# ax1.fill_between(well_densities, ph_1_h_mean_upr, ph_1_h_mean_lwr, alpha=0.5)
ax1.plot(well_densities, ph_1_h_mean, color=ph_1_c, marker="o", label = "Phantom 1")

# ax1.fill_between(well_densities, ph_2_h_mean_upr, ph_2_h_mean_lwr, alpha=0.5)
ax1.plot(well_densities, ph_2_h_mean, color=ph_2_c, marker="o", label = "Phantom 2")

# ax1.fill_between(well_densities, ph_3_h_mean_upr, ph_3_h_mean_lwr, alpha=0.5)
ax1.plot(well_densities, ph_3_h_mean, color=ph_3_c, marker="o", label = "Phantom 3")

plt.ylabel("Mean head difference [m]")
ax1.set_xlim(0, 500)
ax1.text(80, 0.85, '(a) Hydraulic head', ha='center', va='center')
plt.legend()
# plt.grid()
# ax1.axes.get_xaxis().set_ticks([]) # Removed vertical grid lines
ax1.xaxis.set_ticklabels([])
# - - - - 
# - - - - - - - - - - - - - -
ax2 = plt.subplot(2, 1, 2)

# ax1.fill_between(well_densities, ph_1_h_mean_upr, ph_1_h_mean_lwr, alpha=0.5)
ax2.plot(well_densities, ph_1_a_mean, color=ph_1_c, marker="o", label = "Phantom 1")

# ax1.fill_between(well_densities, ph_2_h_mean_upr, ph_2_h_mean_lwr, alpha=0.5)
ax2.plot(well_densities, ph_2_a_mean, color=ph_2_c, marker="o", label = "Phantom 2")

# ax1.fill_between(well_densities, ph_3_h_mean_upr, ph_3_h_mean_lwr, alpha=0.5)
ax2.plot(well_densities, ph_3_a_mean, color=ph_3_c, marker="o", label = "Phantom 3")

plt.xlabel("Number of wells")
plt.ylabel("Mean age difference [yrs]")
ax2.text(70, 25, '(b) Groundwater age', ha='center', va='center')
ax2.set_xlim(0, 500)
# plt.legend()
# plt.grid()

plt.savefig(os.path.join(pprfigureDirectory, "Diff_filteredStats.pdf"), dpi=dpi_value, format=plt_format)  
plt.savefig(os.path.join(pprfigureDirectory, "Diff_filteredStats"), dpi=dpi_value)  

# - - - - - - - - - - - - - - - - - -

# plt.tight_layout()

#%~#%~#%~#%~#%~#%~#%~#%~
plt.close('all')

###############################################################################
###############################################################################
# FIGURE FOURTEEN: MODEL SET UP
###############################################################################
###############################################################################
hk_array_barrier = param_dict["hk_aquifer"] * np.ones((1, 
                                                       100, 
                                                       200), 
                                                       dtype=np.float32)  

for i in range(re_fa_length_list[0]):
    hk_array_barrier[0, int(re_fa_rows_list[0][i]), 
    int(re_fa_cols_list[0][i])] = -3

# Plot K data originally used for the model

plt.figure(figsize=(8, 6))
plt.subplot(1, 1, 1)    

plt.imshow(np.flipud(hk_array_barrier[0, :, :]), extent=extent, cmap='Greys_r', 
                       vmax=1, vmin = -8)
# cbr = plt.colorbar()
# cbr.ax.tick_params(labelsize=14) 
# cbr.set_label('HK (m/d)', fontsize = 14)

ax = plt.gca()
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
   
# plt.grid()
plt.xlabel("$x$ [m]", fontsize = 14)
plt.ylabel("$y$ [m]", fontsize = 14)

plt.axvline(x=0, ymin=0, ymax=1, lw=10, color="r") #, ls=(0, (2,1))
plt.axvline(x=20000, ymin=0, ymax=1, lw=10, color="orange") #, ls=(0, (2,1)),

plt.xlim([0, 20000])
plt.ylim([0, 10000])

# plt.text(4000, 7000, "$K_{a}$ = 1 m/d")
# plt.text(10600, 3500, "$K_{b}$ = 1e-3 m/d")

# plt.arrow(10500, 3700, 500, 0, color="k", lw=1, head_width=100, 
#           overhang=.1, length_includes_head=True, alpha=.8)

# plt.arrow(1000, 1000, 18000, 0, color="dodgerblue", lw=1, head_width=300, 
#           overhang=.1, length_includes_head=True, alpha=.8)

plt.arrow(1000, 9000, 3000, 0, color="dodgerblue", ls="-", lw=1, head_width=300, 
          overhang=.1, length_includes_head=True)
plt.arrow(1000, 5000, 3000, 0, color="dodgerblue", ls="-", lw=1, head_width=300, 
          overhang=.1, length_includes_head=True)
plt.arrow(1000, 1000, 3000, 0, color="dodgerblue", ls="-", lw=1, head_width=300, 
          overhang=.1, length_includes_head=True)

# Add the pilot point array

# Load pilot point dataframe
pp_df = pd.read_csv(r'C:\workspace\Proj3_PilbaraBarriers\UB_51\pilotpoints.csv') # This has python-based numbering

ax.scatter(pp_df.x, pp_df.y,marker='x',c='0.3',s=10)

plt.savefig(pprfigureDirectory + "\model_set_up_notext.pdf", format="pdf", dpi=dpi_value)
plt.savefig(pprfigureDirectory + "\model_set_up_notext", dpi=dpi_value)

###############################################################################
###############################################################################
# FIGURE SIXTEEN: Difference in age and head map for single example
###############################################################################
###############################################################################

# - Setting up the real model -------------------------------------------------
re_fault_coordinates = np.load(os.path.join(dataDirectory, "re_fault_coords.npy"))

# Set up real barriers at different sizes and locations
re_fa_rows_list, re_fa_cols_list, re_fa_length_list = (functions_proj_3.barriers.makeFaultRowCols(
                               re_fault_coordinates, 
                               param_dict["nrow"], 
                               param_dict["ncol"]))

   # Remove any faults that turned out not on the grid
for i in range(len(re_fa_length_list)):
    if re_fa_length_list[i] == 0:
        print("Length of barrier is 0 --> no barrier on grid")
        while re_fa_length_list[i] == 0:
            fault_coordinate_redone = functions_proj_3.barriers.makeFaultCoords(1, 
                                                    param_dict["nrow"], 
                                                    param_dict["ncol"])    
            re_fault_coordinates[i] = fault_coordinate_redone[0]
            re_fa_rows_list, re_fa_cols_list, re_fa_length_list = (
                    functions_proj_3.barriers.makeFaultRowCols(re_fault_coordinates, 
                                                               param_dict["nrow"], 
                                                               param_dict["ncol"]))
            re_fa_length_list[0]

    else:
        print("Length of barrier is fine, there is a barrier on the grid")
   
# - Setting up data output ----------------------------------------------------

all_model_heads = []
all_model_ages = []

# - Setting up the model ------------------------------------------------------

mf_exe = 'mfnwtdbl.exe'
mt3d_exe = 'mt3dusgsdbl.exe'

# Recharge array
recharge_array = recharge_flux * np.ones((nrow, ncol), dtype=np.float32)

# Boundary conditions   
stageleft = float(sHead) # to have a head gradient of 0.001
stageright = float(sHead-100)
bound_sp1 = []
for il in range(nlay):
    condleft = hk_aquifer * (stageleft - zbot) * delc
    condright = hk_aquifer * (stageright - zbot) * delc
    for ir in range(nrow):
        bound_sp1.append([il, ir, 0, stageleft, condleft])
        bound_sp1.append([il, ir, ncol - 1, stageright, condright])
print('Adding ', len(bound_sp1), 'GHBs for stress period 1.')
ghb_spd = {0: bound_sp1}    


# - Make hydraulic conductivity arrays ----------------------------------------

phantom_df_single

hk_array_homog = hk_aquifer * np.ones((1, 100, 200), dtype=np.float32) 
hk_array_single = hk_aquifer * np.ones((1, 100, 200), dtype=np.float32)
hk_array_real = hk_aquifer * np.ones((1, 100, 200), dtype=np.float32)  

# HK array for the single example: POSTERIOR PEST 
for index in phantom_df_single.index:
    print(index)
    rows = all_rows_dict[index]  # Already set up as Ph f 1
    cols = all_cols_dict[index]  # Already set up as Ph f 1
    k_posterior = phantom_df_single.loc[index, "K_posterior"]
    print("K posterior is: %1.3f" %k_posterior)

    for ii in range(len(rows)):
        hk_array_single[0, int(rows[ii]/delc), int(cols[ii]/delr)] = k_posterior
        
# HK array for the real example

for str in range(len(re_fa_rows_list)):
    for i in range(re_fa_length_list[str]):
        hk_array_real[0, int(re_fa_rows_list[str][i]), 
                      int(re_fa_cols_list[str][i])] = real_fault_K

# Check the arrays are correct
plt.figure
plt.imshow(np.flipud(hk_array_single[0, :, :]), extent=extent, cmap='terrain')

plt.figure()
plt.imshow(np.flipud(hk_array_real[0, :, :]), extent=extent, cmap='terrain')

# -----------------------------------------------------------------------------
this_model_folder = r'C:\workspace\Proj3_PilbaraBarriers\PaperThree_Figures\modelRunning'
os.chdir(this_model_folder)

# Running model
# Now I need to run the model to get the age and head data with this file 

for run_n in range(2):
        
    if run_n == 0:
        r_modelname = "realExample" #"homogExample"
        r_modelname_mt = "realExample_MT3D" #"homogExample_MT3D"
    else:
        r_modelname = "singleExample"
        r_modelname_mt = "singleExample_MT3D"
                   
    mf = flopy.modflow.Modflow(r_modelname, exe_name="mfnwtdbl.exe", version='mfnwt') # MODFLOW-NWT
                   
    nper = 1 # Number of model stress periods
    perlen = [length_simulation] 
                   
    nstp = [1] # Number of time steps in each stress period
    tsmult = [1.0]
    steady = [True] # True = steady state
                   
    dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                                   top=ztop, botm=botm[1:], tsmult = tsmult, 
                                   nper=nper, perlen=perlen, nstp=nstp, steady=steady)
                    
                          
    # UPSTREAM WEIGHTING PACKAGE
    if run_n == 0:
        hk_array = hk_array_real
    else:
        hk_array = hk_array_single
               
    uwp = flopy.modflow.mfupw.ModflowUpw(mf, hk=hk_array, vka=vka, sy=sy, ss=ss, 
                                     laytyp=laytyp) 
                   
    # NEWTON SOLVER
    nwt = flopy.modflow.mfnwt.ModflowNwt(mf, headtol=1e-03, fluxtol=0.05, 
                                         maxiterout=100, thickfact=1e-07, 
                                         linmeth=2, iprnwt=1, ibotav=1, 
                                         options='SPECIFIED', Continue=True, 
                                         dbdtheta=0.9, dbdkappa=1e-05, 
                                         dbdgamma=0.0, momfact=0.1, backflag=1, 
                                         maxbackiter=30, backtol=1.05, 
                                         backreduce=0.9, maxitinner=50, 
                                         ilumethod=2, levfill=5, stoptol=1e-10, 
                                         msdr=15, iacl=2, norder=1, level=3, 
                                         north=7, iredsys=1, rrctols=0.0, 
                                         idroptol=1, epsrn=0.0001, 
                                         hclosexmd=0.0001, mxiterxmd=200) 
                   
                           
    # RECHARGE (RCH) PACKAGE
                   
    rch = flopy.modflow.ModflowRch(mf, rech=recharge_array)
                   
    # GENERAL HEAD BOUNDARY (GHB) PACKAGE 
    ghb = flopy.modflow.ModflowGhb(mf, stress_period_data=ghb_spd) 
    
    # OC PACKAGE
                                          
    spd = {}
    for strsp in range(nper):
        tmstp = nstp[strsp]
        for time_ in range(tmstp):
            spd[strsp, time_] = ['save head', 'print budget', 'save budget'] 
                                                      
    oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd)
                                      
    bas = flopy.modflow.ModflowBas(mf,ibound=1., strt=300., ifrefm=True, 
                                   ixsec=False, hnoflo=-999.99)     
    
                   
    # --- LINKING FILE --- #
                   
    lmt = flopy.modflow.ModflowLmt(mf, output_file_name= (r_modelname + '_mt3dLink.ftl'),
                                   output_file_header='extended',
                                   output_file_format='unformatted')   
                     
    mf.write_input() # Write the model input files
                       
    subprocess.run([mf_exe, r_modelname]) # Run flow model 
                                        
    # Instantiate MT3D-USGS object in flopy
                   
    mt = mt3d.Mt3dms(modflowmodel=mf, modelname=r_modelname_mt, model_ws=this_model_folder, 
                     version=mt3d_version, namefile_ext='mtnam', exe_name=mt3d_exe,  
                     ftlfilename=(r_modelname + '_mt3dLink.ftl')) 
                          
    # BASIC TRANSPORT PACKAGE
                   
    ncomp = 1 # Total number of chemical species in simulation  
    mcomp = 1 # Total number of "mobile" species 
    sconc= np.zeros((nlay, nrow, ncol), dtype=np.float32)  # initial conc. = 0 yrs   
    nprs = 0 # Flag indicating frequency of output. = 0: results not saved except 
                       
    # Instantiate basic transport (BTN) package for MT3D-USGS
    btn = mt3d.Mt3dBtn(mt, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper, perlen=perlen, 
                       ncomp=ncomp, mcomp=mcomp, sconc=sconc, prsity=prsity,
                       delr=delr, delc=delc, icbund=1, ifmtcn=-1, savucn=True,
                       nprs=nprs, nprobs=1, cinact=0, ssflag='SState', laycon=1) 
               
    # Instantiate advection (ADV) package for MT3D-USGS
    adv = mt3d.Mt3dAdv(mt, mixelm=0)  
                   
    # Instatiate generalized conjugate gradient solver (GCG) package for MT3D-USGS
    gcg = mt3d.Mt3dGcg(mt, mxiter=30, iter1=50, isolve=2, accl=1, cclose=1e-06)
                   
    # REACTION PACKAGE
                   
    rc1 = np.zeros((nlay, nrow, ncol), dtype=np.float32)
    rc1[:, :, :] = -1/365.25
                       
    isothm = 0      # 0 = no sorption
    ireact = 100    # 100 = zeroth-order reaction option
    rc1 = rc1       # First order reaction rate for diffolved phase of first species.
                       
    rct= mt3d.Mt3dRct(mt, isothm=isothm, ireact=ireact, igetsc=0, rc1=rc1) 
                   
    # SOURCE-SINK MIXING PACKAGE
                   
    crch = np.zeros((nrow, ncol), dtype=np.float32) # The age of recharging water is 0
                   
    itype = mt3d.Mt3dSsm.itype_dict()
                 
    ssm = mt3d.Mt3dSsm(mt, crch=crch) #, stress_period_data=ssm_data) 
                      
    # DISPERSION PACKAGE 
    
    al = 1.5 # The longitudinal dispersivity, default = 0.01
    dmcoef = 1E-4 # Effective molecular diffusion coefficient (for water in my model), default = 1E-9
                                                  
    dsp = mt3d.Mt3dDsp(mt, al=al, dmcoef=dmcoef) 
     
    # Write input
                                 
    mt.write_input()
    mt.write_name_file()
               
    # "Manual" changes to the input files
    conc_filename_dissolved = r_modelname_mt + ".ucn"
    conc_filename_sorbed = r_modelname_mt + "_S.ucn"
    mass_filename = r_modelname_mt + ".mas"
    cnf_filename = r_modelname_mt + ".cnf"
    
    ##Add a line to the MT3D .mtnam file re-naming output UCN file + MAS,CNF
    mt_name_file = r_modelname_mt + '.mtnam'
    
    namfile = open(mt_name_file, 'a')
    namfile.write('data(binary) 201 %s \n' %(conc_filename_dissolved))
    namfile.close()
    
    namfile = open(mt_name_file, 'a')
    namfile.write('data(binary) 301 %s \n' %(conc_filename_sorbed))
    namfile.close()
      
    namfile = open(mt_name_file, 'a')
    namfile.write('data 601 %s \n' %(mass_filename))
    namfile.close()
    
    namfile = open(mt_name_file, 'a')
    namfile.write('data 17 %s \n' %(cnf_filename))
    namfile.close()
        
    ###For USGS need to add DRYCELL keyword to the .btn file on line 3
    mt_btn_file = r_modelname_mt + '.btn'
    
    
    btnfile = open(mt_btn_file, "r")
    contents = btnfile.readlines()
    btnfile.close()
    contents.insert(2, "DRYCELL \n")
    btnfile = open(mt_btn_file, "w")
    contents = "".join(contents)
    btnfile.write(contents)
    btnfile.close()
                   
    # ADD SSTATE FLAG manually in correct spot
                   
    fpath = os.path.join(this_model_folder, mt_btn_file)
    with open(mt_btn_file, 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
    
        a = length_simulation # 
        b = "%d         1         1 S" %a
        c = "%d         1         1         SState" %a
        ##
    
        for line in lines:
            if b in line:
                            line = line.replace(b, c)
            f.write(line)
    
    mt3d_namefile = r_modelname_mt + ".mtnam"
     
    subprocess.run([mt3d_exe, mt3d_namefile]) # Run command 
                   
    # -----------------------------------------------------------------------------              
    # CREATE HEADFILE OBJECTS  
       
    headobj = bf.HeadFile(r_modelname + '.hds')
    data = headobj.get_alldata()
    times = headobj.get_times()
    head_raster = headobj.get_data(totim=times[0]) # One time period: steady-state
       
    all_model_heads.append(head_raster)
       
    # CONCENTRATION/AGE FILES - FROM TRANSPORT MODEL
       
    directage_file = r_modelname_mt + '.ucn'       
    ucnobjc_direct_age = flopy.utils.binaryfile.UcnFile(directage_file, precision="double")        
    age_times = ucnobjc_direct_age.get_times()        
    age_raster = ucnobjc_direct_age.get_data(totim=age_times[0])
       
    all_model_ages.append(age_raster)
                      
    #------------------------------------------------------------------------------
    # END Model running

# Create rasters for difference in head and age
head_diff = all_model_heads[1] - all_model_heads[0]
age_diff = all_model_ages[1] - all_model_ages[0]

np.shape(head_diff)
np.shape(age_diff)

# Remove the barriers to see the max and min without them

head_diff_nobarr = head_diff.copy()
age_diff_nobarr = age_diff.copy()

for index in phantom_df_single.index:
    print(index)
    rows = all_rows_dict[index]  # Already set up as Ph f 1
    cols = all_cols_dict[index]  # Already set up as Ph f 1
    k_posterior = phantom_df_single.loc[index, "K_posterior"]
    print("K posterior is: %1.3f" %k_posterior)

    for ii in range(len(rows)):
        head_diff_nobarr[0, int(rows[ii]/delc), int(cols[ii]/delr)] = np.nan
        age_diff_nobarr[0, int(rows[ii]/delc), int(cols[ii]/delr)] = np.nan

min_h_nobarr = np.nanmin(head_diff_nobarr[0, :, :])
min_a_nobarr = np.nanmin(age_diff_nobarr[0, :, :])
max_h_nobarr = np.nanmax(head_diff_nobarr[0, :, :])
max_a_nobarr = np.nanmax(age_diff_nobarr[0, :, :])

print("Min head: %2.10f, Max head: %2.10f" %(min_h_nobarr, max_h_nobarr))
print("Min age: %2.3f, Max age: %2.3f" %(min_a_nobarr, max_a_nobarr))

# Plot difference in head and age

levels_diff_h = [-2, 0, 2]
levels_diff_a = [-25, 0, 25]

#==================================
fig = plt.figure(figsize = (7, 7))
#==================================

# Plot steady-state hydraulic heads

ax1 = fig.add_subplot(2, 1, 1, aspect='equal')

img1 = ax1.imshow(np.flipud(head_diff_nobarr[0, :, :]), extent=extent, cmap='PiYG',
                  vmin=-17, vmax=17)
'''
cs = plt.contour((head_diff_nobarr[0, :, :]), levels=levels_diff_h, extent=extent, 
                           colors = "k") 
plt.clabel(cs, inline=1, fontsize=14, fmt='%1i')
'''
# Plot the real fault location
for i in range(len(re_fa_rows_list)):
        ax1.plot(re_fa_cols_list_m[i], re_fa_rows_list_m[i], lw = 4, 
                 color = real_plot_c, ls=real_plot_ls)

axes1 = plt.gca()
plt.ylabel("$y$ [m]")
axes1.axes.get_xaxis().set_visible(False)
axes1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

# # # # # # # # # # #
ax2 = fig.add_subplot(2, 1, 2, aspect='equal')

img2 = ax2.imshow(np.flipud(age_diff_nobarr[0, :, :]), extent=extent, cmap="PuOr",
                  vmin = -310, vmax = 310)
'''
cs = plt.contour((age_diff_nobarr[0, :, :]), levels=levels_diff_a, extent=extent, 
                           colors = "k") 
plt.clabel(cs, inline=1, fontsize=14, fmt='%1i')
'''
# Plot the real fault location
for i in range(len(re_fa_rows_list)):
        ax2.plot(re_fa_cols_list_m[i], re_fa_rows_list_m[i], lw = 4, 
                 color = real_plot_c, ls=real_plot_ls)
        

axes2 = plt.gca()
plt.xlabel("$x$ [m]")
plt.ylabel("$y$ [m]")
axes2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
axes2.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

# ---------------------------- #
plt.subplots_adjust(bottom=0.21)

cbaxes1 = fig.add_axes([0.2, 0.1, 0.3, 0.03]) # [0.133, 0.1, 0.32, 0.03] 
cb1 = plt.colorbar(img1, cax = cbaxes1, orientation="horizontal", format='%1.0f')  
cb1.set_label('Head difference [m]')
tick_locator = ticker.MaxNLocator(nbins = 6)
cb1.locator = tick_locator
cb1.update_ticks()

cbaxes2 = fig.add_axes([0.55, 0.1, 0.3, 0.03]) # [0.547, 0.1, 0.32, 0.03] 
cb2 = plt.colorbar(img2, cax = cbaxes2, orientation="horizontal", format='%1.0f')  
cb2.set_label('Age difference [yrs]')
tick_locator = ticker.MaxNLocator(nbins = 6)
cb2.locator = tick_locator
cb2.update_ticks()

# Add text
pos1 = axes1.get_position() # shows x, y, width, height
pos2 = axes2.get_position()

gap_amount = 0.315
plt.gcf().text(0.19513686463836777, (0.575+gap_amount), "(a) Hydraulic head difference")
plt.gcf().text(0.19513686463836777, (0.210+gap_amount), "(b) Groundwater age difference")

plt.savefig(os.path.join(pprfigureDirectory, "HeadAndAgeDiff.pdf"), dpi=dpi_value,format=plt_format)  
plt.savefig(os.path.join(pprfigureDirectory, "HeadAndAgeDiff"), dpi=dpi_value)  

###############################################################################
###############################################################################
# FIGURE FOUR: RMSE 100 only
###############################################################################
###############################################################################

well_densities # x axis

head_ps1        = []
age_ps1         = []
c_ps1           = []

head_ps2        = []
age_ps2         = []
c_ps2           = []

head_ps3        = []
age_ps3         = []
c_ps3           = []

rmse_h_f_ps1    = []
rmse_a_f_ps1    = []
rmse_h_uf_ps1   = []
rmse_a_uf_ps1   = []

rmse_h_f_ps2    = []
rmse_a_f_ps2    = []
rmse_h_uf_ps2   = []
rmse_a_uf_ps2   = []

rmse_h_f_ps3    = []
rmse_a_f_ps3    = []
rmse_h_uf_ps3   = []
rmse_a_uf_ps3   = []


for welldensoptn in range(len(well_densities)):
    # ps1
    head_ps1.append((all_well_dens_rmse_h[welldensoptn][0][-1:][0] + all_well_dens_rmse_h[welldensoptn][3][-1:][0] +
    all_well_dens_rmse_h[welldensoptn][6][-1:][0])/3)
    
    age_ps1.append((all_well_dens_rmse_a[welldensoptn][0][-1:][0] + all_well_dens_rmse_a[welldensoptn][3][-1:][0] +
    all_well_dens_rmse_a[welldensoptn][6][-1:][0])/3)
    
    c_ps1.append((all_well_dens_rmse_c[welldensoptn][0][-1:][0] + all_well_dens_rmse_c[welldensoptn][3][-1:][0] +
    all_well_dens_rmse_c[welldensoptn][6][-1:][0])/3)
    
    rmse_h_f_ps1.append((all_well_dens_mdl_rmse_f_uf[welldensoptn][0][0] + all_well_dens_mdl_rmse_f_uf[welldensoptn][3][0] +
    all_well_dens_mdl_rmse_f_uf[welldensoptn][6][0])/3)
    
    rmse_a_f_ps1.append((all_well_dens_mdl_rmse_f_uf[welldensoptn][0][1] + all_well_dens_mdl_rmse_f_uf[welldensoptn][3][1] +
    all_well_dens_mdl_rmse_f_uf[welldensoptn][6][1])/3)
    
    rmse_h_uf_ps1.append((all_well_dens_mdl_rmse_f_uf[welldensoptn][0][2] + all_well_dens_mdl_rmse_f_uf[welldensoptn][3][2] +
    all_well_dens_mdl_rmse_f_uf[welldensoptn][6][2])/3)
    
    rmse_a_uf_ps1.append((all_well_dens_mdl_rmse_f_uf[welldensoptn][0][3] + all_well_dens_mdl_rmse_f_uf[welldensoptn][3][3] +
    all_well_dens_mdl_rmse_f_uf[welldensoptn][6][3])/3)
    
    # ps2
    head_ps2.append((all_well_dens_rmse_h[welldensoptn][1][-1:][0] + all_well_dens_rmse_h[welldensoptn][4][-1:][0] +
    all_well_dens_rmse_h[welldensoptn][7][-1:][0])/3)
    
    age_ps2.append((all_well_dens_rmse_a[welldensoptn][1][-1:][0] + all_well_dens_rmse_a[welldensoptn][4][-1:][0] +
    all_well_dens_rmse_a[welldensoptn][7][-1:][0])/3)
    
    c_ps2.append((all_well_dens_rmse_c[welldensoptn][1][-1:][0] + all_well_dens_rmse_c[welldensoptn][4][-1:][0] +
    all_well_dens_rmse_c[welldensoptn][7][-1:][0])/3)
    
    rmse_h_f_ps2.append((all_well_dens_mdl_rmse_f_uf[welldensoptn][1][0] + all_well_dens_mdl_rmse_f_uf[welldensoptn][4][0] +
    all_well_dens_mdl_rmse_f_uf[welldensoptn][7][0])/3)
    
    rmse_a_f_ps2.append((all_well_dens_mdl_rmse_f_uf[welldensoptn][1][1] + all_well_dens_mdl_rmse_f_uf[welldensoptn][4][1] +
    all_well_dens_mdl_rmse_f_uf[welldensoptn][7][1])/3)
    
    rmse_h_uf_ps2.append((all_well_dens_mdl_rmse_f_uf[welldensoptn][1][2] + all_well_dens_mdl_rmse_f_uf[welldensoptn][4][2] +
    all_well_dens_mdl_rmse_f_uf[welldensoptn][7][2])/3)
    
    rmse_a_uf_ps2.append((all_well_dens_mdl_rmse_f_uf[welldensoptn][1][3] + all_well_dens_mdl_rmse_f_uf[welldensoptn][4][3] +
    all_well_dens_mdl_rmse_f_uf[welldensoptn][7][3])/3)
    
    # ps3
    head_ps3.append((all_well_dens_rmse_h[welldensoptn][2][-1:][0] + all_well_dens_rmse_h[welldensoptn][5][-1:][0] +
    all_well_dens_rmse_h[welldensoptn][8][-1:][0])/3)
        
    age_ps3.append((all_well_dens_rmse_a[welldensoptn][2][-1:][0] + all_well_dens_rmse_a[welldensoptn][5][-1:][0] +
    all_well_dens_rmse_a[welldensoptn][8][-1:][0])/3)
            
    c_ps3.append((all_well_dens_rmse_c[welldensoptn][2][-1:][0] + all_well_dens_rmse_c[welldensoptn][5][-1:][0] +
    all_well_dens_rmse_c[welldensoptn][8][-1:][0])/3)
    
    rmse_h_f_ps3.append((all_well_dens_mdl_rmse_f_uf[welldensoptn][2][0] + all_well_dens_mdl_rmse_f_uf[welldensoptn][5][0] +
    all_well_dens_mdl_rmse_f_uf[welldensoptn][8][0])/3)
    
    rmse_a_f_ps3.append((all_well_dens_mdl_rmse_f_uf[welldensoptn][2][1] + all_well_dens_mdl_rmse_f_uf[welldensoptn][5][1] +
    all_well_dens_mdl_rmse_f_uf[welldensoptn][8][1])/3)
    
    rmse_h_uf_ps3.append((all_well_dens_mdl_rmse_f_uf[welldensoptn][2][2] + all_well_dens_mdl_rmse_f_uf[welldensoptn][5][2] +
    all_well_dens_mdl_rmse_f_uf[welldensoptn][8][2])/3)
    
    rmse_a_uf_ps3.append((all_well_dens_mdl_rmse_f_uf[welldensoptn][2][3] + all_well_dens_mdl_rmse_f_uf[welldensoptn][5][3] +
    all_well_dens_mdl_rmse_f_uf[welldensoptn][8][3])/3)
    
# Taking rmse for every cell in the model. Filtered and unfiltered
all_well_dens_mdl_rmse_f_uf # [rmse_head_f, rmse_age_f, rmse_head_uf, rmse_age_uf]

pattern_list = ["-"] + [":"] + ["--"] # For the observation wells

avr_rmse_h = (np.asarray(head_ps1) + np.asarray(head_ps2) + np.asarray(head_ps3))/3
avr_rmse_a = (np.asarray(age_ps1) + np.asarray(age_ps2) + np.asarray(age_ps3))/3
avr_rmse_c = (np.asarray(c_ps1) + np.asarray(c_ps2) + np.asarray(c_ps3))/3

avr_rmse_h_f  = (np.asarray(rmse_h_f_ps1) + np.asarray(rmse_h_f_ps2) + np.asarray(rmse_h_f_ps3))/3
avr_rmse_a_f  = (np.asarray(rmse_a_f_ps1) + np.asarray(rmse_a_f_ps2) + np.asarray(rmse_a_f_ps3))/3
avr_rmse_c_f  = avr_rmse_h_f*25 + avr_rmse_a_f

avr_rmse_h_uf = (np.asarray(rmse_h_uf_ps1) + np.asarray(rmse_h_uf_ps2) + np.asarray(rmse_h_uf_ps3))/3
avr_rmse_a_uf = (np.asarray(rmse_a_uf_ps1) + np.asarray(rmse_a_uf_ps2) + np.asarray(rmse_a_uf_ps3))/3
avr_rmse_c_uf = avr_rmse_h_uf*25 + avr_rmse_a_uf

c_ps1_f     = np.array(rmse_h_f_ps1)*25 + np.array(rmse_a_f_ps1)
c_ps1_uf    = np.array(rmse_h_uf_ps1)*25 + np.array(rmse_a_uf_ps1)
c_ps2_f     = np.array(rmse_h_f_ps2)*25 + np.array(rmse_a_f_ps2)
c_ps2_uf    = np.array(rmse_h_uf_ps2)*25 + np.array(rmse_a_uf_ps2)
c_ps3_f     = np.array(rmse_h_f_ps3)*25 + np.array(rmse_a_f_ps3)
c_ps3_uf    = np.array(rmse_h_uf_ps3)*25 + np.array(rmse_a_uf_ps3)

double_check_f  = (c_ps1_f+c_ps2_f+c_ps3_f)/3    # they are the same :)
double_check_uf = (c_ps1_uf+c_ps2_uf+c_ps3_uf)/3 # they are the same :)

#==========================  
plt.figure(figsize=(8, 12))
#==========================  

# Head plot - - - - - - - - - - - - - - - - - - - - - - - - -
ax1 = plt.subplot(3, 1, 1)

ax1.plot(well_densities, head_ps1,      c="r", ls = pattern_list[0])
ax1.plot(well_densities, rmse_h_f_ps1,  c="r", ls = pattern_list[1])
# ax1.plot(well_densities, rmse_h_uf_ps1, c="r", ls = pattern_list[2])

ax1.plot(well_densities, head_ps2,      c="g", ls = pattern_list[0])
ax1.plot(well_densities, rmse_h_f_ps2,  c="g", ls = pattern_list[1])
# ax1.plot(well_densities, rmse_h_uf_ps2, c="g", ls = pattern_list[2])

ax1.plot(well_densities, head_ps3,      c="b", ls = pattern_list[0])
ax1.plot(well_densities, rmse_h_f_ps3,  c="b", ls = pattern_list[1])
# ax1.plot(well_densities, rmse_h_uf_ps3, c="b", ls = pattern_list[2])

ax1.plot(well_densities, avr_rmse_h,    c="k", ls = pattern_list[0], lw=2)
ax1.plot(well_densities, avr_rmse_h_f,  c="k", ls = pattern_list[1], lw=2)
# ax1.plot(well_densities, avr_rmse_h_uf, c="k", ls = pattern_list[2])

ax1.set_xlim([500,1])
ax1.text(450, 3.2, "(a) Head rmse")

# Age plot - - - - - - - - - - - - - - - - - - - - - - - - -
ax2 = plt.subplot(3, 1, 2)

ax2.plot(well_densities, age_ps1,       c="r", ls = pattern_list[0])
ax2.plot(well_densities, rmse_a_f_ps1,  c="r", ls = pattern_list[1])
# ax2.plot(well_densities, rmse_a_uf_ps1, c="r", ls = pattern_list[2])

ax2.plot(well_densities, age_ps2,       c="g", ls = pattern_list[0])
ax2.plot(well_densities, rmse_a_f_ps2,  c="g", ls = pattern_list[1])
# ax2.plot(well_densities, rmse_a_uf_ps2, c="g", ls = pattern_list[2])

ax2.plot(well_densities, age_ps3,       c="b", ls = pattern_list[0])
ax2.plot(well_densities, rmse_a_f_ps3,  c="b", ls = pattern_list[1])
# ax2.plot(well_densities, rmse_a_uf_ps3, c="b", ls = pattern_list[2])

ax2.plot(well_densities, avr_rmse_a,    c="k", ls = pattern_list[0], lw=2)
ax2.plot(well_densities, avr_rmse_a_f,  c="k", ls = pattern_list[1], lw=2)
# ax2.plot(well_densities, avr_rmse_a_uf, c="k", ls = pattern_list[2])

ax2.set_xlim([500,1])
ax2.text(450, 87, "(b) Age rmse")

# Combined plot - - - - - - - - - - - - - - - - - - - - - - - - -
ax3 = plt.subplot(3, 1, 3)

ax3.plot(well_densities, c_ps1,         c="r", ls = pattern_list[0])
ax3.plot(well_densities, c_ps1_f,       c="r", ls = pattern_list[1])
# ax3.plot(well_densities, c_ps1_uf,      c="r", ls = pattern_list[2])

ax3.plot(well_densities, c_ps2,         c="g", ls = pattern_list[0])
ax3.plot(well_densities, c_ps2_f,       c="g", ls = pattern_list[1])
# ax3.plot(well_densities, c_ps2_uf,      c="g", ls = pattern_list[2])

ax3.plot(well_densities, c_ps3,         c="b", ls = pattern_list[0])
ax3.plot(well_densities, c_ps3_f,       c="b", ls = pattern_list[1])
# ax3.plot(well_densities, c_ps3_uf,      c="b", ls = pattern_list[2])

ax3.plot(well_densities, avr_rmse_c,    c="k", ls = pattern_list[0], lw=2)
ax3.plot(well_densities, avr_rmse_c_f,  c="k", ls = pattern_list[1], lw=2)
# ax3.plot(well_densities, avr_rmse_c_uf, c="k", ls = pattern_list[2])

ax3.set_xlim([500,1])
ax3.text(450, 150, "(c) Combined rmse")
    
plt.savefig(os.path.join(pprfigureDirectory, "RMSE_only100.pdf"), dpi=dpi_value, format=plt_format)     
plt.savefig(os.path.join(pprfigureDirectory, "RMSE_only100"), dpi=dpi_value)     

# RMSE with just the combined RMSE 

#==========================  
plt.figure(figsize=(8, 7))
#==========================  

# PS1 - - - - - - - - - - - - - - - - - - - - - - - - -
ax1 = plt.subplot(3, 1, 1)

ax1.plot(well_densities, c_ps1,      c="r", ls = pattern_list[0], marker="o")
ax1.plot(well_densities, c_ps1_f,  c="r", ls = pattern_list[1], marker="o")
# ax1.plot(well_densities, rmse_h_uf_ps1, c="r", ls = pattern_list[2])

# Legend
# ax1.plot([-5, -4], [-5, -4], c="k", ls = pattern_list[0], marker="o", label="Observation wells")
# ax1.plot([-5, -4], [-5, -4], c="k", ls = pattern_list[1], marker="o", label="Filtered model")

# ax1.legend(bbox_to_anchor=(0.98, 1.3))

ax1.set_xlim([500,1])
ax1.set_ylim([0,175])
ax1.xaxis.set_ticklabels([]) # This should just remove the tick labels

ax1.text(450, 150, "(a) PS1")

# PS2 - - - - - - - - - - - - - - - - - - - - - - - - -
ax2 = plt.subplot(3, 1, 2)

ax2.plot(well_densities, c_ps2,       c="g", ls = pattern_list[0], marker="o")
ax2.plot(well_densities, c_ps2_f,  c="g", ls = pattern_list[1], marker="o")
# ax2.plot(well_densities, rmse_a_uf_ps2, c="g", ls = pattern_list[2])

ax2.set_xlim([500,1])
ax2.set_ylim([0,175])
ax2.xaxis.set_ticklabels([]) # This should just remove the tick labels
plt.ylabel("$RMSE_c$")

ax2.text(450, 150, "(b) PS2")

# PS3 - - - - - - - - - - - - - - - - - - - - - - - - -
ax3 = plt.subplot(3, 1, 3)

ax3.plot(well_densities, c_ps3,         c="b", ls = pattern_list[0], marker="o")
ax3.plot(well_densities, c_ps3_f,       c="b", ls = pattern_list[1], marker="o")
# ax3.plot(well_densities, c_ps3_uf,      c="b", ls = pattern_list[2])

# Legend
ax3.plot([-5, -4], [-5, -4], c="k", ls = pattern_list[0], marker="o", label="Observation wells")
ax3.plot([-5, -4], [-5, -4], c="k", ls = pattern_list[1], marker="o", label="Filtered model")

ax3.set_xlim([500,1])
ax3.set_ylim([0,175])
plt.xlabel("Number of observation wells") # , horizontalalignment='right', x=1.0

ax3.legend() # bbox_to_anchor=(0.38, -0.07)

ax3.text(450, 150, "(c) PS3")

# Average - - - - - - - - - - - - - - - - - - - - - - - - -
# ax4 = plt.subplot(4, 1, 4)

# ax4.plot(well_densities, avr_rmse_c,    c="k", ls = pattern_list[0], lw=1, 
#          marker="o", label="Obs wells")
# ax4.plot(well_densities, avr_rmse_c_f,  c="k", ls = pattern_list[1], lw=1, 
#          marker="o", label="Mdl cells")
# ax3.plot(well_densities, avr_rmse_c_uf, c="k", ls = pattern_list[2])

# ax4.set_xlim([500,1])
# ax4.set_ylim([0,175])

# ax4.text(450, 150, "(d) Average combined rmse")
# ax4.legend(loc=3)
plt.subplots_adjust(top = 0.93, hspace=0.1, wspace=0.01) # , bottom=0.01, 

plt.savefig(os.path.join(pprfigureDirectory, "RMSE_only100_sep.pdf"), dpi=dpi_value,format=plt_format)    
plt.savefig(os.path.join(pprfigureDirectory, "RMSE_only100_sep"), dpi=dpi_value)    

plt.close("all")

##############################################################################
# Table for paper - Single results

key_df_single_table = key_df_single[['param_name', 
                                    'K_posterior', 
                                    'identif',
                                    'distance',
                                    'orientation',
                                    'length']].copy()


fileName = os.path.join(pprfigureDirectory, "key_df_single_table.csv")
key_df_single_table.to_csv(fileName, encoding='utf-8', index=True)

###############################################################################
# PILOT POINTS RMSE VALUES
###############################################################################

pilotpt_mdls = ['18_b', '18_h', '18_n',
                '19_b', '19_h', '19_n',
                '20_b', '20_h', '20_n',
                '21_b', '21_h', '21_n',
                '27_b', '27_h', '27_n',
                '28_b', '28_h', '28_n',
                '31_b', '31_h', '31_n']

list_plt_rmse_h_obs = []
list_plt_rmse_a_obs = []
list_plt_rmse_c_obs = []
list_f_uf_rmse      = []

for mdl in range(len(pilotpt_mdls)):
    datafolder = os.path.join(proj_3_folder, ("UB_" + pilotpt_mdls[mdl] + "_plt"), "Data")
    print(datafolder)
    
    with open (os.path.join(datafolder, 'rmse_h'), 'rb') as fp:
        list_plt_rmse_h_obs.append(pickle.load(fp))
        
    with open (os.path.join(datafolder, 'rmse_a'), 'rb') as fp:
        list_plt_rmse_a_obs.append(pickle.load(fp))
        
    with open (os.path.join(datafolder, 'rmse_c'), 'rb') as fp:
        list_plt_rmse_c_obs.append(pickle.load(fp))
        
    with open (os.path.join(datafolder, 'rmse_f_uf_whole_mdl'), 'rb') as fp:
        list_f_uf_rmse.append(pickle.load(fp))

# Now average out the three obs well configurations for each list
plt_rmse_h_obs_avr = []
plt_rmse_a_obs_avr = []
plt_rmse_c_obs_avr = []
f_h_rmse_avr       = [] # Filtered head
f_a_rmse_avr       = [] # Filtered age

for mdl in range(0, len(pilotpt_mdls), 3):
    plt_rmse_h_obs_avr.append((list_plt_rmse_h_obs[mdl][0] + 
                       list_plt_rmse_h_obs[mdl+1][0] + list_plt_rmse_h_obs[mdl+2][0])/3)
    plt_rmse_a_obs_avr.append((list_plt_rmse_a_obs[mdl][0] + 
                       list_plt_rmse_a_obs[mdl+1][0] + list_plt_rmse_a_obs[mdl+2][0])/3)  
    plt_rmse_c_obs_avr.append((list_plt_rmse_c_obs[mdl][0] + 
                       list_plt_rmse_c_obs[mdl+1][0] + list_plt_rmse_c_obs[mdl+2][0])/3)  
    
    f_h_rmse_avr.append((list_f_uf_rmse[mdl][0] + list_f_uf_rmse[mdl+1][0] + 
                        list_f_uf_rmse[mdl+2][0])/3) 
    f_a_rmse_avr.append((list_f_uf_rmse[mdl][1] + list_f_uf_rmse[mdl+1][1] + 
                        list_f_uf_rmse[mdl+2][1])/3) 
    
    
# Make filtered combined list   
f_c_rmse_avr = 25*np.array(f_h_rmse_avr) + np.array(f_a_rmse_avr)   
    
#+_!+_!+_!+_!+_!+_!+_!+_!+_!+_!+_!+_!+_!+_!+_!+_!+_!+_!+_!+_!+_!+_!+_!+_!+_!+_!
# Re-do plot above but with phantom faults also    



#==========================  
plt.figure(figsize=(4.6, 8))
#==========================

xlimscale = [3.8, 505]
xtext = 4.0
y2 = 255
y_text = 210

# PS1 - - - - - - - - - - - - - - - - - - - - - - - - -
ax1 = plt.subplot(4, 1, 1)

ax1.plot(well_densities, c_ps1,      c="r", ls = pattern_list[0], marker="v")
ax1.plot(well_densities, c_ps1_f,  c="r", ls = pattern_list[1], marker="o")
# ax1.plot(well_densities, rmse_h_uf_ps1, c="r", ls = pattern_list[2])


# Legend
ax1.plot([-5, -4], [-5, -4], c="k", ls = pattern_list[0], marker="v", label="Observation wells")
ax1.plot([-5, -4], [-5, -4], c="k", ls = pattern_list[1], marker="o", label="Filtered model")

ax1.legend() # bbox_to_anchor=(0.38, -0.07)

ax1.set_xlim(xlimscale)
ax1.set_xscale("log")

ax1.set_ylim([0,y2])
ax1.xaxis.set_ticklabels([]) # This should just remove the tick labels
plt.ylabel("$RMSE_c$")

plt.grid()

ax1.text(xtext, y_text, "(a) PS1")

# PS2 - - - - - - - - - - - - - - - - - - - - - - - - -
ax2 = plt.subplot(4, 1, 2)

ax2.plot(well_densities, c_ps2,       c="g", ls = pattern_list[0], marker="v")
ax2.plot(well_densities, c_ps2_f,  c="g", ls = pattern_list[1], marker="o")
# ax2.plot(well_densities, rmse_a_uf_ps2, c="g", ls = pattern_list[2])

ax2.set_xlim(xlimscale)
ax2.set_xscale("log")

ax2.set_ylim([0,y2])
ax2.xaxis.set_ticklabels([]) # This should just remove the tick labels
plt.ylabel("$RMSE_c$")
plt.grid()

ax2.text(xtext, y_text, "(b) PS2")

# PS3 - - - - - - - - - - - - - - - - - - - - - - - - -
ax3 = plt.subplot(4, 1, 3)

ax3.plot(well_densities, c_ps3,         c="b", ls = pattern_list[0], marker="v")
ax3.plot(well_densities, c_ps3_f,       c="b", ls = pattern_list[1], marker="o")
# ax3.plot(well_densities, c_ps3_uf,      c="b", ls = pattern_list[2])


ax3.set_xlim(xlimscale)
ax3.set_xscale("log")

ax3.set_ylim([0,y2])
ax3.xaxis.set_ticklabels([]) # This should just remove the tick labels
plt.ylabel("$RMSE_c$")

plt.grid()

ax3.text(xtext, y_text, "(c) PS3")

#Pilot point data- - - - - - - - - - - - - - - - - - - - - -
ax4 = plt.subplot(4, 1, 4)

ax4.plot(well_densities, plt_rmse_c_obs_avr,    c="0.5", ls = pattern_list[0], lw=1, 
          marker="v")
ax4.plot(well_densities, f_c_rmse_avr,  c="0.5", ls = pattern_list[1], lw=1, 
          marker="o")

ax4.set_xlim(xlimscale)
ax4.set_xscale("log")
plt.ylabel("$RMSE_c$")

ax4.set_ylim([0,y2])
plt.xlabel("Number of observation wells") # , horizontalalignment='right', x=1.0
plt.grid()

ax4.text(xtext, y_text, "(d) Pilot points")
ax4.legend(loc=3)
plt.subplots_adjust(top = 0.93, hspace=0.1, wspace=0.01) # , bottom=0.01, 

plt.tight_layout()
plt.savefig(os.path.join(pprfigureDirectory, "RMSE_only100_sep_plt.pdf"), dpi=dpi_value,format=plt_format)     
plt.savefig(os.path.join(pprfigureDirectory, "RMSE_only100_sep_plt"), dpi=dpi_value)     

### Values of change for paper

# RMSE with obs wells
print("PS1")
c_ps1[0] - c_ps1[-1]
print("PS2")
c_ps2[0] - c_ps2[-1]
print("PS3")
c_ps3[0] - c_ps3[-1]
print("PLT")
plt_rmse_c_obs_avr[0] - plt_rmse_c_obs_avr[-1]


# RMSE with filt model
print("PS1")
c_ps1_f[0] - c_ps1_f[-1]
print("PS2")
c_ps2_f[0] - c_ps2_f[-1]
print("PS3")
c_ps3_f[0] - c_ps3_f[-1]
print("PLT")
f_c_rmse_avr[0] - f_c_rmse_avr[-1]


###############################################################################
# PILOT POINTS RMSE VALUES - all combined figure
###############################################################################  
    
#==========================  
plt.figure(figsize=(12, 6))
#==========================
ax1 = plt.subplot(1,1,1)

y2 = 255
y_text = 200

# PS1 - - - - - - - - - - - - - - - - - - - - - - - - -

ax1.plot(well_densities, c_ps1,      c="r", ls = pattern_list[0], marker="v")
ax1.plot(well_densities, c_ps1_f,  c="r", ls = pattern_list[1], marker="o")

# PS2 - - - - - - - - - - - - - - - - - - - - - - - - -

ax1.plot(well_densities, c_ps2,       c="g", ls = pattern_list[0], marker="v")
ax1.plot(well_densities, c_ps2_f,  c="g", ls = pattern_list[1], marker="o")

# PS3 - - - - - - - - - - - - - - - - - - - - - - - - -

ax1.plot(well_densities, c_ps3,         c="b", ls = pattern_list[0], marker="v")
ax1.plot(well_densities, c_ps3_f,       c="b", ls = pattern_list[1], marker="o")

#Pilot point data- - - - - - - - - - - - - - - - - - - - - -

ax1.plot(well_densities, plt_rmse_c_obs_avr,    c="0.5", ls = pattern_list[0], lw=1, 
          marker="v")
ax1.plot(well_densities, f_c_rmse_avr,  c="0.5", ls = pattern_list[1], lw=1, 
          marker="o")

# Legend
ax1.plot([-5, -4], [-5, -4], c="k", ls = pattern_list[0], marker="o", label="Observation wells")
ax1.plot([-5, -4], [-5, -4], c="k", ls = pattern_list[1], marker="o", label="Filtered model")

ax1.plot([-5, -4], [-5, -4], c="r", ls = pattern_list[0], marker="o", label="PS1")
ax1.plot([-5, -4], [-5, -4], c="g", ls = pattern_list[0], marker="o", label="PS2")
ax1.plot([-5, -4], [-5, -4], c="b", ls = pattern_list[0], marker="o", label="PS3")
ax1.plot([-5, -4], [-5, -4], c="0.5", ls = pattern_list[0], marker="o", label="Pilot Points")

ax1.set_xlim([505,4])
ax1.set_ylim([0,y2])

ax1.set_xscale("log")

ax1.legend() # bbox_to_anchor=(0.38, -0.07)

plt.xlabel("Number of observation wells") # , horizontalalignment='right', x=1.0

  
plt.savefig(os.path.join(pprfigureDirectory, "RMSE_only100_plt.pdf"), dpi=dpi_value,format=plt_format)     
plt.savefig(os.path.join(pprfigureDirectory, "RMSE_only100_plt"), dpi=dpi_value)     

###############################################################################
# PILOT POINTS RMSE VALUES - all combined figure
###############################################################################  
   
avr_phstr_rmse_c_obs = (np.array(c_ps1) + np.array(c_ps2) + np.array(c_ps2))/3 
avr_phstr_rmse_c_fil = (np.array(c_ps1_f) + np.array(c_ps2_f) + np.array(c_ps3_f))/3 
    
pltcolour = "teal"
#==========================  
plt.figure(figsize=(12, 6))
#==========================
ax1 = plt.subplot(1,1,1)

y2 = 255
y_text = 200

# PS-AVERAGE - - - - - - - - - - - - - - - - - - - - - - - - -

ax1.plot(well_densities, avr_phstr_rmse_c_obs,  c="k", ls = pattern_list[0], 
         marker="o", label="Phantom structures - observation wells")
ax1.plot(well_densities, avr_phstr_rmse_c_fil,  c="k", ls = pattern_list[1], 
         marker="o", label="Phantom structures - filtered model")

#Pilot point data- - - - - - - - - - - - - - - - - - - - - -

ax1.plot(well_densities, plt_rmse_c_obs_avr,    c=pltcolour, ls = pattern_list[0], lw=1, 
          marker="o", label="Pilot points - observation wells")
ax1.plot(well_densities, f_c_rmse_avr,  c=pltcolour, ls = pattern_list[1], lw=1, 
          marker="o", label="Pilot points - filtered model")


ax1.set_xlim([525,4])
ax1.set_ylim([0,y2])

ax1.set_xscale("log")

ax1.legend(loc=2) # bbox_to_anchor=(0.38, -0.07)

plt.xlabel("Number of observation wells") # , horizontalalignment='right', x=1.0

  
plt.savefig(os.path.join(pprfigureDirectory, "RMSE_avr_plt.pdf"), dpi=dpi_value, format=plt_format)     
plt.savefig(os.path.join(pprfigureDirectory, "RMSE_avr_plt"), dpi=dpi_value)     

plt.close('all')
    
    
    
    
###############################################################################
###############################################################################
# FIGURE FIFTEEN: PILOT POINT FIGURE
###############################################################################
###############################################################################

# First I need to pull in all of the arrays

pltnames_names_list = ["UB_18", "UB_19", "UB_20", "UB_21", "UB_27", "UB_28", "UB_31"]

plt_names_list = ["b", "h", "n"]

pltHkDict = {}

hPltDict = {}
aPltDict = {}
cPltDict = {}

for mi, mdl in enumerate(pltnames_names_list):
    plt.figure(figsize=(5, 10))
    plt.suptitle("Number of wells: %s" %well_densities[mi])
    for plt_optn in range(len(plt_names_list)):
        plt_modelname = mdl + "_" + plt_names_list[plt_optn] + "_plt"

        this_model_folder = os.path.join(r'C:\workspace\Proj3_PilbaraBarriers', plt_modelname)
        print(this_model_folder)
        
        this_model_folder_data = os.path.join(this_model_folder, "Data")
        print(this_model_folder_data)
        

        with open (os.path.join(this_model_folder_data, 'rmse_h'), 'rb') as fp:
            rmse_plt_h = pickle.load(fp)
        hPltDict[plt_modelname] = rmse_plt_h[0]

        with open (os.path.join(this_model_folder_data, 'rmse_a'), 'rb') as fp:
            rmse_plt_a = pickle.load(fp)
        aPltDict[plt_modelname] = rmse_plt_a[0]
            
        with open (os.path.join(this_model_folder_data, 'rmse_c'), 'rb') as fp:
            rmse_plt_c = pickle.load(fp)
        cPltDict[plt_modelname] = rmse_plt_c[0]
              
        final_pest_files_folder = os.path.join(this_model_folder, "final_from_pest")
        hk_file = os.path.join(final_pest_files_folder, "hk_Layer_1.ref")
        hk_final_array = np.loadtxt(hk_file)
        log_interp_pest_hk = np.log10(hk_final_array)
        
        # Save this log array
        pltHkDict[plt_modelname] = log_interp_pest_hk
        
        # Plot this array - just to check them out
        plt.subplot(3, 1, plt_optn+1)

        plt.imshow(np.flipud(log_interp_pest_hk[ :, :]), extent=extent, cmap='terrain',
                 vmin = -2.5, vmax = 0) 

        cbr = plt.colorbar()
        cbr.ax.tick_params(labelsize=14) 
        cbr.set_label('LOG Estimated K (m/d)', fontsize = 14)

        plt_title = "hkpp_%s" %well_densities[mi]
        plt.savefig(os.path.join(pprfigureDirectory, plt_title))


# I need to do the average here !!

all_rmse_values_for_plt = ["64.5", "62.9", "57.4", "46.3", "42.6", "22.8", "14.2"]

names_list = ["(a) 500 wells", "(b) 250 wells", "(c) 100 wells", "(d) 50 wells",
              "(e) 25 wells", "(f) 10 wells", "(g) 5 wells"]

avr_pp_rasters = []

avr_h_plt_rmse = []
avr_a_plt_rmse = []
avr_c_plt_rmse = []

#==================================
fig = plt.figure(figsize=(7, 7))
#==================================

for mi, mdl in enumerate(pltnames_names_list):
    plt_modelname1 = mdl + "_b_plt"
    plt_modelname2 = mdl + "_h_plt"
    plt_modelname3 = mdl + "_n_plt"
    
    raster1 = pltHkDict[plt_modelname1]
    raster2 = pltHkDict[plt_modelname2]
    raster3 = pltHkDict[plt_modelname3]
    
    avr_raster = (raster1 + raster2 + raster3)/3
    avr_pp_rasters.append(avr_raster)
    
    # Plot the average pp raster
    ax = plt.subplot(4, 2, mi+1)
    img1 = plt.imshow(np.flipud(avr_raster[ :, :]), extent=extent, 
                      cmap='terrain', vmin = -2.5, vmax = 0) 
    
    # Plot the real fault location
    for i in range(len(re_fa_rows_list)):
        ax.plot(re_fa_cols_list_m[i], re_fa_rows_list_m[i], lw = 3, 
                 color = real_plot_c, ls=real_plot_ls)
    
    plt.text(1500, 8000, names_list[mi])
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    

    if mi == 1 or mi == 3:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
    elif mi == 0 or mi == 2 or mi == 4:
        ax.set_xticklabels([])
        plt.ylabel("$y$ [m]")
        
    elif mi == 5:
        ax.set_yticklabels([])
        plt.xlabel("$x$ [m]")
        
    else:
        plt.xlabel("$x$ [m]")
        plt.ylabel("$y$ [m]")
    
    # Get the RMSE values and take the averages
    avr_h_plt_rmse.append((hPltDict[plt_modelname1] + hPltDict[plt_modelname1] 
                           + hPltDict[plt_modelname1])/3)
    avr_a_plt_rmse.append((aPltDict[plt_modelname1] + aPltDict[plt_modelname1] 
                           + aPltDict[plt_modelname1])/3)
    avr_c_plt_rmse.append((cPltDict[plt_modelname1] + cPltDict[plt_modelname1] 
                           + cPltDict[plt_modelname1])/3)
        
plt.subplots_adjust(top = 0.90, hspace=0.01, wspace=0.05) # , bottom=0.01, 

# plt.subplots_adjust(bottom=0.21)

cbaxes1 = fig.add_axes([0.56, 0.18, 0.32, 0.03]) 
cb1 = plt.colorbar(img1, cax = cbaxes1, orientation="horizontal", format='%1.1f')  


cb1.set_label('Log $K$ [m/d]')




plt.savefig(os.path.join(pprfigureDirectory, "hkpp_avr_all.pdf"), dpi=dpi_value, format=plt_format)     
plt.savefig(os.path.join(pprfigureDirectory, "hkpp_avr_all"), dpi=dpi_value)     


# Get max and min values and change vmin, vmax above if necessary

max_list = []
min_list = []
for i, ras in enumerate(avr_pp_rasters):
    max_list.append(np.max(ras))
    min_list.append(np.min(ras))
    
max(max_list) #0.033
min(min_list) #-2.005

# Download the RMSE values for each of the pilot point simulations    
    
    
plt.close('all')


###############################################################################
###############################################################################
# FIGURE SIXTEEN: ALL OBS WELL LOCATIONS
###############################################################################
###############################################################################
all_obs_rc_df = []

all_mdls = []

for ovrn in overall_names_list:
    for cnfg in ["b", "h", "n"]: # Don't need all bc some are just same obs wells
        all_mdls.append("%s_%s" %(ovrn, cnfg))        
                
for mdl in all_mdls:
    path = os.path.join(proj_3_folder, mdl, "Data", "obs_rowcol.csv")
    all_obs_rc_df.append(pd.read_csv(path))
    
for n_df in range(len(all_obs_rc_df)):
    all_obs_rc_df[n_df]["obs_well_x"] = (all_obs_rc_df[n_df]["c_wellcol"]*param_dict["delr"] 
                                         + 0.5*param_dict["delr"])
    all_obs_rc_df[n_df]["obs_well_y"] = (all_obs_rc_df[n_df]["b_wellrow"]*param_dict["delc"] 
                                         + 0.5*param_dict["delc"])
    
captn_wells_obs = []

for captn in range(len(captn_wells)):
    lst = [captn_wells[captn]]*3
    captn_wells_obs = captn_wells_obs + lst


#==========================  
fig = plt.figure(figsize=(8, 11))
#==========================  

pltcount = 1
symbl = "o"
clr = "dodgerblue"
sz = 4

for n_df in range(len(all_obs_rc_df)):
    
    # Phantom fault config 1
    ax = plt.subplot(7, 3, pltcount)
 
    plt.plot(all_obs_rc_df[n_df]["obs_well_x"], 
             all_obs_rc_df[n_df]["obs_well_y"], ls="", 
             marker=symbl, c=clr, markersize= sz, label="Observation well")

    axes = plt.gca()
    axes.set_xlim([0, param_dict["Lx"]])                
    axes.set_ylim([0, param_dict["Ly"]])   
         
    # Now add the real fault
    for i in range(len(re_fa_rows_list)):
        plt.plot(re_fa_cols_list_m[i], re_fa_rows_list_m[i], lw = 4, 
                 color = real_plot_c, ls=real_plot_ls, label="Real model structure")
    
    if pltcount == 19 or pltcount == 20 or pltcount == 21:
        plt.xticks(fontsize=12, rotation=45)
        plt.xticks([10000, 20000]) # , ["", "2 km"]
        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    else:
        ax.axes.get_xaxis().set_ticks([])
    
    plt.text(xcaptn, ycaptn, "(%s)" % string.ascii_lowercase[pltcount-1],
             fontweight="bold") 
    ax_positions.append(axes.get_position())
    
    if (pltcount == 1 or pltcount == 4 or pltcount == 7 or pltcount == 10 
        or pltcount == 13 or pltcount == 16 or pltcount == 19):
        plt.yticks(fontsize=12, rotation=45)
        # plt.yticks([0, 5000, 10000], ["", "", "2 km"])
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    
        # Label the well number
        plt.ylabel(captn_wells_obs[n_df], fontweight="bold", labelpad=-2.2) # On the side
    else:
        ax.axes.get_yaxis().set_ticks([])

    ##### End of subplot
    pltcount = pltcount+1 
          
# Adding common x and y labels
fig.text(0.5, 0.056, '$x$ [m]', ha='center', va='center')
fig.text(0.013, 0.52, '$y$ [m]', ha='center', va='center', rotation='vertical')

# ax = plt.subplot(7, 3, 3)
# pos1 = ax.get_position() 
# x1t = 0.125 +0.2279411764705883*0.5#+ 0.5*0.35294117647058826
# x2t = 0.3985294117647059 + 0.2279411764705883*0.5 #+ 0.5*0.6264705882352941
# x3t = 0.6720588235294118 + 0.2279411764705883*0.5#+ 0.5*0.9000000000000001
# Adding phantom fault labels
fig.text(x1t, 0.94, 'OW1', fontweight="bold", ha='center', va='center')
fig.text(x2t, 0.94, 'OW2', fontweight="bold", ha='center', va='center')
fig.text(x3t, 0.94, 'OW3', fontweight="bold", ha='center', va='center')

# Adding number of wells labels
plt.subplots_adjust(top=0.93, hspace=0.1, wspace=0.1, bottom=0.12) # , 

ax.legend(bbox_to_anchor=(1.15, -0.50))


plt.savefig(os.path.join(pprfigureDirectory, "ObsLocsAll.pdf"), dpi=dpi_value,format=plt_format)  
plt.savefig(os.path.join(pprfigureDirectory, "ObsLocsAll"), dpi=dpi_value)  


###############################################################################
###############################################################################
# FIGURE ONE: LOCATION OF PHANTOM FAULTS - BUT IN UNITS SAME AS MODEL
###############################################################################
###############################################################################
   
def plot_phantom_faults_m(ph_fa_cols_list, ph_fa_rows_list, 
                        re_fa_cols_list, re_fa_rows_list):
    for i in range(len(ph_fa_rows_list)):
        plt.plot(ph_fa_cols_list[i], ph_fa_rows_list[i], lw = 3, 
                 color = 'lightsteelblue', label = "_nolegend_")
    # plt.xlabel("$x axis$", fontsize=14)
    # plt.ylabel("$y axis$", fontsize=14)
      
    axes = plt.gca()
    axes.set_xlim([0, param_dict["Lx"]])                
    axes.set_ylim([0, param_dict["Ly"]])                
    
    # Now add the real fault
    for i in range(len(re_fa_rows_list)):
        plt.plot(re_fa_cols_list[i], re_fa_rows_list[i], lw = 4, 
                 color = real_plot_c, ls = real_plot_ls, label = "Real model structure")
        
#==========================    
plt.figure(figsize=(6, 10))
#==========================    

plt.subplot(3, 1, 1)
plot_phantom_faults_m(ph_fa_cols_list1_m, ph_fa_rows_list1_m, 
                    re_fa_cols_list_m, re_fa_rows_list_m)
axes1 = plt.gca()
axes1.axes.get_xaxis().set_visible(False)
axes1.text(0.5, 10500, "(a) PS1")
axes1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

axes1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.ylabel('$y$ [m]')

plt.subplot(3, 1, 2)
plot_phantom_faults_m(ph_fa_cols_list2_m, ph_fa_rows_list2_m, 
                    re_fa_cols_list_m, re_fa_rows_list_m)
axes2 = plt.gca()
axes2.axes.get_xaxis().set_visible(False)
axes2.text(0.5, 10500, "(b) PS2")
axes2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

plt.ylabel('$y$ [m]')

plt.subplot(3, 1, 3)
# For legend
plt.plot([-10, -11], [-12, -13], lw = 4, 
                 color = 'lightsteelblue', label = "Phantom structure")


plot_phantom_faults_m(ph_fa_cols_list3_m, ph_fa_rows_list3_m, 
                    re_fa_cols_list_m, re_fa_rows_list_m)

axes3 = plt.gca()
axes3.text(0.5, 10500, "(c) PS3")
axes3.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
axes3.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')

axes3.legend(bbox_to_anchor=(1.1, -0.2))

plt.tight_layout()

plt.savefig(os.path.join(pprfigureDirectory, "AllPhStr.pdf"), dpi=dpi_value, format=plt_format)
plt.savefig(os.path.join(pprfigureDirectory, "AllPhStr"), dpi=dpi_value)

###############################################################################
###############################################################################
# FIGURE: HEAD AND AGE DIFFERENCES, CHANGED CONTOURS
###############################################################################
###############################################################################

#==================================
fig = plt.figure(figsize = (7, 7))
#==================================

# Plot steady-state hydraulic heads

ax1 = fig.add_subplot(2, 1, 1, aspect='equal')

img1 = ax1.imshow(np.flipud(head_diff_nobarr[0, :, :]), extent=extent, cmap='PiYG',
                  vmin=-17, vmax=17)

# Plot the real fault location
for i in range(len(re_fa_rows_list)):
        ax1.plot(re_fa_cols_list_m[i], re_fa_rows_list_m[i], lw = 4, 
                 color = real_plot_c, ls=real_plot_ls)

axes1 = plt.gca()
plt.ylabel("$y$ [m]")
axes1.axes.get_xaxis().set_visible(False)
axes1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.text(1000, 8500, "(a) Hydraulic head difference")

# # # # # # # # # # # --------------------------------------------------------
ax2 = fig.add_subplot(2, 1, 2, aspect='equal')

img2 = ax2.imshow(np.flipud(age_diff_nobarr[0, :, :]), extent=extent, cmap="PuOr",
                  vmin = -310, vmax = 310)

# Plot the real fault location
for i in range(len(re_fa_rows_list)):
        ax2.plot(re_fa_cols_list_m[i], re_fa_rows_list_m[i], lw = 4, 
                 color = real_plot_c, ls=real_plot_ls)
        

axes2 = plt.gca()
plt.xlabel("$x$ [m]")
plt.ylabel("$y$ [m]")
axes2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
axes2.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.text(1000, 8500, "(b) Groundwater age difference")

# Colourbars
# ---------------------------- #
plt.subplots_adjust(bottom=0.21, hspace=0.05)

cbaxes1 = fig.add_axes([0.2, 0.1, 0.3, 0.03]) # [0.133, 0.1, 0.32, 0.03] 
cb1 = plt.colorbar(img1, cax = cbaxes1, orientation="horizontal", format='%1.0f')  
cb1.set_label('Head difference [m]')
tick_locator = ticker.MaxNLocator(nbins = 6)
cb1.locator = tick_locator
cb1.update_ticks()

cbaxes2 = fig.add_axes([0.55, 0.1, 0.3, 0.03]) # [0.547, 0.1, 0.32, 0.03] 
cb2 = plt.colorbar(img2, cax = cbaxes2, orientation="horizontal", format='%1.0f')  
cb2.set_label('Age difference [yrs]')
tick_locator = ticker.MaxNLocator(nbins = 6)
cb2.locator = tick_locator
cb2.update_ticks()

plt.savefig(os.path.join(pprfigureDirectory, "HeadAndAgeDiff2.pdf"), dpi=dpi_value,format=plt_format)  
plt.savefig(os.path.join(pprfigureDirectory, "HeadAndAgeDiff2"), dpi=dpi_value)  


plt.close("all")
