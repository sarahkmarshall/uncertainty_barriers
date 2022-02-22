# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:51:07 2022

@author: mars0451
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

pprfigureDirectory = r"C:\workspace\Proj3_PilbaraBarriers\Paper3Figs_march2022"

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
head_cmap = 'coolwarm'
age_cmap = 'jet'
ident_colourmap = "winter"
metr_colourmap  = "nipy_spectral"
n_wells_cmap    = "gnuplot"

levels_h = np.linspace(0, 400, 51)
levels_a = np.linspace(0, 1400, 15)
extent = (100/2., 20000 - 100/2., 100/2., 10000 - 100/2.)

real_plot_c = "k"
real_plot_ls = (0, (1,.6)) #":"
xcaptn = 10*100
ycaptn = 80*100
ax_positions = []
alphaval = 0.7
captn_wells = ["500 wells", "250 wells", "100 wells", "50 wells",
               "25 wells", "10 wells", "5 wells"]

x1t = 0.125 +0.2279411764705883*0.5#+ 0.5*0.35294117647058826
x2t = 0.3985294117647059 + 0.2279411764705883*0.5 #+ 0.5*0.6264705882352941
x3t = 0.6720588235294118 + 0.2279411764705883*0.5#+ 0.5*0.9000000000000001

lw_ps = 1.5


c_list = [ph_1_c, ph_1_c, ph_2_c, ph_2_c, ph_3_c, ph_3_c, 
          ph_1_c, ph_1_c, ph_2_c, ph_2_c, ph_3_c, ph_3_c, 
          ph_1_c, ph_1_c, ph_2_c, ph_2_c, ph_3_c, ph_3_c]

c_list_reg = c_list[::2]

#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
#------------------------------------------------------------------------------    
# SETTING UP DATA FOR PLOTTING
# -----------------------------------------------------------------------------
#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~


# Setting up phantom structure locations # 

# Using UB 29 for params (500 wells) but it shouldn't matter they are all the same
# Setting up parameters and data for the plotting 
dataDirectory1 = r'C:\workspace\Proj3_PilbaraBarriers\UB_29_b\Data'
dataDirectory2 = r'C:\workspace\Proj3_PilbaraBarriers\UB_29_d\Data'
dataDirectory3 = r'C:\workspace\Proj3_PilbaraBarriers\UB_29_f\Data'

# Real Barrier
re_fault_coordinates = np.load(os.path.join(dataDirectory1, "re_fault_coords.npy"))

# Real Barrier
re_fault_coordinates = np.load(os.path.join(dataDirectory1, "re_fault_coords.npy"))

# Set up real barriers at different sizes and locations
re_fa_rows_list, re_fa_cols_list, re_fa_length_list = (functions_proj_3.barriers.makeFaultRowCols(
                            re_fault_coordinates, 
                            param_dict["nrow"], 
                            param_dict["ncol"]))




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




## ----------------

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

#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
#------------------------------------------------------------------------------    
# PLOTTING
# -----------------------------------------------------------------------------
#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

###############################################################################
# FIGURE ONE: MODEL SET UP
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

#plt.savefig(pprfigureDirectory + "\model_set_up_notext.pdf", format="pdf", dpi=dpi_value)
plt.savefig(pprfigureDirectory + "\model_set_up_notext", dpi=dpi_value)


###############################################################################
###############################################################################
# FIGURE TWO: PLOTTING ORIGINAL K AND AGE ARRAY
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
#plt.savefig(os.path.join(pprfigureDirectory, "HeadAndAgeNoBarr.pdf"), dpi=dpi_value,format=plt_format)  

# Get mean values
mean_h = np.mean(original_head_file[:, :])
mean_a = np.mean(original_age_file[:, :])

# Get max values
max_h = np.max(original_head_file[:, :])
max_a = np.max(original_age_file[:, :])

###############################################################################
###############################################################################
# FIGURE THREE: ALL OBS WELL LOCATIONS
###############################################################################
###############################################################################



all_mdls = []

for ovrn in overall_names_list:
    for cnfg in ["b", "h", "n"]: # Don't need all bc some are just same obs wells
        all_mdls.append("%s_%s" %(ovrn, cnfg))        
        
all_obs_pest_obs_file = []
                
for mdl in all_mdls:
    path = os.path.join(proj_3_folder, mdl)
    pest_obs_filename_txt = os.path.join(path, (mdl + "_reg.rei"))
    pest_obs_file = pd.read_csv(pest_obs_filename_txt, skiprows=6,
                        delim_whitespace=True, encoding='utf-8-sig') 
    pest_obs_file.index = pest_obs_file.Name
    
    # Add the row and column information to the head and age observations
    pest_obs_file["Row"] = np.zeros(len(pest_obs_file.index))
    pest_obs_file["Col"] = np.zeros(len(pest_obs_file.index))
    
    for idx in pest_obs_file.index:
        if (pest_obs_file.loc[idx, "Group"] == "head" or 
            pest_obs_file.loc[idx, "Group"] == "age"):
            pest_obs_file.loc[idx, "Row"] = int(idx[1:4])
            pest_obs_file.loc[idx, "Col"] = int(idx[5:8])
        else:
            pass
    
    # Change row and column information to be same units as model
    
    pest_obs_file["Row100"] = np.zeros(len(pest_obs_file.index))
    pest_obs_file["Col100"] = np.zeros(len(pest_obs_file.index))
    
    for idx in pest_obs_file.index:
        if (pest_obs_file.loc[idx, "Group"] == "head" or 
            pest_obs_file.loc[idx, "Group"] == "age"):
            pest_obs_file.loc[idx, "Row100"] = pest_obs_file.loc[idx, "Row"]*100
            pest_obs_file.loc[idx, "Col100"] = pest_obs_file.loc[idx, "Col"]*100
        else:
            pass
        
    all_obs_pest_obs_file.append(pest_obs_file)
    
    
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


n_df = 0  

for n_df in range(len(all_mdls)):
    
    # Phantom fault config 1
    ax = plt.subplot(7, 3, pltcount)
    
    pest_obs_file = all_obs_pest_obs_file[n_df]
    pest_obs_file_head = pest_obs_file[pest_obs_file["Group"]=="head"]
    
    img = ax.scatter(pest_obs_file_head.Col100, pest_obs_file_head.Row100,
               c=pest_obs_file_head.Measured,
               s=sz, cmap="coolwarm_r")
    
  

    axes = plt.gca()
    axes.set_xlim([0, param_dict["Lx"]])                
    axes.set_ylim([0, param_dict["Ly"]])   
    
         
    #Now add the real fault
    # for i in range(len(re_fa_rows_list)):
    #     plt.plot(re_fa_cols_list_m[i], re_fa_rows_list_m[i], lw = 4, 
    #               color = real_plot_c, ls=real_plot_ls, alpha=0.3,
    #               label="Real model structure")
    
    #Change axes labels
    
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

# Adding colour-bars

plt.subplots_adjust(bottom=0.16)

#cbaxes1 = fig.add_axes([0.2, 0.05, 0.6, 0.02])  # This is in the middle
#cbaxes1 = fig.add_axes([0.2, 0.033, 0.6, 0.01])  # This is in the middle but skinny
cbaxes1 = fig.add_axes([0.66, 0.041, 0.24, 0.02]) # This is on the side
cb1 = plt.colorbar(img, cax = cbaxes1, orientation="horizontal")  
cb1.set_label('Hydraulic head [m]', fontsize=12)
cb1.ax.tick_params(labelsize=12)

tick_locator = ticker.MaxNLocator(nbins = 5)
cb1.locator = tick_locator
cb1.update_ticks()

          
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


#plt.savefig(os.path.join(pprfigureDirectory, "ObsLocsAll.pdf"), dpi=dpi_value,format=plt_format)  
plt.savefig(os.path.join(pprfigureDirectory, "ObsLocsAll_head"), dpi=dpi_value)  

###############################################################################
###############################################################################
# FIGURE FOUR: LOCATION OF PHANTOM FAULTS - BUT IN UNITS SAME AS MODEL
###############################################################################
###############################################################################

ph_f_col = "0.95"# 'lightsteelblue'

def plot_phantom_faults_m(ph_fa_cols_list, ph_fa_rows_list, 
                        re_fa_cols_list, re_fa_rows_list):
    for i in range(len(ph_fa_rows_list)):
        plt.plot(ph_fa_cols_list[i], ph_fa_rows_list[i], lw = lw_ps , 
                 color = ph_f_col, label = "_nolegend_")
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
plt.figure(figsize=(5, 8))
#==========================    

plt.subplot(3, 1, 1)
plot_phantom_faults_m(ph_fa_cols_list1_m, ph_fa_rows_list1_m, 
                    re_fa_cols_list_m, re_fa_rows_list_m)
axes1 = plt.gca()
axes1.axes.get_xaxis().set_visible(False)
axes1.text(250, 8500, "(a) PS1")
axes1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

axes1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.ylabel('$y$ [m]')

plt.subplot(3, 1, 2)
plot_phantom_faults_m(ph_fa_cols_list2_m, ph_fa_rows_list2_m, 
                    re_fa_cols_list_m, re_fa_rows_list_m)
axes2 = plt.gca()
axes2.axes.get_xaxis().set_visible(False)
axes2.text(250, 8500, "(b) PS2")
axes2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

plt.ylabel('$y$ [m]')

plt.subplot(3, 1, 3)
# For legend
plt.plot([-10, -11], [-12, -13], lw = lw_ps , 
                 color = ph_f_col, label = "Phantom structure")


plot_phantom_faults_m(ph_fa_cols_list3_m, ph_fa_rows_list3_m, 
                    re_fa_cols_list_m, re_fa_rows_list_m)

axes3 = plt.gca()
#axes3.text(0.5, 10500, "(c) PS3")
axes3.text(250, 8500, "(c) PS3")

axes3.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
axes3.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')



plt.tight_layout()
# Adding number of wells labels
plt.subplots_adjust(top=0.93, hspace=0.1, wspace=0.1, bottom=0.15) # , 

axes3.legend(bbox_to_anchor=(1.1, -0.22))

#plt.savefig(os.path.join(pprfigureDirectory, "AllPhStr.pdf"), dpi=dpi_value, format=plt_format)
plt.savefig(os.path.join(pprfigureDirectory, "AllPhStr"), dpi=dpi_value)

###############################################################################
###############################################################################
# FIGURE FIVE: PLOTTING K AND IDENTIFIABILITY OF SINGLE EXAMPLE KEY FAULTS
# WITH MODEL SIZES
###############################################################################
###############################################################################

### SET UP SINGLE EXAMPLE INDEXES### 

sing_wellno = 3 # UB_33 = 50 wells
sing_cs = 0 # b = obs well and ps 1


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

phantom_df_single =    all_well_dens_ph_dfs[sing_wellno][sing_cs].copy()    # UB_33_b
phantom_df_single.index = phantom_df.param_name
key_df_single =        all_well_dens_key_dfs[sing_wellno][sing_cs].copy()   # UB_33_b
key_df_single.index = key_df_single.param_name



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
             lw = lw_ps, color = K_col)
        
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

axes.text(8000, 4100, "1")  # Old 1
axes.text(6200, 1800, "2")  # Old 3
axes.text(10350, 5600, "3") # Old 2
axes.text(9900, 7000, "4") # Wasn't in old


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

#plt.savefig(os.path.join(pprfigureDirectory, "SingleKandIdent_m.pdf"), dpi=dpi_value,format=plt_format)  
plt.savefig(os.path.join(pprfigureDirectory, "SingleKandIdent_m"), dpi=dpi_value)  


###############################################################################
###############################################################################
# FIGURE SIX: HEAD AND AGE DIFFERENCES, CHANGED CONTOURS
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
plt.figure()
loghkarray_single = np.log10(hk_array_single)
img = plt.imshow(np.flipud(loghkarray_single[0, :, :]), extent=extent, cmap='terrain')
plt.colorbar(img)


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

#plt.savefig(os.path.join(pprfigureDirectory, "HeadAndAgeDiff2.pdf"), dpi=dpi_value,format=plt_format)  
plt.savefig(os.path.join(pprfigureDirectory, "HeadAndAgeDiff2"), dpi=dpi_value)  

###############################################################################
###############################################################################
# FIGURE NINE: VARIATIONS - WITH ERROR AND WITH HEAD ONLY
###############################################################################
###############################################################################

# Download the data

# EXAMPLE WITH ERROR = UB_62_b
dataDirectory_s1 = r'C:\workspace\Proj3_PilbaraBarriers\UB_62_b\Data'
phantom_df_f_s1 = os.path.join(dataDirectory_s1, "phantom_df.csv")
phantom_df_s1 = pd.read_csv(phantom_df_f_s1)

key_df_f_s1 = os.path.join(dataDirectory_s1, "key_dataframe.csv") # or key_df for all faults, key is just those identified >0.8 and low K
key_df_s1 = pd.read_csv(key_df_f_s1)

key_df_s1["length_diff"] = abs(length_real_barrier - key_df_s1["length"])

n_key_f_s1 = len(key_df_s1.index)

phantom_df_s1.index = phantom_df_s1.param_name
phantom_df_s1.head(10)

key_df_s1.index = key_df_s1.param_name
key_df_s1.head(10)


# EXAMPLE WITH HEAD ONLY = UB_63_b

dataDirectory_s2 = r'C:\workspace\Proj3_PilbaraBarriers\UB_63_b\Data'
phantom_df_f_s2 = os.path.join(dataDirectory_s2, "phantom_df.csv")
phantom_df_s2 = pd.read_csv(phantom_df_f_s2)

key_df_f_s2 = os.path.join(dataDirectory_s2, "key_dataframe.csv") # or key_df for all faults, key is just those identified >0.8 and low K
key_df_s2 = pd.read_csv(key_df_f_s2)

key_df_s2["length_diff"] = abs(length_real_barrier - key_df_s2["length"])

n_key_f_s2 = len(key_df_s2.index)

phantom_df_s2.index = phantom_df_s2.param_name
phantom_df_s2.head(10)

key_df_s2.index = key_df_s2.param_name
key_df_s2.head(10)

####### STARTING PLOTTING =====================================================

### Set up the phantom and key dataframes for the single example

phantom_df_single =    phantom_df_s1
key_df_single =        key_df_s1

###################################
# Table for paper - Single results

key_df_single_table_s1 = key_df_single[['param_name', 
                                    'K_posterior', 
                                    'identif',
                                    'distance',
                                    'orientation',
                                    'length']].copy()


fileName = os.path.join(pprfigureDirectory, "key_df_single_table_s1.csv")
key_df_single_table_s1.to_csv(fileName, encoding='utf-8', index=True)

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
fig = plt.figure(figsize=(7, 7))    
#==========================  

ax  = plt.subplot(2,1,1)

for i in range(len(ph_fa_rows_list)):
    
    # First just plot the faults that are not key
    if (phantom_df_single.index[i] in key_df_single.index) == True:
        pass
                            
    else:
        K_col = "0.95"
        
        ax.plot(all_cols_dict[phantom_df_single.index[i]], all_rows_dict[phantom_df_single.index[i]],  
             lw = lw_ps, color = K_col)
        
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

axes.text(8000, 4100, "1")  # Old 1
axes.text(6200, 1800, "2")  # Old 3
axes.text(10350, 5600, "3") # Old 2
axes.text(9900, 7000, "4") # Wasn't in old

axes.axes.get_xaxis().set_visible(False)
plt.text(500, 9000, "(a) Head and age data (with error)")

############## s2 -------------------------------------------------------------


phantom_df_single =    phantom_df_s2
key_df_single =        key_df_s2

# Table for paper - Single results

key_df_single_table_s2 = key_df_single[['param_name', 
                                    'K_posterior', 
                                    'identif',
                                    'distance',
                                    'orientation',
                                    'length']].copy()


fileName = os.path.join(pprfigureDirectory, "key_df_single_table_s2.csv")
key_df_single_table_s2.to_csv(fileName, encoding='utf-8', index=True)

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


ax  = plt.subplot(2,1,2)

for i in range(len(ph_fa_rows_list)):
    
    # First just plot the faults that are not key
    if (phantom_df_single.index[i] in key_df_single.index) == True:
        pass
                            
    else:
        K_col = "0.95"
        
        ax.plot(all_cols_dict[phantom_df_single.index[i]], all_rows_dict[phantom_df_single.index[i]],  
             lw = lw_ps, color = K_col)
        
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

axes.text(8000, 4100, "1")  # Old 1
axes.text(6200, 1800, "2")  # Old 3
axes.text(10350, 5600, "3") # Old 2
axes.text(9900, 7000, "4") # Wasn't in old

plt.text(500, 9000, "(a) Head data only (no error)")

#---------------------------------------------------------------

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
fig.text(0.02, 0.53, '$y$ [m]', ha='center', va='center', rotation='vertical')

#plt.savefig(os.path.join(pprfigureDirectory, "SingleKandIdent_m.pdf"), dpi=dpi_value,format=plt_format)  
plt.savefig(os.path.join(pprfigureDirectory, "SingleKandIdent_m_s1s2"), dpi=dpi_value)  


###############################################################################
###############################################################################
# FIGURE NINE A: DIFFERENT CASE STUDIES
###############################################################################
###############################################################################


# Setting up Phantom structure parameters and data for the plotting  ----------------------------

# Example 1 - in line with the flow regime
re_fault_coordinates_e1 = np.load(os.path.join(
    'C:\\workspace\\Proj3_PilbaraBarriers\\UB_54\\re_fault_coords.npy'))

# Set up real barriers at different sizes and locations
re_fa_rows_list_e1, re_fa_cols_list_e1, re_fa_length_list_e1 = (functions_proj_3.barriers.makeFaultRowCols(
                            re_fault_coordinates_e1, 
                            param_dict["nrow"], 
                            param_dict["ncol"]))

re_fa_rows_list_e1_m = []
re_fa_cols_list_e1_m = []

for fault in range(len(re_fa_cols_list_e1)):
    faultcols_new = []
    faultrows_new = []
    for cell in range(len(re_fa_cols_list_e1[fault])):
        faultcols_new.append(re_fa_cols_list_e1[fault][cell]*delr)
        faultrows_new.append(re_fa_rows_list_e1[fault][cell]*delc)
    re_fa_cols_list_e1_m.append(faultcols_new)
    re_fa_rows_list_e1_m.append(faultrows_new)


# # #
re_fault_coordinates_e2 = np.load(os.path.join(
    'C:\\workspace\\Proj3_PilbaraBarriers\\UB_55\\re_fault_coords.npy'))

# Set up real barriers at different sizes and locations
re_fa_rows_list_e2, re_fa_cols_list_e2, re_fa_length_list_e2 = (functions_proj_3.barriers.makeFaultRowCols(
                            re_fault_coordinates_e2, 
                            param_dict["nrow"], 
                            param_dict["ncol"]))

re_fa_rows_list_e2_m = []
re_fa_cols_list_e2_m = []

for fault in range(len(re_fa_cols_list_e2)):
    faultcols_new = []
    faultrows_new = []
    for cell in range(len(re_fa_cols_list_e2[fault])):
        faultcols_new.append(re_fa_cols_list_e2[fault][cell]*delr)
        faultrows_new.append(re_fa_rows_list_e2[fault][cell]*delc)
    re_fa_cols_list_e2_m.append(faultcols_new)
    re_fa_rows_list_e2_m.append(faultrows_new)


re_fault_coordinates_e3 = np.load(os.path.join(
    'C:\\workspace\\Proj3_PilbaraBarriers\\UB_65\\re_fault_coords.npy'))

# Set up real barriers at different sizes and locations
re_fa_rows_list_e3, re_fa_cols_list_e3, re_fa_length_list_e3 = (functions_proj_3.barriers.makeFaultRowCols(
                            re_fault_coordinates_e3, 
                            param_dict["nrow"], 
                            param_dict["ncol"]))

re_fa_rows_list_e3_m = []
re_fa_cols_list_e3_m = []

for fault in range(len(re_fa_cols_list_e3)):
    faultcols_new = []
    faultrows_new = []
    for cell in range(len(re_fa_cols_list_e3[fault])):
        faultcols_new.append(re_fa_cols_list_e3[fault][cell]*delr)
        faultrows_new.append(re_fa_rows_list_e3[fault][cell]*delc)
    re_fa_cols_list_e3_m.append(faultcols_new)
    re_fa_rows_list_e3_m.append(faultrows_new)


re_fault_coordinates_e4 = np.load(os.path.join(
    'C:\\workspace\\Proj3_PilbaraBarriers\\UB_57\\re_fault_coords.npy'))

# Set up real barriers at different sizes and locations
re_fa_rows_list_e4, re_fa_cols_list_e4, re_fa_length_list_e4 = (functions_proj_3.barriers.makeFaultRowCols(
                            re_fault_coordinates_e4, 
                            param_dict["nrow"], 
                            param_dict["ncol"]))

re_fa_rows_list_e4_m = []
re_fa_cols_list_e4_m = []

for fault in range(len(re_fa_cols_list_e4)):
    faultcols_new = []
    faultrows_new = []
    for cell in range(len(re_fa_cols_list_e4[fault])):
        faultcols_new.append(re_fa_cols_list_e4[fault][cell]*delr)
        faultrows_new.append(re_fa_rows_list_e4[fault][cell]*delc)
    re_fa_cols_list_e4_m.append(faultcols_new)
    re_fa_rows_list_e4_m.append(faultrows_new)


#------------------------------------------------------------------------------
# Setting up the obs well data

all_mdls_e = ["UB_54_b", "UB_55_b", "UB_65_b", "UB_57_b",
              "UB_58_b", "UB_59_b", "UB_60_b"]  
        
all_obs_pest_obs_file_e = []
         
mdl = all_mdls_e[0]       
for mdl in all_mdls_e:
    path = os.path.join(proj_3_folder, mdl)
    pest_obs_filename_txt = os.path.join(path, (mdl + "_reg.rei"))
    pest_obs_file = pd.read_csv(pest_obs_filename_txt, skiprows=6,
                        delim_whitespace=True, encoding='utf-8-sig') 
    pest_obs_file.index = pest_obs_file.Name
    
    # Add the row and column information to the head and age observations
    pest_obs_file["Row"] = np.zeros(len(pest_obs_file.index))
    pest_obs_file["Col"] = np.zeros(len(pest_obs_file.index))
    
    for idx in pest_obs_file.index:
        if (pest_obs_file.loc[idx, "Group"] == "head" or 
            pest_obs_file.loc[idx, "Group"] == "age"):
            pest_obs_file.loc[idx, "Row"] = int(idx[1:4])
            pest_obs_file.loc[idx, "Col"] = int(idx[5:8])
        else:
            pass
    
    # Change row and column information to be same units as model
    
    pest_obs_file["Row100"] = np.zeros(len(pest_obs_file.index))
    pest_obs_file["Col100"] = np.zeros(len(pest_obs_file.index))
    
    for idx in pest_obs_file.index:
        if (pest_obs_file.loc[idx, "Group"] == "head" or 
            pest_obs_file.loc[idx, "Group"] == "age"):
            pest_obs_file.loc[idx, "Row100"] = pest_obs_file.loc[idx, "Row"]*100
            pest_obs_file.loc[idx, "Col100"] = pest_obs_file.loc[idx, "Col"]*100
        else:
            pass
        
    all_obs_pest_obs_file_e.append(pest_obs_file)
    
    
#------------------------------------------------------------------------------
# Setting up the results

phantom_dfs_e = []
key_dfs_e = []

# Loop
mdl = all_mdls_e[0]       
for mdl in all_mdls_e:
    
    path = os.path.join(proj_3_folder, mdl, "Data")
    phantom_df_f = os.path.join(path, "phantom_df.csv")
    
    phantom_df = pd.read_csv(phantom_df_f)
    
    phantom_dfs_e.append(phantom_df)

    key_df_f = os.path.join(path, "key_dataframe.csv") # or key_df for all faults, key is just those identified >0.8 and low K
    key_df = pd.read_csv(key_df_f)

    key_df["length_diff"] = abs(length_real_barrier - key_df["length"])

    n_key_f = len(key_df.index)

    phantom_df.index = phantom_df.param_name

    key_df.index = key_df.param_name
    
    key_dfs_e.append(key_df)
    

#------------------------------------------------------------------------------
# Plotting the data

#==========================  
fig = plt.figure(figsize=(7, 7))    
#==========================  ##
lw_fig9a=3


ax1  = plt.subplot(4,2,1)

# Plot the real fault location
for i in range(len(re_fa_rows_list_e1)):
        ax1.plot(re_fa_cols_list_e1_m[i], re_fa_rows_list_e1_m[i], lw = lw_fig9a, 
                 color = real_plot_c, ls='-')
        
# Add the observation data
pest_obs_file = all_obs_pest_obs_file_e[0]
pest_obs_file_head = pest_obs_file[pest_obs_file["Group"]=="head"]
  
img = ax1.scatter(pest_obs_file_head.Col100, pest_obs_file_head.Row100,
             c=pest_obs_file_head.Measured,
             s=sz, cmap="coolwarm_r")
        
# Format plot
axes = plt.gca()
axes.set_xlim([0, param_dict["Lx"]])                
axes.set_ylim([0, param_dict["Ly"]]) 
axes.axes.get_xaxis().set_visible(False)

plt.yticks(fontsize=12, rotation=45)
axes.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

# -----------------------------------------------
ax2  = plt.subplot(4,2,3)

# Plot the real fault location
for i in range(len(re_fa_rows_list_e2)):
        ax2.plot(re_fa_cols_list_e2_m[i], re_fa_rows_list_e2_m[i], lw = lw_fig9a, 
                 color = real_plot_c, ls='-')
# Add the observation data
pest_obs_file = all_obs_pest_obs_file_e[1]
pest_obs_file_head = pest_obs_file[pest_obs_file["Group"]=="head"]
  
img = ax2.scatter(pest_obs_file_head.Col100, pest_obs_file_head.Row100,
             c=pest_obs_file_head.Measured,
             s=sz, cmap="coolwarm_r")   

# Format plot   
axes = plt.gca()
axes.set_xlim([0, param_dict["Lx"]])                
axes.set_ylim([0, param_dict["Ly"]]) 
axes.axes.get_xaxis().set_visible(False)

plt.yticks(fontsize=12, rotation=45)
axes.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

#------------------------------------------------
ax3  = plt.subplot(4,2,5)

# Plot the real fault location
for i in range(len(re_fa_rows_list_e3)):
        ax3.plot(re_fa_cols_list_e3_m[i], re_fa_rows_list_e3_m[i], lw = lw_fig9a, 
                 color = real_plot_c, ls='-')
# Add the observation data
pest_obs_file = all_obs_pest_obs_file_e[2]
pest_obs_file_head = pest_obs_file[pest_obs_file["Group"]=="head"]
  
img = ax3.scatter(pest_obs_file_head.Col100, pest_obs_file_head.Row100,
             c=pest_obs_file_head.Measured,
             s=sz, cmap="coolwarm_r")   

# Format plot           
axes = plt.gca()
axes.set_xlim([0, param_dict["Lx"]])                
axes.set_ylim([0, param_dict["Ly"]]) 
axes.axes.get_xaxis().set_visible(False)

plt.yticks(fontsize=12, rotation=45)
axes.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

# ----------------------------------------------------------
ax4  = plt.subplot(4,2,7)

# Plot the real fault location
for i in range(len(re_fa_rows_list_e4)):
        ax4.plot(re_fa_cols_list_e4_m[i], re_fa_rows_list_e4_m[i], lw = lw_fig9a, 
                 color = real_plot_c, ls='-') #real_plot_ls
# Add the observation data
pest_obs_file = all_obs_pest_obs_file_e[3]
pest_obs_file_head = pest_obs_file[pest_obs_file["Group"]=="head"]
  
img = ax4.scatter(pest_obs_file_head.Col100, pest_obs_file_head.Row100,
             c=pest_obs_file_head.Measured,
             s=sz, cmap="coolwarm_r")   

# Format plot           
axes = plt.gca()
axes.set_xlim([0, param_dict["Lx"]])                
axes.set_ylim([0, param_dict["Ly"]]) 

plt.xticks(fontsize=12, rotation=45)
plt.xticks([10000, 20000]) # , ["", "2 km"]
axes.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

plt.yticks(fontsize=12, rotation=45)
axes.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))


#------------------------------------------------------------------------------
ax5  = plt.subplot(4,2,2)

ix = 0

key_df_single = key_dfs_e[ix]
phantom_df_single = phantom_dfs_e[ix]

min_value = np.floor(np.log10(key_df_single["K_posterior"].min()))
max_value = np.ceil(np.log10(key_df_single["K_posterior"].max()))
range_values = max_value*100-min_value*100 #max_value-min_value 
ranges = np.linspace(int(min_value*100), int(max_value*100), int(range_values)+1)

# Add the observation data
pest_obs_file = all_obs_pest_obs_file_e[ix]
pest_obs_file_head = pest_obs_file[pest_obs_file["Group"]=="head"]

img = ax5.scatter(pest_obs_file_head.Col100, pest_obs_file_head.Row100,
             c=pest_obs_file_head.Measured,
             s=sz, cmap="coolwarm_r")

# Plot the phantom fault results
K_leg_value = []
for i in ranges:
    K_leg_value.append("$\geqslant$ %d" %i)
                          
color = cm.terrain(np.linspace(0,1,len(ranges)))
        
    # I am going to re-plot the key faults so they are over the top of the grey ones
for i in range(len(ph_fa_rows_list)):  
    if (phantom_df_single.index[i] in key_df_single.index) == True:
        K_val = int(np.log10(phantom_df_single["K_posterior"][i])*100)
        # K_val = np.floor(np.log10((phantom_df_single["K_posterior"][i])))
        K_col = color[np.where(ranges==K_val)[0][0]]
        K_leg = K_leg_value[np.where(ranges==K_val)[0][0]]
        
        ax5.plot(all_cols_dict[phantom_df_single.index[i]], all_rows_dict[phantom_df_single.index[i]],  
             lw = 4, color = K_col)
                        
    else:
        pass

axes = plt.gca()    
axes.axes.get_xaxis().set_visible(False)
axes.axes.get_yaxis().set_visible(False)

#------------------------------------------------------------------------------
ax6  = plt.subplot(4,2,4)
ix = 1

key_df_single = key_dfs_e[ix]
phantom_df_single = phantom_dfs_e[ix]

min_value = np.floor(np.log10(key_df_single["K_posterior"].min()))
max_value = np.ceil(np.log10(key_df_single["K_posterior"].max()))
range_values = max_value*100-min_value*100 #max_value-min_value 
ranges = np.linspace(int(min_value*100), int(max_value*100), int(range_values)+1)


# Add the observation data
pest_obs_file = all_obs_pest_obs_file_e[ix]
pest_obs_file_head = pest_obs_file[pest_obs_file["Group"]=="head"]

img = ax6.scatter(pest_obs_file_head.Col100, pest_obs_file_head.Row100,
             c=pest_obs_file_head.Measured,
             s=sz, cmap="coolwarm_r")
        
K_leg_value = []
for i in ranges:
    K_leg_value.append("$\geqslant$ %d" %i)
                          
color = cm.terrain(np.linspace(0,1,len(ranges)))

       
    # I am going to re-plot the key faults so they are over the top of the grey ones
for i in range(len(ph_fa_rows_list)):  
    if (phantom_df_single.index[i] in key_df_single.index) == True:
        K_val = int(np.log10(phantom_df_single["K_posterior"][i])*100)
        # K_val = np.floor(np.log10((phantom_df_single["K_posterior"][i])))
        K_col = color[np.where(ranges==K_val)[0][0]]
        K_leg = K_leg_value[np.where(ranges==K_val)[0][0]]
        
        ax6.plot(all_cols_dict[phantom_df_single.index[i]], all_rows_dict[phantom_df_single.index[i]],  
             lw = 4, color = K_col)
                        
    else:
        pass

axes = plt.gca()    
axes.axes.get_xaxis().set_visible(False)
axes.axes.get_yaxis().set_visible(False)


#------------------------------------------------------------------------------
ax7  = plt.subplot(4,2,6)

ix = 2

key_df_single = key_dfs_e[ix]
phantom_df_single = phantom_dfs_e[ix]

min_value = np.floor(np.log10(key_df_single["K_posterior"].min()))
max_value = np.ceil(np.log10(key_df_single["K_posterior"].max()))
range_values = max_value*100-min_value*100 #max_value-min_value 
ranges = np.linspace(int(min_value*100), int(max_value*100), int(range_values)+1)


# Add the observation data
pest_obs_file = all_obs_pest_obs_file_e[ix]
pest_obs_file_head = pest_obs_file[pest_obs_file["Group"]=="head"]

img = ax7.scatter(pest_obs_file_head.Col100, pest_obs_file_head.Row100,
             c=pest_obs_file_head.Measured,
             s=sz, cmap="coolwarm_r")
        
K_leg_value = []
for i in ranges:
    K_leg_value.append("$\geqslant$ %d" %i)
                          
color = cm.terrain(np.linspace(0,1,len(ranges)))

       
    # I am going to re-plot the key faults so they are over the top of the grey ones
for i in range(len(ph_fa_rows_list)):  
    if (phantom_df_single.index[i] in key_df_single.index) == True:
        K_val = int(np.log10(phantom_df_single["K_posterior"][i])*100)
        # K_val = np.floor(np.log10((phantom_df_single["K_posterior"][i])))
        K_col = color[np.where(ranges==K_val)[0][0]]
        K_leg = K_leg_value[np.where(ranges==K_val)[0][0]]
        
        ax7.plot(all_cols_dict[phantom_df_single.index[i]], all_rows_dict[phantom_df_single.index[i]],  
             lw = 4, color = K_col)
                        
    else:
        pass

axes = plt.gca()    
axes.axes.get_xaxis().set_visible(False)
axes.axes.get_yaxis().set_visible(False)

#------------------------------------------------------------------------------
ax8  = plt.subplot(4,2,8)

ix = 3

key_df_single = key_dfs_e[ix]
phantom_df_single = phantom_dfs_e[ix]

min_value = np.floor(np.log10(key_df_single["K_posterior"].min()))
max_value = np.ceil(np.log10(key_df_single["K_posterior"].max()))
range_values = max_value*100-min_value*100 #max_value-min_value 
ranges = np.linspace(int(min_value*100), int(max_value*100), int(range_values)+1)


# Add the observation data
pest_obs_file = all_obs_pest_obs_file_e[ix]
pest_obs_file_head = pest_obs_file[pest_obs_file["Group"]=="head"]

img = ax8.scatter(pest_obs_file_head.Col100, pest_obs_file_head.Row100,
             c=pest_obs_file_head.Measured,
             s=sz, cmap="coolwarm_r")
        
K_leg_value = []
for i in ranges:
    K_leg_value.append("$\geqslant$ %d" %i)
                          
color = cm.terrain(np.linspace(0,1,len(ranges)))

       
    # I am going to re-plot the key faults so they are over the top of the grey ones
for i in range(len(ph_fa_rows_list)):  
    if (phantom_df_single.index[i] in key_df_single.index) == True:
        K_val = int(np.log10(phantom_df_single["K_posterior"][i])*100)
        # K_val = np.floor(np.log10((phantom_df_single["K_posterior"][i])))
        K_col = color[np.where(ranges==K_val)[0][0]]
        K_leg = K_leg_value[np.where(ranges==K_val)[0][0]]
        
        ax8.plot(all_cols_dict[phantom_df_single.index[i]], all_rows_dict[phantom_df_single.index[i]],  
             lw = 4, color = K_col)
                        
    else:
        pass

axes = plt.gca()    
axes.axes.get_yaxis().set_visible(False)

'''

#------------------------------------------------------------------------------
ax9  = plt.subplot(4,3,3)

ix = 4

key_df_single = key_dfs_e[ix]
phantom_df_single = phantom_dfs_e[ix]

#------------------------------------------------------------------------------
ax10  = plt.subplot(4,3,6)

ix = 5

key_df_single = key_dfs_e[ix]
phantom_df_single = phantom_dfs_e[ix]

min_value = np.floor(np.log10(key_df_single["K_posterior"].min()))
max_value = np.ceil(np.log10(key_df_single["K_posterior"].max()))
range_values = max_value*100-min_value*100 #max_value-min_value 
ranges = np.linspace(int(min_value*100), int(max_value*100), int(range_values)+1)


# Add the observation data
pest_obs_file = all_obs_pest_obs_file_e[ix]
pest_obs_file_head = pest_obs_file[pest_obs_file["Group"]=="head"]

img = ax10.scatter(pest_obs_file_head.Col100, pest_obs_file_head.Row100,
             c=pest_obs_file_head.Measured,
             s=sz, cmap="coolwarm_r")
        
K_leg_value = []
for i in ranges:
    K_leg_value.append("$\geqslant$ %d" %i)
                          
color = cm.terrain(np.linspace(0,1,len(ranges)))

       
    # I am going to re-plot the key faults so they are over the top of the grey ones
for i in range(len(ph_fa_rows_list)):  
    if (phantom_df_single.index[i] in key_df_single.index) == True:
        K_val = int(np.log10(phantom_df_single["K_posterior"][i])*100)
        # K_val = np.floor(np.log10((phantom_df_single["K_posterior"][i])))
        K_col = color[np.where(ranges==K_val)[0][0]]
        K_leg = K_leg_value[np.where(ranges==K_val)[0][0]]
        
        ax10.plot(all_cols_dict[phantom_df_single.index[i]], all_rows_dict[phantom_df_single.index[i]],  
             lw = 4, color = K_col)
                        
    else:
        pass

axes = plt.gca()    
axes.axes.get_xaxis().set_visible(False)
axes.axes.get_yaxis().set_visible(False)

#------------------------------------------------------------------------------
ax11  = plt.subplot(4,3,9)

ix = 6

key_df_single = key_dfs_e[ix]
phantom_df_single = phantom_dfs_e[ix]

#------------------------------------------------------------------------------
ax12  = plt.subplot(4,3,9)

ix = 6

key_df_single = key_dfs_e[ix]
phantom_df_single = phantom_dfs_e[ix]

min_value = np.floor(np.log10(key_df_single["K_posterior"].min()))
max_value = np.ceil(np.log10(key_df_single["K_posterior"].max()))
range_values = max_value*100-min_value*100 #max_value-min_value 
ranges = np.linspace(int(min_value*100), int(max_value*100), int(range_values)+1)


# Add the observation data
pest_obs_file = all_obs_pest_obs_file_e[ix]
pest_obs_file_head = pest_obs_file[pest_obs_file["Group"]=="head"]

img = ax11.scatter(pest_obs_file_head.Col100, pest_obs_file_head.Row100,
             c=pest_obs_file_head.Measured,
             s=sz, cmap="coolwarm_r")
        
K_leg_value = []
for i in ranges:
    K_leg_value.append("$\geqslant$ %d" %i)
                          
color = cm.terrain(np.linspace(0,1,len(ranges)))

       
    # I am going to re-plot the key faults so they are over the top of the grey ones
for i in range(len(ph_fa_rows_list)):  
    if (phantom_df_single.index[i] in key_df_single.index) == True:
        K_val = int(np.log10(phantom_df_single["K_posterior"][i])*100)
        # K_val = np.floor(np.log10((phantom_df_single["K_posterior"][i])))
        K_col = color[np.where(ranges==K_val)[0][0]]
        K_leg = K_leg_value[np.where(ranges==K_val)[0][0]]
        
        ax11.plot(all_cols_dict[phantom_df_single.index[i]], all_rows_dict[phantom_df_single.index[i]],  
             lw = 4, color = K_col)
                        
    else:
        pass

axes = plt.gca()    
axes.axes.get_xaxis().set_visible(False)
axes.axes.get_yaxis().set_visible(False)

'''

###############################################################################
###############################################################################
# FIGURE TEN: SINGLE RMSE PLOT
###############################################################################
###############################################################################

all_well_dens_rmse_h
all_well_dens_rmse_a
all_well_dens_rmse_c

rmse_h_list = all_well_dens_rmse_h[sing_wellno][sing_cs] # UB_33_b
rmse_a_list = all_well_dens_rmse_a[sing_wellno][sing_cs] # UB_33_b
rmse_c_list = all_well_dens_rmse_c[sing_wellno][sing_cs] # UB_33_b

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

plt.ylabel("$RMSE_h$", fontsize = 14, labelpad=14) 
plt.grid(which="both")
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14, rotation=0)

axes = plt.gca()

ax1.set_xticklabels([])
axes.set_ylim([0.3,3.0])
axes.set_xlim([-0.2,4.2])

axes.text(3.6, 3.0, "(a)", fontweight="bold")

# --------------------------------------
ax2 = plt.subplot(3, 1, 2)

plt.axhline(y=rmse_a_list[-1], color="m", ls=":", lw=3)
plt.plot(n_faults_ranking_rmse[:-1], rmse_a_list[:-1], "mo", ls="-", label="Age")


plt.ylabel("$RMSE_a$", fontsize = 14, labelpad=6.9)
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
#plt.savefig(os.path.join(pprfigureDirectory, "SingleRMSE_h_a_c.pdf"), dpi=dpi_value, format=plt_format)  
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

###############################################################################
###############################################################################
# FIGURE EIGHT: SINGLE SENSITIVITIY DATA PLOT
###############################################################################
###############################################################################

os.chdir(r'C:\workspace\Proj3_PilbaraBarriers\UB_33_b')
modelname = "UB_33_b"
modelname_mt = "UB_33_b_MT3D"

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

#plt.savefig(os.path.join(pprfigureDirectory, "Single_Sens.pdf"), dpi=dpi_value, format=plt_format)  
plt.savefig(os.path.join(pprfigureDirectory, "Single_Sens"), dpi=dpi_value)  


###############################################################################
###############################################################################
# FIGURE ELEVEN: PLOTTING K AVERAGED OUT 
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

#plt.savefig(os.path.join(pprfigureDirectory, "AllKVals.pdf"), dpi=dpi_value,format=plt_format)  
plt.savefig(os.path.join(pprfigureDirectory, "AllKVals"), dpi=dpi_value)  

###############################################################################
###############################################################################
# FIGURE TWELVE: PLOTTING RMSE AVERAGED OUT - average obs wells & ph faults 
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


colour_rmse = cm.brg(np.linspace(0, 1, len(well_densities))) # gnuplot

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#==========================  
plt.figure(figsize=(7, 4))
#==========================  

ax = plt.subplot(1, 1, 1)
x_axis = list(range(0, 10)) 
for welldensoptn in range(len(well_dens_mean_rmse_lists)):
    lbl = well_densities[welldensoptn]
    ax.plot(x_axis, well_dens_mean_rmse_lists[welldensoptn], 
            marker="o", label=lbl, c=colour_rmse[welldensoptn])
    
plt.legend(title="Number of wells")
plt.xlabel("Number of phantom faults")
plt.ylabel(r"$RMSE_c$")
plt.grid()
plt.tight_layout()

#plt.savefig(os.path.join(pprfigureDirectory, "RMSE_all_mean.pdf"), dpi=dpi_value, format=plt_format)        
plt.savefig(os.path.join(pprfigureDirectory, "RMSE_all_mean"), dpi=dpi_value)        



#_ _ _ _ _ _ _ _ _ 
plt.close("all")
