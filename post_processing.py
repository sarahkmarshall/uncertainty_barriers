# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:13:03 2019

@author: mars0451
"""

#=====IMPORTS==================================================================

import flopy
import flopy.utils.binaryfile as bf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os, shutil
import pandas as pd
import pyemu
import sys
import string
import matplotlib.cm as cm
import math
import subprocess
from flopy import mt3d
import pickle
from scipy import stats

functions_path = "C:\workspace\sarahs_functions"

if any(functions_path in s for s in sys.path):
    print("Functions path already in list")
else:
    sys.path.append(r"C:\workspace\sarahs_functions")

import functions_proj_3

#==============================================================================
#FIGURE SETTINGS ==============================================================

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
mpl.rc('legend', labelspacing=0.1, frameon=True, fontsize=fontSize)
mpl.rc("figure", figsize=(cm2inch(8.25*2), cm2inch(8.25*1.25)), titlesize='medium')
#mpl.rc('font', **{'sans-serif' : 'Helvetica','family' : 'sans-serif'}) 
mpl.rc('font',family='Arial')
# # # Settings

#==============================================================================
#=====SETTING UP MULTIPLE RUNS=================================================

already_run_this_script = True
putting_in_key_faults = False


savedata_question = 'NO' # 'YES' or 'NO'

n_runs = 18

### VARIABLES 

n_real_faults = 1 


hk_real_fault = 1e-3

normal_score_transform_used = "NO" # "YES"  or "NO"

hk_array_cmap = 'BrBG'

include_real_with_ph = False

length_real_barrier = 5515 #m

#=====DIRECTORIES==============================================================

proj_3_folder       = r'C:\workspace\Proj3_PilbaraBarriers'

overallName = "UB_53"

overarching_model_folder = os.path.join(proj_3_folder, overallName)

run = 1
for run in range(0, n_runs): # n_runs       
    if putting_in_key_faults == True:
        modelname = overallName + "_" + string.ascii_lowercase[run] + "_key"
    else:
        modelname = overallName + "_" + string.ascii_lowercase[run]
    print(modelname)

    modelname_mt = modelname + "_MT3D" 
    print(modelname_mt)
    
    savedata_question   =   'NO' # 'YES' or 'NO'
    
    MODFLOW_folder      =   r'C:\workspace\modflow_dbl\mfnwtdbl.exe' # double precision MODFLOW NWT
    mf_exe              =   "mfnwtdbl.exe" 
    
    MT3D_USGS_folder    =   r'C:\workspace\modflow_dbl\mt3dusgsdbl.exe' # Double precision MT3D-USGS
    mt3d_version        =   'mt3d-usgs'  
    mt3d_exe            =   'mt3dusgsdbl.exe'
    
    this_model_folder = os.path.join(proj_3_folder, modelname)
    if not os.path.exists(this_model_folder):
        os.makedirs(this_model_folder)
    
    dataDirectory = os.path.join(this_model_folder, 'Data')
    if not os.path.exists(dataDirectory):
        os.makedirs(dataDirectory)
    
    figureDirectory = os.path.join(this_model_folder, 'Figures')
    if not os.path.exists(figureDirectory):
        os.makedirs(figureDirectory)
      
    pst_name_reg = str(modelname) + '_reg.pst'
    
    os.chdir(this_model_folder)

#=====MODEL PARAMETERS=====================================================
    
    os.chdir(overarching_model_folder)
    
    with open("param_dict.p", "rb") as fp:
        param_dict = pickle.load(fp)
        
    os.chdir(this_model_folder)
    
    recharge_array = param_dict["recharge_flux"] * np.ones((param_dict["nrow"], 
                     param_dict["ncol"]), dtype=np.float32)
    
#=====BOUNDARY CONDITIONS======================================================

    stageleft = float(param_dict["sHead"]) # to have a head gradient of 0.001
    stageright = float(param_dict["sHead"]-100)
    bound_sp1 = []
    for il in range(param_dict["nlay"]):
        condleft = param_dict["hk_aquifer"] * (stageleft - param_dict["zbot"]) * param_dict["delc"]
        condright = param_dict["hk_aquifer"] * (stageright - param_dict["zbot"]) * param_dict["delc"]
        for ir in range(param_dict["nrow"]):
            bound_sp1.append([il, ir, 0, stageleft, condleft])
            bound_sp1.append([il, ir, param_dict["ncol"] - 1, stageright, condright])
    print('Adding ', len(bound_sp1), 'GHBs for stress period 1.')
    ghb_spd = {0: bound_sp1}   
       
###############################################################################
# SAVE SECURE COPY OF PEST FILES
    
    ext_path = os.path.join(this_model_folder, "ref")

    ext_files = [f for f in os.listdir(ext_path)]

    
    # I need to rerun the model, because I "accidentally" wrote over the previous 
    # head and age files by running script "UB_01_h.py" again.
    
    final_pest_files_folder = os.path.join(this_model_folder, "final_from_pest")
    
    # Recursively delete a directory tree if it already exists so I'm not using old data.
    if already_run_this_script == False:
        if os.path.exists(final_pest_files_folder):
            shutil.rmtree(final_pest_files_folder) 
    
        else:
            pass
        
    # Now make a new external directory:
        os.mkdir(final_pest_files_folder)
        
    # Copy all of the final pest files from ref folder to "final pest files folder"
    # so that if I accidentally change them they are preserv
        for f in ext_files:
            shutil.copy2(os.path.join(ext_path, f),  
                         os.path.join(final_pest_files_folder, f))
        
    else:
        pass
    
#==============================================================================
#  EXPLORING THE RESULTS OF THE PEST RUN
# # # This section needs work!
#==============================================================================
    # I've gone through and taken stuff from mike fienen github, activities, freyberg_k
    # Set up all the names etc.
    
    '''
    Let's explore the results
    
    First let's look at the measurement objective function (Phi), which is calculated using the sum of squared weighted residuals. First we'll look at a table, then plots
    
    I can't do this because iobj is a PEST++ file!
    '''
    
    #df_obj = pd.read_csv(os.path.join(working_dir, pst_name_reg.replace(".pst",".iobj")),index_col=0)
    
    '''
    Let's evaulate our fit using the observed-simulated residuals
    '''
    
    # define pst to be all the pest-type things of the *.pst file defined above by pst_name
    pst = pyemu.Pst(os.path.join(this_model_folder,pst_name_reg))
    # define res_nz to equal a table of residuals for all observations with non-zero weight
    res_nz = pst.res.loc[pst.nnz_obs_names,:]
    # echo out res_nz
    res_nz
    # use pyemu's plot utilities to plot 1:1 line and the residuals as fxn of observation magnitude
    plt.figure()
    pyemu.plot_utils.res_1to1(pst)
    plt.savefig(figureDirectory + "\plotresidual")
    
    print(pst.phi)
    
    plt.figure()
    pst.plot(kind='phi_pie')
    plt.savefig(figureDirectory + "\contribution_to_calib")
    
    # from pst demo --> Jeremy notebook
    from pyemu import Pst
    
    p = Pst(pst_name_reg)
    
    # A residual file (.rei or res) can also be passed to the resfile argument 
    # at instantiation to enable some simple residual analysis and weight 
    # adjustments. If the residual file is in the same directory as the pest 
    # control file and has the same base name, it will be accessed automatically:
    p.res
    
    #Convenient methods:
    print(p.phi,p.phi_components)
    print(p.npar,p.nobs,p.nprior)
    print(p.par_groups,p.obs_groups)
    print(type(p.par_names)) # all parameter names
    print(type(p.adj_par_names)) # adjustable parameter names
    print(type(p.obs_names)) # all observation names
    print(type(p.nnz_obs_names)) # non-zero weight observations
    
    
    print('jacupdate = {0}'.format(p.control_data.jacupdate))
    print('numlam = {0}'.format(p.control_data.numlam))
    p.control_data.numlam = 100
    print('numlam has been changed to --> {0}'.format(p.control_data.numlam))
    
# Download par file from pest results-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    
    pest_par_filename_txt = os.path.join(this_model_folder, (modelname + "_reg.par"))
    
    pest_par_file = pd.read_csv(pest_par_filename_txt, header=None, 
                                delim_whitespace=True, skiprows=1, encoding='utf-8-sig') 
    
    pest_par_file.columns = ["param_name", "k_est", "scale", "offset"]
    pest_par_file = pest_par_file.set_index('param_name')
       
    
    pest_par_file.index[0] # Check that it hasn't got a weird index - I fixed this with encoding 'utf-8-sig'
    
    # Now upload original par values
    real_values_filename = os.path.join(this_model_folder, 'real_K_barr_values.csv')
    real_values_file = pd.read_csv(real_values_filename)
    
    # Add aquifer value to original par values
    aq_name_list = ["hkaq"]
    K_aq_list = [1.0]
    aquifer_param = pd.DataFrame({
                                      
                "barr_name": aq_name_list,
                "K_barr_orig": K_aq_list
                                            })
    real_values_file = real_values_file.append(aquifer_param)
    
    rl_fault_name_format = "hkre_%03d"
    real_fault_prior_value = 1.
    
    if include_real_with_ph == True:
        rl_fault_name_list = []
        rl_fault_K_list = []    

        for i in range(n_real_faults):
            rl_fault_name_list.append(rl_fault_name_format % (i+1))   
        
        for i in range(n_real_faults):
            rl_fault_K_list.append(real_fault_prior_value)
            
        realfault_param = pd.DataFrame({
                              
        "barr_name": rl_fault_name_list,
        "K_barr_orig": rl_fault_K_list
                                    })    
        
        real_values_file = real_values_file.append(realfault_param)

    else:
        pass
    

    real_values_file = real_values_file.rename(columns={"barr_name": "param_name"})
    
    real_values_file = real_values_file.set_index("param_name")
          
    # Get obs well locations 
    
    # These are modflow based
    
    # I think I should change the name of "pest_obswell_file"! It's not really from PEST
    rc_df_filename  = os.path.join(dataDirectory, 'obs_rowcol.csv') 
    rc_df           = pd.read_csv(rc_df_filename)
    
    
    # Make x and y arrays for plotting - based on middle of the cells    
    obs_well_x      = rc_df["c_wellcol"]*param_dict["delr"] + 0.5*param_dict["delr"]
    obs_well_y      = rc_df["b_wellrow"]*param_dict["delc"] + 0.5*param_dict["delc"]

# Download obs file from pest results-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

    obs_name_format = "i{1:03d}j{2:03d}"

    pest_obs_filename_txt = os.path.join(this_model_folder, (modelname + "_reg.rei"))
    
    # BUT THIS IS AT THE END OF ITERATION RUN # 29 --> i MIGHT NEED TO RUN PEST AGAIN TO GET THE FINAL FINAL?
    '''
    PEST record file says:
        "The model has been run one final time using best parameters. 
   Thus all model input files contain best parameter values, and model 
   output files contain model results based on these parameters."
   '''
   
    pest_obs_file = pd.read_csv(pest_obs_filename_txt, skiprows=6,
                            delim_whitespace=True, encoding='utf-8-sig') 
    
    pest_obs_file.index = pest_obs_file.Name
    
#==============================================================================
#  EXPLORING THE FINAL PARAMETER VALUES
#==============================================================================
    # Compare real to estimated parameters
    # Check this df that the "barr_name" seems to equal "param_name"
    pest_real_pars =  pd.merge(left=pest_par_file, right=real_values_file,
                                             how='left', left_on="param_name", 
                                             right_on="param_name")
    
    pest_real_pars.head(10)
    
    #pest_real_pars.drop("barr_name", axis=1)
    
    # pest_real_pars.index = pest_real_pars.param_name
    
    # Print index to check, and checking my sampling technique works.
    pest_real_pars.index[0]
    
    a = pest_real_pars.loc['hkaq', 'K_barr_orig']
    b = pest_real_pars.loc["hkaq", "k_est"]

    if include_real_with_ph == True:
        c = pest_real_pars.loc['hkre_001', 'K_barr_orig']
        d = pest_real_pars.loc["hkre_001", "k_est"]
    else:
        pass
    
    # ----------------------
    # Normal score transform
    #-----------------------
    # Note that because of normal score transform, the parameter is between 0 and 1 probability
    # NOT an actual K estimate.

    if normal_score_transform_used == "YES":  
        def normal(x,mu,sigma):
            return(1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (x - mu)**2 / (2 * sigma**2)))
        
        min_value_x = -10
        max_value_x = 3
            
        x = np.linspace(min_value_x, max_value_x, 1000)
        
        # Creating a bimodal distribution from two separate distributions
        mu_aquifer_K = 0
        mu_barrier_K = -3
        
        st_dev_aquifer_K = 0.1
        st_dev_barrier_K = 0.1
        
        p1 = normal(x, mu_barrier_K, st_dev_barrier_K) # Barrier distribution
        p2 = normal(x, mu_aquifer_K, st_dev_aquifer_K) # Aquifer distribution
            
        p = 0.5*p1 + 0.5*p2 # Combined distribution    
        P = np.cumsum(p) * (x[1]-x[0]) # Cumulative frequency plot
    #    plt.plot(x, P)
        #=#= Normal score transformation using my cumulative frequency distribution, P
        for i in pest_real_pars.index:
            print(i)        
            PP = pest_real_pars.loc[i, "k_est"] # = 1.0
            print(PP)
            n = np.argmin((P-PP)**2)
            K = 10**x[n]
            print(K)
            pest_real_pars.loc[i, "k_est"] = K # Now replace the value to be the actual parameter value not probability
    else:
        pass
# end normal score transform --------------------------------------------------

    # Create a dataframe that can include all of the information/results
    # Called "phantom_df" --> Just the phantom faults, not the aquifer.
    
    phantom_df = pest_real_pars
    # phantom_df = phantom_df.set_index("param_name")
    idx_c = phantom_df.columns
    idx_r = phantom_df.index
    
    #Rename labels
    phantom_df = phantom_df.rename(columns={"k_est":"K_posterior", 
                                            "K_barr_orig":"K_prior"})

    # Drop the aquifer values
    phantom_df = phantom_df.drop('hkaq')
    
    # -------------------------------------------------------------------------
    # PLOT PRIOR VERSUS POSTERIOR FOR ALL PHANTOM FAULTS AND AQUIFER
    #plt.figure()
    ax = pest_real_pars.plot.scatter(x="K_barr_orig", y="k_est", label="Phantom fault")
    
    if include_real_with_ph == True:
        ax.plot(c, d, "m*", markersize=15, label="Real fault values")
    else:
        pass
    ax.plot(a, b, "r*", markersize=15, label="Aquifer values")

    x_1to1 = [-10, 100]
    y_1to1 = [-10, 100]
    
    ax.plot(x_1to1, y_1to1, ls="--", color="k")
    plt.yscale("log")
    plt.xscale("log")
    
    ax.set_xlim([1e-1, 10])
    ax.set_ylim([1e-10, 10])
    
    plt.xlabel("Parameter PRIOR value", fontsize=14)
    plt.ylabel("Parameter POSTERIOR value", fontsize=14)
    
    plt.legend(loc=3)
    
    plt.grid()
    
    plt.savefig(figureDirectory + "\prior_vs_posterior")
    
    # -----------------------------------------------------------------------------
    
    # Importing kriged parameter files from PEST (the file where pilot point estimates
    # have been interpolated using fac2real)
    
    # Determine the final files to access & load them
    #hk_file = os.path.join(this_model_folder, "ref", "hk_Layer_1.ref")
    hk_file = os.path.join(final_pest_files_folder, "hk_Layer_1.ref")
    hk_array_pest = np.loadtxt(hk_file)
    
    # Load original age file
    # original_age_file = os.path.join(dataDirectory, "original_age_file")
    # original_age_file = np.loadtxt(original_age_file)
    
    # Load original head file
    original_head_file = os.path.join(dataDirectory, "original_head_file")
    original_head_file = np.loadtxt(original_head_file)
    
    # Load original hk file
    original_hk_file = os.path.join(dataDirectory, "original_hk_file")
    original_hk_file = np.loadtxt(original_hk_file)
    
    #=====HYDRAULIC BARRIER========================================================
    # Set up various barriers at different sizes and locations
    n_phantom_faults = len(phantom_df)
    
    ph_fault_name_list = []
    
    for i in range(n_phantom_faults):
        ph_fault_name_list.append("hkph_%03d" % (i+1)) 
        
    ph_fault_coordinates = np.load(os.path.join(dataDirectory, "ph_fault_coords.npy")) # Saved before I run original model
    re_fault_coordinates = np.load(os.path.join(dataDirectory, "re_fault_coords.npy"))
    
    # Set up various barriers at different sizes and locations
    
    ph_fa_rows_list, ph_fa_cols_list, ph_fa_length_list = (functions_proj_3.barriers.makeFaultRowCols(
                                ph_fault_coordinates, 
                                param_dict["nrow"], 
                                param_dict["ncol"]))

    # Remove any faults that turned out not on the grid
    for i in range(len(ph_fa_length_list)):
        if ph_fa_length_list[i] == 0:
            print("Length of barrier is 0 --> no barrier on grid")
            while ph_fa_length_list[i] == 0:
                fault_coordinate_redone = functions_proj_3.barriers.makeFaultCoords(1, 
                                                        param_dict["nrow"], 
                                                        param_dict["ncol"])    
                ph_fault_coordinates[i] = fault_coordinate_redone[0]
                ph_fa_rows_list, ph_fa_cols_list, ph_fa_length_list = (
                        functions_proj_3.barriers.makeFaultRowCols(ph_fault_coordinates, 
                                                                   param_dict["nrow"], 
                                                                   param_dict["ncol"]))
                ph_fa_length_list[0]
    
        else:
            print("Length of barrier is fine, there is a barrier on the grid")
            
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
    
    # Make empty column for each barrier coordinate
    phantom_df["row_start"] = np.zeros(len(phantom_df.index))
    phantom_df["col_start"] = np.zeros(len(phantom_df.index))
    phantom_df["row_end"]   = np.zeros(len(phantom_df.index))
    phantom_df["col_end"]   = np.zeros(len(phantom_df.index))
            
            
    # Be careful when I do this that the faults are labelled in numerical order    
    for i in range(len(phantom_df.index)):
        index = phantom_df.index[i]
        if index[:4] == "hkph":
            phantom_df.loc[index, "row_start"] = ph_fault_coordinates[i, 0]
            phantom_df.loc[index, "col_start"] = ph_fault_coordinates[i, 1]
            phantom_df.loc[index, "row_end"]   = ph_fault_coordinates[i, 2]
            phantom_df.loc[index, "col_end"]   = ph_fault_coordinates[i, 3]
        elif index[:4] == "hkre":
            phantom_df.loc[index, "row_start"] = re_fault_coordinates[i-n_phantom_faults, 0]
            phantom_df.loc[index, "col_start"] = re_fault_coordinates[i-n_phantom_faults, 1]
            phantom_df.loc[index, "row_end"]   = re_fault_coordinates[i-n_phantom_faults, 2]
            phantom_df.loc[index, "col_end"]   = re_fault_coordinates[i-n_phantom_faults, 3]   
        else:
            print("Check what the index is, should only be ph or re")
    
    # I can't add rows and cols to the df as they contain lists of different 
    # lengths, use dictionaries instead 
    
    ph_rows_dict = {}
    ph_cols_dict = {}
    re_rows_dict = {}
    re_cols_dict = {}

    for i in range(len(ph_fault_name_list)):
        ph_rows_dict.update({ph_fault_name_list[i]: ph_fa_rows_list[i]})
        ph_cols_dict.update({ph_fault_name_list[i]: ph_fa_cols_list[i]})
        
    if include_real_with_ph == True:
        for i in range(len(rl_fault_name_list)):
            re_rows_dict.update({rl_fault_name_list[i]: re_fa_rows_list[i]})
            re_cols_dict.update({rl_fault_name_list[i]: re_fa_cols_list[i]}) 
            
        # Combine real and phantom row and cols dicts..this will make things easier I think!
        all_rows_dict = {**ph_rows_dict, **re_rows_dict}
        all_cols_dict = {**ph_cols_dict, **re_cols_dict}
    else:
        all_rows_dict = ph_rows_dict
        all_cols_dict = ph_cols_dict
        
    # Add legend list to the df
    
    phantom_df["hk_log10"] = np.log10(phantom_df["K_posterior"])
    
    # Compare real fault hk to the pest-generated prior
    phantom_df["hk_real_diff"] = phantom_df["K_posterior"] - hk_real_fault
    phantom_df["hk_real_diff_log"] = phantom_df["hk_log10"] - np.log10(hk_real_fault)
    
    # Compare real aquifer hk to the pest-generated prior
    phantom_df["hk_real_diff_aq"] = phantom_df["K_posterior"] - 1.0
    phantom_df["hk_real_diff_log_aq"] = phantom_df["hk_log10"] - np.log10(1.0)
    
    # Plot pest-determined barriers -----------------------------------------------
    
    # Use log for max and min values
    min_value = np.log10(np.min(phantom_df['K_posterior']))
    max_value = np.log10(np.max(phantom_df['K_posterior']))
       
    a = np.linspace(math.floor(min_value),math.ceil(max_value)+1,6)
    leg_value = []

    for i in range(len(a)-1):
        leg_value.append("%2d to %2d" %(a[i], a[i+1]))
   
    leg_colour = cm.rainbow(np.linspace(0,1,5))
    
        
    color=cm.rainbow(np.linspace(0,1,(math.ceil(max_value)-math.floor(min_value)+1)))    
       
    plt.figure(figsize=(20, 10))
    plt.title("PEST-determined barriers", fontsize=14)
    
    for i in range(n_phantom_faults):
        val = (np.log10(phantom_df["K_posterior"][i]) - math.floor(min_value))
        val_color = color[int(val)] 
        plt.plot(ph_fa_cols_list[i], ph_fa_rows_list[i], color=val_color, lw = 3)
        
    # Now plot real fault --> this is set up to just plot one real fault
    if include_real_with_ph == True:
        val = (np.log10(phantom_df.loc["hkre_001", "K_posterior"]) - math.floor(min_value))
        val_color = color[int(val)] 
        plt.plot(re_fa_cols_list[0], re_fa_rows_list[0], color=val_color, lw = 3)
    else:
        pass
        
    plt.xlabel("Columns", fontsize=14)
    plt.ylabel("Rows", fontsize=14)
    
    for i in range(len(leg_colour)):
        plt.plot([-10, -15], [-10, -15], lw=3, color=leg_colour[i], label=leg_value[i])
                
    plt.legend()
  
    axes = plt.gca()
    axes.set_xlim([0, param_dict["ncol"]])                
    axes.set_ylim([0, param_dict["nrow"]])                
    
    plt.legend(loc=4, fontsize=14) # legend_list  
    plt.savefig(os.path.join(figureDirectory, "barrs_hk_pest"))
    
    # Plotting final PEST hk as a log ########################################################
    
    # This is the hk file that I have loaded from PEST - 
    # It is NOT necessarily the "best" parameter set as the point within the parameter estimation process
    # where the "best" Jacobian martiex was computed varies from run to run.
    
    log_interp_pest_hk = np.log10(hk_array_pest)
    
    plt.figure(figsize=(10, 5))
    plt.title("HK as a log", fontsize=14)
    plt.subplot(1, 1, 1, aspect='equal')    
    extent = (param_dict["delr"]/2., param_dict["Lx"] - param_dict["delr"]/2., 
              param_dict["delc"]/2., param_dict["Ly"] - param_dict["delc"]/2.)
    plt.imshow(np.flipud(log_interp_pest_hk[ :, :]), extent=extent, cmap='BrBG',
                         vmin = -3, vmax = 3) # vmax=np.amax(head_1[0, :, :]), vmin = np.amin(head_1[0, :, :])
    # vmin = -2.5, vmax = 2.5, Standard colormap = BrBG
    cbr = plt.colorbar()
    cbr.ax.tick_params(labelsize=14) 
    cbr.set_label('LOG Estimated K (m/d)', fontsize = 14)
    
        
    plt.grid()
    plt.xlabel("x axis (m)", fontsize = 14)
    plt.ylabel("y axis (m)", fontsize = 14)
    
    plt.savefig(figureDirectory + "\hk_pestfinal_interp_log_noobs")    
    
    # Plot location of observation wells
    
    for i in range(len(obs_well_x)):
        plt.plot(obs_well_x[i], 
                 obs_well_y[i], 
        "m*", markersize=10)
    
    plt.savefig(figureDirectory + "\hk_pestfinal_interp_log")
    
   # WHAT IS THE final obs data used --> i.e. with residual  ------------------------------

    # Plot head & age obs -----------------------------------------------------------   
    
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
        
    # Plotting residuals -----------------------------------------------------------------------------
    plt.figure()
    plt.suptitle("Residuals coloured by column")
    ax = plt.subplot(2, 1, 1)
    ax.title.set_text("Head")
    axes = plt.gca()
    pest_obs_file[pest_obs_file["Group"]=="head"].plot.scatter(x="Measured", y="Modelled", 
                                c = "Col", colormap = 'plasma', 
                               ax=axes, s=20, linewidths=0.0)
    # ~ # ~ # ~ # ~ 
    ax = plt.subplot(2, 1, 2)    
    ax.title.set_text("Age")
    axes = plt.gca()
    pest_obs_file[pest_obs_file["Group"]=="age"].plot.scatter(x="Measured", y="Modelled", 
                                c = "Col", colormap = 'plasma', 
                               ax=axes, s=20, linewidths=0.0)
    
    plt.savefig(os.path.join(figureDirectory, "resid_column"))
    
    # ------------------------------------------------------------------------------------------------
    # Spatially plotting the modelled results from pest
    
    plt.figure(figsize=(10,10))
    plt.suptitle("Head and age data - Final PEST", fontsize=20)
    ax = plt.subplot(2, 1, 1)
    ax.title.set_text("Head")
    axes1 = plt.gca() 

    pest_obs_file[pest_obs_file["Group"]=="head"].plot.scatter(x="Col", y="Row", c="Modelled", s=30, 
                                                   colormap="coolwarm_r", linewidths=0.0, ax=axes1)

    axes1.set_xlim([0, param_dict["Lx"]/param_dict["delr"]])
    axes1.set_ylim([0, param_dict["Ly"]/param_dict["delc"]])
    plt.xlabel("Columns", fontsize=20)
    plt.ylabel("Rows", fontsize=20)

    # ~ # ~ # ~ # ~    
    ax = plt.subplot(2, 1, 2)
    ax.title.set_text("Age")
    axes2 = plt.gca()
    pest_obs_file[pest_obs_file["Group"]=="age"].plot.scatter(x="Col", y="Row", c="Modelled", s=30, 
                                                           colormap="jet", linewidths=0.0, ax=axes2)

    axes2.set_xlim([0, param_dict["Lx"]/param_dict["delr"]])
    axes2.set_ylim([0, param_dict["Ly"]/param_dict["delc"]])
    plt.xlabel("Columns", fontsize=20)
    plt.ylabel("Rows", fontsize=20)
    
    plt.savefig(os.path.join(figureDirectory, "a_h_obs_pest_modelled"))    
    
    # ------------------------------------------------------------------------------------------------
    
    plt.figure(figsize=(10,10))
    plt.suptitle("Residual value - Final PEST", fontsize=20)
    ax = plt.subplot(2, 1, 1)
    ax.title.set_text("Head")
    axes1 = plt.gca() 

    pest_obs_file[pest_obs_file["Group"]=="head"].plot.scatter(x="Col", y="Row", c="Residual", s=30, 
                                                   colormap="plasma", linewidths=0.0, ax=axes1)

    axes1.set_xlim([0, param_dict["Lx"]/param_dict["delr"]])
    axes1.set_ylim([0, param_dict["Ly"]/param_dict["delc"]])
    plt.xlabel("Columns", fontsize=20)
    plt.ylabel("Rows", fontsize=20)

    # ~ # ~ # ~ # ~    
    ax = plt.subplot(2, 1, 2)
    ax.title.set_text("Age")
    axes2 = plt.gca()
    pest_obs_file[pest_obs_file["Group"]=="age"].plot.scatter(x="Col", y="Row", c="Residual", s=30, 
                                                           colormap="plasma", linewidths=0.0, ax=axes2)

    axes2.set_xlim([0, param_dict["Lx"]/param_dict["delr"]])
    axes2.set_ylim([0, param_dict["Ly"]/param_dict["delc"]])
    plt.xlabel("Columns", fontsize=20)
    plt.ylabel("Rows", fontsize=20)
    
    plt.savefig(os.path.join(figureDirectory, "a_h_obs_pest_resid"))       
    
    # ------------------------------------------------------------------------------------------------   
    # SENSITIVITY DATA --> OBSERVATIONS
    
    # Upload observation sensitivity data
    sens_obs_data_name = os.path.join(this_model_folder, (modelname + "_reg.seo"))
      
    sens_obs_file = pd.read_csv(sens_obs_data_name, delim_whitespace=True, encoding='utf-8-sig') 
    sens_obs_file.index = sens_obs_file.Observation

    # Get row and column from sensitivity data for plotting
    for idx in sens_obs_file.index:
        if (sens_obs_file.loc[idx, "Group"] == "head" or 
            sens_obs_file.loc[idx, "Group"] == "age"):
            sens_obs_file.loc[idx, "Row"] = int(idx[1:4])
            sens_obs_file.loc[idx, "Col"] = int(idx[5:8])
        else:
            pass

    # Plot observation sensitivity data
    plt.figure(figsize=(10,10))
    plt.suptitle("PEST SENSITIVITY value of observation", fontsize=20)
    ax = plt.subplot(2, 1, 1)
    ax.title.set_text("Head")
    axes1 = plt.gca() 
    
    # Plot observations sensitivity data
    sens_obs_file[sens_obs_file["Group"]=="head"].plot.scatter(x="Col", y="Row", c="Sensitivity", s=30, 
                                                   colormap="cool", linewidths=0.0, ax=axes1)

    axes1.set_xlim([0, param_dict["Lx"]/param_dict["delr"]])
    axes1.set_ylim([0, param_dict["Ly"]/param_dict["delc"]])
    plt.xlabel("Columns", fontsize=20)
    plt.ylabel("Rows", fontsize=20)

    # ~ # ~ # ~ # ~    
    ax = plt.subplot(2, 1, 2)
    ax.title.set_text("Age")
    axes2 = plt.gca()
    sens_obs_file[sens_obs_file["Group"]=="age"].plot.scatter(x="Col", y="Row", c="Sensitivity", s=30, 
                                                           colormap="cool", linewidths=0.0, ax=axes2)

    axes2.set_xlim([0, param_dict["Lx"]/param_dict["delr"]])
    axes2.set_ylim([0, param_dict["Ly"]/param_dict["delc"]])
    plt.xlabel("Columns", fontsize=20)
    plt.ylabel("Rows", fontsize=20)
    
    plt.savefig(os.path.join(figureDirectory, "a_h_obs_sens"))         
  
    
    # PLOT LOG DIFF IN HK  ----------------------------------------------------
    min_value = np.min(phantom_df['hk_real_diff_log'])
    max_value = np.max(phantom_df['hk_real_diff_log'])
    
    diffhklog_leg_colour = cm.rainbow(np.linspace(0,1,5))
    
   
    a = np.linspace(math.floor(min_value),math.ceil(max_value)+1,6)
    diffhkog_leg_value = []
    for i in range(len(a)-1):
        diffhkog_leg_value.append("%2d to %2d" %(a[i], a[i+1]))

    
    color=cm.rainbow(np.linspace(0,1,(math.ceil(max_value)-math.floor(min_value)+1)))
    
    plt.figure(figsize=(20, 10))
    plt.title("Diff in hk log", fontsize=14)
    for i in range(len(ph_fa_rows_list)):
        val = (phantom_df["hk_real_diff_log"][i] - math.floor(min_value))
        print(val)
        val_color = color[int(val)] 
        print("Orientation value is: %d" %val)
        plt.plot(ph_fa_cols_list[i], ph_fa_rows_list[i], lw = 3, 
                 color = val_color)
        
        
     # Now plot real fault --> this is set up to just plot one real fault
    if include_real_with_ph == True:
        val = (np.log10(phantom_df.loc["hkre_001", "hk_real_diff_log"]) - math.floor(min_value))
        val_color = color[int(val)] 
        plt.plot(re_fa_cols_list[0], re_fa_rows_list[0], color=val_color, lw = 3)
    else:
        pass

    plt.xlabel("Columns", fontsize=14)
    plt.ylabel("Rows", fontsize=14)
    
    # Legend
    for i in range(len(diffhklog_leg_colour)):
        plt.plot([-10, -15], [-10, -15], lw=3, color=diffhklog_leg_colour[i], label=diffhkog_leg_value[i])
      
    axes = plt.gca()
    axes.set_xlim([0, param_dict["ncol"]])                
    axes.set_ylim([0, param_dict["nrow"]])                
    
    plt.legend(loc=4, fontsize=14) # legend_list  
    plt.savefig(os.path.join(figureDirectory, "diff_hk_log"))  
    
    
    ######################## ######################## ########################
    # Phantom fault identifiability
    # How much did each parameter contribute to the calibration?
    ######################## ######################## ########################
    plt.close("all")
    
    # Copy identpar to folder
    # Copy pest into folder
    if putting_in_key_faults == True:
        n_singular_values = int(0.5*n_phantom_faults)
    else:
        n_singular_values = 10
    
    
    identpar_folder = os.path.join('C:\workspace\Proj3_PilbaraBarriers\pest_files', "identpar.exe")
    this_model_identpar = os.path.join(this_model_folder, "identpar.exe")
    
    if not os.path.isfile(this_model_identpar):
        print("identpar not yet in folder, so it's being added")
        shutil.copy2(identpar_folder, this_model_identpar)
    else:
        print("identpar is already in folder")
    
    reg_pest_name = modelname + "_reg"
    identparrun = subprocess.run([this_model_identpar, reg_pest_name, str(n_singular_values), 
                                  "null", "null", "ident10.dat"],
                              stdout=subprocess.PIPE, universal_newlines=True) # Run command 
    #
    print(identparrun.stdout)
        
    ### First need to run identpar with PEST
    # Copy and paste macro identpar then run
    # identpar UB_08_a_reg 10 null null ident10.dat   
    
    pal = cm.rainbow_r(np.linspace(0,1,10))
    
    iden_file_name = "ident10.dat"
    
    identif_df = pd.read_csv(iden_file_name, header=0,
                                delim_whitespace=True, encoding='utf-8-sig') 
    # Set index of identif df
    identif_df.index = identif_df.parameter
    
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    identif_df.iloc[:, 0:-2].plot(kind = "bar", ax=ax, stacked = True, colormap="rainbow") #, colors=pal
    ax.set_title("Identifiability of phantom faults", fontsize=14)
    ax.set_xlabel("Parameter number", fontsize=14)
    ax.set_ylabel("Parameter Identifiability", fontsize=14)
    
    fig.tight_layout(pad=1)    
    plt.savefig(os.path.join(figureDirectory, "ph_identifiability"))  

    
    # Add identifiability statistic to the phantom df 
    identif_df = identif_df.drop("hkaq", axis=0) # Drop aquifer just in case
    phantom_df["identif"] = identif_df["identifiability"]
    
    # Plot scatter plot of identifiability versus hk of phantom fault----------

    ax = phantom_df.plot.scatter(x="K_posterior", y="identif", s=50, 
                                            lw = 0)
    ax.set_xlabel("Posterior K value [m/d]", fontsize=14)
    ax.set_ylabel("Parameter identifiability", fontsize=14)
    ax.axvline(x=1e-3, color="0.2", ls="--")
    axes=plt.gca()
    axes.set_xlim([.1e-10,5.1])    
    axes.set_ylim([0,1.1])  
    axes.set_xscale("log")
    axes.grid()
    
    plt.savefig(os.path.join(figureDirectory, "ph_identif_vs_K"))  
    
    # SPATIAL PLOT OF PARAMETER IDENTIFIABILITY -------------------------------
    
    identif_colour = cm.jet(np.linspace(0,1,5))
    identif_leg_value = ["0 - 0.2", "0.2 - 0.4", "0.4 - 0.6", 
                      "0.6 - 0.8", "0.8 - 1"]  
    color = cm.jet(np.linspace(0,1,101))

    plt.figure(figsize=(20, 10))
    plt.title("Identifiability of phantom fault", fontsize=14)
    for i in range(len(ph_fa_rows_list)):
        identif_val = phantom_df["identif"][i]*1
        identif_color = color[int(identif_val*100)] 
        print("Ident value is: %5f" %identif_val)
        plt.plot(ph_fa_cols_list[i], ph_fa_rows_list[i], lw = 3, 
                 color = identif_color)
        
        # Now plot real fault --> this is set up to just plot one real fault
    if include_real_with_ph == True:
        val = phantom_df.loc["hkre_001", "identif"]*1
        val_color = color[int(val*100)] 
        plt.plot(re_fa_cols_list[0], re_fa_rows_list[0], color=val_color, lw = 3)
    else:
        pass    
        
        # Legend
    for i in range(len(identif_leg_value)):
        plt.plot([-10, -15], [-10, -15], lw=3, color=identif_colour[i], 
                 label=identif_leg_value[i])
        
    plt.xlabel("Columns", fontsize=14)
    plt.ylabel("Rows", fontsize=14)        
    
    axes = plt.gca()
    axes.set_xlim([0, param_dict["ncol"]])                
    axes.set_ylim([0, param_dict["nrow"]]) 

    plt.legend(loc=4, fontsize=14) # legend_list  
    plt.savefig(os.path.join(figureDirectory, "ph_identifiability"))  
      
    
   
    ######################## ######################## ########################
    # Phantom fault orientations and lengths #####################
    ######################## ######################## ########################
    alpha_vals, len_hypotenuse_vals = functions_proj_3.barrierPP.getAngleAndLength(
                                    n_phantom_faults, ph_fa_rows_list, ph_fa_cols_list,
                                    param_dict["delr"], param_dict["delc"])  



    phantom_df["orientation"] = alpha_vals
    phantom_df["length"] = len_hypotenuse_vals
    
    
    # Get the orientation and lengths of the real fault
    
    alpha_vals_r, len_hypotenuse_vals_r = functions_proj_3.barrierPP.getAngleAndLength(
                                    n_real_faults, re_fa_rows_list, re_fa_cols_list,
                                    param_dict["delr"], param_dict["delc"])  

    # Calculate mid-point of faults --> note that these are the mid of the rows/cols ACTUALLY ON THE GRID
    # If I just used start/end values some might be off the grid
    mid_value_r = []
    mid_value_c = []
    middle_values_ph = []

    for i in range(len(ph_fa_rows_list)):
        middle_val_r = ph_fa_rows_list[i][int(len(ph_fa_rows_list[i])/2)]
        mid_value_r.append(middle_val_r)
        middle_val_c = ph_fa_cols_list[i][int(len(ph_fa_cols_list[i])/2)]
        mid_value_c.append(middle_val_c)
        middle_values_ph.append([middle_val_r, middle_val_c])    
        
    phantom_df["mid_row"] = mid_value_r
    phantom_df["y_mid_loc"] = phantom_df["mid_row"]*param_dict["delc"]
    phantom_df["mid_col"] = mid_value_c
    phantom_df["x_mid_loc"] = phantom_df["mid_col"]*param_dict["delr"]

   # SPATIALLY PLOT PHANTOM FAULT ORIENTATIONS --------------------------------
    
    # Orientations legend 
    or_leg_colour = cm.rainbow(np.linspace(0,1,6))
    or_leg_value = [r"0 - 30$^\circ$", r"30 - 60$^\circ$", r"60 - 90$^\circ$", 
                    r"90 - 120$^\circ$", r"120 - 150$^\circ$", r"150 - 180$^\circ$"]    
    
    color=cm.rainbow(np.linspace(0,1,181))
    plt.figure(figsize=(20, 10))
    plt.title(r"Phantom orientations [$^\circ$]", fontsize=14)
    for i in range(len(ph_fa_rows_list)):
        orientat_val = phantom_df["orientation"][i]
        orient_color = color[int(orientat_val)] 
        print("Orientation value is: %d" %orientat_val)
        plt.plot(ph_fa_cols_list[i], ph_fa_rows_list[i], lw = 3, 
                 color = orient_color, label='_nolegend_')

    plt.xlabel("Columns", fontsize=14)
    plt.ylabel("Rows", fontsize=14)
    
    # Legend
    for i in range(len(or_leg_colour)):
        plt.plot([-10, -15], [-10, -15], lw=3, color=or_leg_colour[i], label=or_leg_value[i])
      
    axes = plt.gca()
    axes.set_xlim([0, param_dict["ncol"]])                
    axes.set_ylim([0, param_dict["nrow"]])                
    
    plt.legend(loc=4, fontsize=14) # legend_list  
    plt.savefig(os.path.join(figureDirectory, "ph_orientations"))  
    
    # Plot orientation vs K posterior
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)
    phantom_df.plot.scatter(ax=ax, y="K_posterior", x="orientation", marker="+", lw=3, s=250, c="identif", cmap="rainbow")
    axes = plt.gca()
    axes.set_yscale("log")
    axes.set_ylim([(phantom_df["K_posterior"].min()-0.001), 1.1])
    axes.set_xlim([0, 180])
    axes.grid()
    plt.xlabel("Strike orientation of fault [degr]")
    plt.ylabel("K posterior [m]")
    f = plt.gcf()
    f.get_axes() # First axes contains the plot, second aces contains the colorbar
    cax = f.get_axes()[1]
    cax.set_ylabel("Identifiability")
    plt.savefig(os.path.join(figureDirectory, "orient_v_Kpost"))    
    
        # Plot orientation vs length
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)
    phantom_df.plot.scatter(ax=ax, y="length", x="orientation", marker="+", lw=3, s=250, c="identif", cmap="rainbow")
    axes = plt.gca()
    axes.set_ylim([0, 10000])
    axes.set_xlim([0, 180])
    axes.grid()
    plt.xlabel("Strike orientation of fault [degr]")
    plt.ylabel("Fault length [m]")
    f = plt.gcf()
    f.get_axes() # First axes contains the plot, second aces contains the colorbar
    cax = f.get_axes()[1]
    cax.set_ylabel("Identifiability")
    plt.savefig(os.path.join(figureDirectory, "orient_v_length"))

            # Plot orientation vs length - normalised
    phantom_df["orientation_norm"] = np.abs(90 - phantom_df["orientation"])
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)
    phantom_df.plot.scatter(ax=ax, y="length", x="orientation_norm", marker="+", lw=3, s=250, c="identif", cmap="rainbow")
    axes = plt.gca()
    axes.set_ylim([0, 10000])
    axes.set_xlim([0, 90])
    axes.grid()
    plt.xlabel("Strike orientation of fault normalised [degr]")
    plt.ylabel("Fault length [m]")
    f = plt.gcf()
    f.get_axes() # First axes contains the plot, second aces contains the colorbar
    cax = f.get_axes()[1]
    cax.set_ylabel("Identifiability")
    plt.savefig(os.path.join(figureDirectory, "orient_v_length_norm"))
    
    # Plot y location vs normalised strike orientation
    phantom_df["orientation_norm"] = np.abs(90 - phantom_df["orientation"])
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)
    phantom_df.plot.scatter(ax=ax, y="y_mid_loc", x="length", marker="+", lw=3, s=250, c="K_posterior", cmap="jet")
    axes = plt.gca()
    axes.set_ylim([0, param_dict["Ly"]])
    axes.set_xlim([0, 10000])
    axes.grid()
    plt.xlabel("Length [m]")
    plt.ylabel("Midpoint location of fault [m]")
    f = plt.gcf()
    f.get_axes() # First axes contains the plot, second aces contains the colorbar
    cax = f.get_axes()[1]
    cax.set_ylabel("K posterior")
    plt.savefig(os.path.join(figureDirectory, "y_vs_length"))
    
   # Plot x location vs normalised strike orientation
    phantom_df["orientation_norm"] = np.abs(90 - phantom_df["orientation"])
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)
    phantom_df.plot.scatter(ax=ax, y="length", x="x_mid_loc", marker="+", lw=3, s=250, c="K_posterior", cmap="jet")
    axes = plt.gca()
    axes.set_ylim([0, 10000])
    axes.set_xlim([0, param_dict["Lx"]])
    axes.grid()
    plt.xlabel("Midpoint location of fault [m]")
    plt.ylabel("Length [m]")
    f = plt.gcf()
    f.get_axes() # First axes contains the plot, second aces contains the colorbar
    cax = f.get_axes()[1]
    cax.set_ylabel("K posterior")
    plt.savefig(os.path.join(figureDirectory, "x_vs_length"))
       
    
# ORIENTATION - ROSE PLOT --> BASED ON THE K POSTERIOR
    
    plt.figure(figsize=(5, 8))
    plt.suptitle("Line length depends on K posterior")
    for idx in phantom_df.index:   
    
        orn = phantom_df.loc[idx, "orientation"]
    
        hypotenuse = np.abs(phantom_df.loc[idx, "hk_real_diff_log_aq"])
        
        if orn > 90:
            x_add = hypotenuse*math.sin(np.deg2rad(orn-90))
            y_add = hypotenuse*math.cos(np.deg2rad(orn-90))
            add_or_subtract = -1
        else:
            x_add = hypotenuse*math.sin(np.deg2rad(orn))
            y_add = hypotenuse*math.cos(np.deg2rad(orn))
            add_or_subtract = 1
        
        # Colour based on identifiability
        if phantom_df.loc[idx, "identif"] < 0.8:
            col = "0.5"
        else:
            col = "dodgerblue"
            
        plt.plot([0, x_add], [0, y_add*add_or_subtract], lw=20, color=col, alpha=0.5)
    axes = plt.gca()
    axes.grid()
    plt.ylabel("Orientation of fault [degr]")
    plt.tight_layout()
        
    plt.plot([0, 0], [0, 0], color="dodgerblue", label="i > 0.8", lw=2)
    plt.plot([0, 0], [0, 0], color="0.5", label="i < 0.8", lw=2)
    plt.legend()
    plt.savefig(os.path.join(figureDirectory, "rose_Kposterior"))
        
# ORIENTATION - ROSE PLOT --> BASED ON THE LENGTH
    
    plt.figure(figsize=(5, 8))
    plt.suptitle("Line length depends on length")
    for idx in phantom_df.index:   
    
        orn = phantom_df.loc[idx, "orientation"]
    
        hypotenuse = np.abs(phantom_df.loc[idx, "length"])
        
        if orn > 90:
            x_add = hypotenuse*math.sin(np.deg2rad(orn-90))
            y_add = hypotenuse*math.cos(np.deg2rad(orn-90))
            add_or_subtract = -1
        else:
            x_add = hypotenuse*math.sin(np.deg2rad(orn))
            y_add = hypotenuse*math.cos(np.deg2rad(orn))
            add_or_subtract = 1
        
        # Colour based on identifiability
        if phantom_df.loc[idx, "identif"] < 0.8:
            col = "0.5"
        else:
            col = "dodgerblue"
            
        plt.plot([0, x_add], [0, y_add*add_or_subtract], lw=20, color=col, alpha=0.5, label="_nolegend_")
        
    axes = plt.gca()
    axes.grid()
    plt.ylabel("Orientation of fault [degr]")
    plt.tight_layout()
        
    plt.plot([0, 0], [0, 0], color="dodgerblue", label="i > 0.8", lw=2)
    plt.plot([0, 0], [0, 0], color="0.5", label="i < 0.8", lw=2)
    plt.legend()
    plt.savefig(os.path.join(figureDirectory, "rose_length"))

    
    

 # SPATIALLY PLOT LENGTH ------------------------------------------------------
    max_length = np.max(phantom_df["length"])
    
        # Length legend
    len_leg_colour = cm.rainbow(np.linspace(0,1,6))
    a = np.linspace(0,max_length+1,7)
    print("Value is %2d" %a[5])
    len_leg_value = ["0 - %2d m" %a[1], "%2d - %2d m" %(a[1], a[2]), 
                     "%2d - %2d m" %(a[2], a[3]), "%2d - %2d m" %(a[3], a[4]),
                    "%2d - %2d m" %(a[4], a[5]), "%2d - %2d m" %(a[5], a[6])]    
    
       
    color=cm.rainbow(np.linspace(0,1,int(max_length)+1))
    plt.figure(figsize=(20, 10))
    plt.title("Length of barrier [m]", fontsize=14)
    for i in range(len(ph_fa_rows_list)):
        length_val = phantom_df["length"][i]
        length_color = color[int(length_val)] 
        print("Length value is: %d" %length_val)
        plt.plot(ph_fa_cols_list[i], ph_fa_rows_list[i], lw = 3, 
                 color = length_color)

    plt.xlabel("Columns", fontsize=14)
    plt.ylabel("Rows", fontsize=14)
    
    # Legend
    for i in range(len(len_leg_value)):
        plt.plot([-10, -15], [-10, -15], lw=3, color=len_leg_colour[i], label=len_leg_value[i])

      
    axes = plt.gca()
    axes.set_xlim([0, param_dict["ncol"]])                
    axes.set_ylim([0, param_dict["nrow"]])                
    
    plt.legend(loc=4, fontsize=14) # legend_list  
    plt.savefig(os.path.join(figureDirectory, "ph_length"))    

    # PLOT LENGTH VERSUS K POSTERIOR ------------------------------------------
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)
    phantom_df.plot.scatter(ax=ax, y="K_posterior", x="length", marker="+", lw=3, s=250, c="identif", cmap="rainbow")
    axes = plt.gca()
    axes.set_yscale("log")
    axes.set_ylim([(phantom_df["K_posterior"].min()*0.5), 1.1])
    axes.set_xlim([-10, phantom_df["length"].max()])
    axes.grid()
    plt.xlabel("Length of fault [m]")
    plt.ylabel("K posterior [m]")
    f = plt.gcf()
    f.get_axes() # First axes contains the plot, second aces contains the colorbar
    cax = f.get_axes()[1]
    cax.set_ylabel("Identifiability")
    plt.savefig(os.path.join(figureDirectory, "length_v_Kpost"))    
    
    
    # PLOT FAULT LOCATION (MID-POINT) VERSUS K POSTERIOR ----------------------
    


    # Plot midpoint in y direction (row)
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)
    phantom_df.plot.scatter(ax=ax, y="K_posterior", x="y_mid_loc", marker="+", lw=3, s=250, c="orientation", cmap="Spectral")
    axes = plt.gca()
    axes.set_yscale("log")
    axes.set_ylim([(phantom_df["K_posterior"].min()*0.5), 1.1])
    axes.set_xlim([-10, param_dict["Ly"]+10])
    axes.grid()
    plt.xlabel("Mid location of fault in y direction (row) [m]")
    plt.ylabel("K posterior [m]")
    f = plt.gcf()
    f.get_axes() # First axes contains the plot, second aces contains the colorbar
    cax = f.get_axes()[1]
    cax.set_ylabel("Orientation")
    plt.savefig(os.path.join(figureDirectory, "mid_y_v_Kpost"))    
    
    # Plot midpoint in x direction (column)
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)
    phantom_df.plot.scatter(ax=ax, y="K_posterior", x="x_mid_loc", marker="+", lw=3, s=250, c="orientation", cmap="Spectral")
    axes = plt.gca()
    axes.set_yscale("log")
    axes.set_ylim([(phantom_df["K_posterior"].min()*0.5), 1.1])
    axes.set_xlim([-10, param_dict["Lx"]+10])
    axes.grid()
    plt.xlabel("Mid location of fault in x direction (column) [m]")
    plt.ylabel("K posterior [m]")
    f = plt.gcf()
    f.get_axes() # First axes contains the plot, second aces contains the colorbar
    cax = f.get_axes()[1]
    cax.set_ylabel("Orientation")
    plt.savefig(os.path.join(figureDirectory, "mid_x_v_Kpost"))    
    
    #--------------------------------------------------------------------------
    # Difference in orientation...

    phantom_df['diff_in_orientations'] = [np.nan] * len(phantom_df.index)
    real_fault_orientation = 45
    for idx in phantom_df.index:
        print(idx)
        orn = phantom_df.loc[idx, "orientation"]
        print(orn)
        diff_orn = real_fault_orientation - orn
        phantom_df.loc[idx, 'diff_in_orientations'] = abs(diff_orn)
        
   
    # PLOT PHANTOM FAULT ORIENTATIONS -----------------------------------------
    
    # Orientations legend 
    or_leg_colour = cm.rainbow(np.linspace(0,1,6))
    or_leg_value = [r"0 - 30$^\circ$", r"30 - 60$^\circ$", r"60 - 90$^\circ$", 
                    r"90 - 120$^\circ$", r"120 - 150$^\circ$", r"150 - 180$^\circ$"]    
    
    color=cm.rainbow(np.linspace(0,1,181))
    plt.figure(figsize=(20, 10))
    plt.title(r"Phantom orientations [$^\circ$]", fontsize=14)
    for i in range(len(ph_fa_rows_list)):
        orientat_val = phantom_df["orientation"][i]
        orient_color = color[int(orientat_val)] 
        print("Orientation value is: %d" %orientat_val)
        plt.plot(ph_fa_cols_list[i], ph_fa_rows_list[i], lw = 3, 
                 color = orient_color, label='_nolegend_')

    plt.xlabel("Columns", fontsize=14)
    plt.ylabel("Rows", fontsize=14)
    
    # Legend
    for i in range(len(or_leg_colour)):
        plt.plot([-10, -15], [-10, -15], lw=3, color=or_leg_colour[i], label=or_leg_value[i])
      
    axes = plt.gca()
    axes.set_xlim([0, 200])                
    axes.set_ylim([0, 100])                
    
    plt.legend(loc=4, fontsize=14) # legend_list  
    plt.savefig(os.path.join(figureDirectory, "ph_orientations"))    
     
    # PLOT DIFFERENCE IN ORIENTATION ------------------------------------------
  
    # PLOT LENGTH -------------------------------------------------------------
   
 
    # Phantom fault distance away from real fault -----------------------------

    # Calculate the distance that the phantom fault is away from the real fault
    middle_values_re = [] # row, col

    for i in range(len(re_fa_rows_list)):
        middle_val_r = re_fa_rows_list[i][int(len(re_fa_rows_list[i])/2)]
        middle_val_c = re_fa_cols_list[i][int(len(re_fa_cols_list[i])/2)]
        middle_values_re.append([middle_val_r, middle_val_c])
    
    mid_value_r = []
    mid_value_c = []
    middle_values_ph = []

    for i in range(len(ph_fa_rows_list)):
        middle_val_r = ph_fa_rows_list[i][int(len(ph_fa_rows_list[i])/2)]
        mid_value_r.append(middle_val_r)
        middle_val_c = ph_fa_cols_list[i][int(len(ph_fa_cols_list[i])/2)]
        mid_value_c.append(middle_val_c)
        middle_values_ph.append([middle_val_r, middle_val_c])
        
    phantom_df["mid_value_r"] = mid_value_r
    phantom_df["mid_value_c"] = mid_value_c

    distance_from_real = []
    real_fault_idx = 0
    for fault_n in range(n_phantom_faults):
        x_distance = param_dict["delr"]*abs(middle_values_re[real_fault_idx][0] - middle_values_ph[fault_n][0])
        y_distance = param_dict["delc"]*abs(middle_values_re[real_fault_idx][1] - middle_values_ph[fault_n][1])
        c_distance = math.sqrt(x_distance**2 + y_distance**2)
        distance_from_real.append(c_distance)
    
    phantom_df["distance"] = distance_from_real

    # PLOT DISTANCE FROM THE PHANTOM BARRIER TO THE REAL BARRIER.
    # Legend
    dist_leg_colour = cm.rainbow(np.linspace(0,1,6))
    dist_leg_value = ["0 - 2000 m", "2000 - 4000 m", "4000 - 6000 m", 
                      "6000 - 8000 m", "8000 - 12000 m", "> 12000 m"]  
    
    
    max_length = np.max(phantom_df["distance"])   
    color=cm.rainbow(np.linspace(0,1,int(max_length)+1))
    plt.figure(figsize=(20, 10))
    plt.title("Distance from real fault", fontsize=14)
    for i in range(len(ph_fa_rows_list)):
        length_val = phantom_df["distance"][i]
        length_color = color[int(length_val)] 
        print("Length value is: %d" %length_val)
        plt.plot(ph_fa_cols_list[i], ph_fa_rows_list[i], lw = 3, 
                 color = length_color)

    plt.xlabel("Columns", fontsize=14)
    plt.ylabel("Rows", fontsize=14)
    
    # Legend
    for i in range(len(dist_leg_value)):
        plt.plot([-10, -15], [-10, -15], lw=3, color=dist_leg_colour[i], label=dist_leg_value[i])
      
    axes = plt.gca()
    axes.set_xlim([0, 200])                
    axes.set_ylim([0, 100])                
    
    
    plt.legend(loc=4, fontsize=14) # legend_list  
    plt.savefig(os.path.join(figureDirectory, "ph_distance"))  
    
    # Parameter Composite Scaled Sensitivity (CSS) ----------------------------
    
    # Do this section
    # I am confused about this...not sure which value to use. Whether the values
    # after the "Completion of Optimisation Process" are those affiliated
    # with the "best" iteration.
    # OR do I need to complete new sensitivities with NOPTMAX = -1 as per pest manual pg. 133
    
    phantom_df["length_diff"] = abs(length_real_barrier - phantom_df["length"])
    phantom_df["metric"] = [np.nan]*len(phantom_df.index)
    #--------------------------------------------------------------------------
    # Calculating an index to assess the success of the fault placement
    
    w_dist = .001 # Weight for distance
    w_orient = .1 # Weight for orientation/angle/alpha
    w_len = .001 # Weight got length
    w_K = 1. # Weight for K value

    # idx=phantom_df.index[37]
    for idx in phantom_df.index: #[37, 74, 84, 85, 61, 24]:
        # idx = phantom_df.index[i]
        metric = ((w_dist*phantom_df.loc[idx, "distance"] + 
                   w_orient*phantom_df.loc[idx, "diff_in_orientations"] + 
                  w_len*phantom_df.loc[idx, "length_diff"] + 
                  w_K*abs(phantom_df.loc[idx, "hk_real_diff_log"])
                  )) #*abs(phantom_df.loc[idx, "hk_real_diff_log_aq"])*w_len*(
                  #phantom_df.loc[idx, "length"]))
        print(metric)
        phantom_df.loc[idx, "metric"] = metric
    
    phantom_df.plot.scatter("K_posterior", "metric", c="distance", colormap='viridis')
    plt.savefig(os.path.join(figureDirectory, "metric_vs_K_posterior")) 
    
    #-------------------------------------------------------------------------
    # Plot metric
    
    # Legend
    metric_leg_colour = cm.plasma(np.linspace(0,1,10))
    
    min_value = phantom_df["metric"].min()
    max_value = phantom_df["metric"].max()
    
    metric_scale = np.linspace(min_value, max_value, 10)
    
    leg_value = []
    for i in range(len(metric_scale)):
        leg_value.append("> %2.2d" %metric_scale[i])

  
    plt.figure(figsize=(20, 10))
    plt.title("Fault metric", fontsize=14)
    i=0
    for i in range(len(ph_fa_rows_list)):
        

        metric_val = phantom_df["metric"][i]
        print("Metric value is %d" %metric_val)        
        if metric_val >= metric_scale[0] and metric_val < metric_scale[1]:
            c = metric_leg_colour[0]
            print("c=0")
            
        elif metric_val >= metric_scale[1] and metric_val < metric_scale[2]:
            c = metric_leg_colour[1]
            print("c=1")
        elif metric_val >= metric_scale[2] and metric_val < metric_scale[3]:
            c = metric_leg_colour[2]
            print("c=2")
        elif metric_val >= metric_scale[3] and metric_val < metric_scale[4]:
            c = metric_leg_colour[3]    
            print("c=3")
        elif metric_val >= metric_scale[4] and metric_val < metric_scale[5]:
            c = metric_leg_colour[4]
            print("c=4")
        elif metric_val >= metric_scale[5] and metric_val < metric_scale[6]:
            c = metric_leg_colour[5]
            print("c=5")
        elif metric_val >= metric_scale[6] and metric_val < metric_scale[7]:
            c = metric_leg_colour[6]
            print("c=6")
        elif metric_val >= metric_scale[7] and metric_val < metric_scale[8]:
            c = metric_leg_colour[7]
            print("c=7")
        elif metric_val >= metric_scale[8] and metric_val < metric_scale[9]:
            c = metric_leg_colour[8]
            print("c=8")
        else:
            c = metric_leg_colour[9]
            print("c=9")

        plt.plot(ph_fa_cols_list[i], ph_fa_rows_list[i], lw = 3, 
                 color = c)

    plt.xlabel("Columns", fontsize=14)
    plt.ylabel("Rows", fontsize=14)
    
    # Legend
    for i in range(len(metric_leg_colour)):
        plt.plot([-10, -15], [-10, -15], lw=3, color=metric_leg_colour[i], 
                 label=leg_value[i])
      
    axes = plt.gca()
    axes.set_xlim([0, 200])                
    axes.set_ylim([0, 100])                
    
    
    plt.legend(loc=4, fontsize=14, framealpha=1.0, facecolor='w', edgecolor="k") 

    plt.savefig(os.path.join(figureDirectory, "ph_metric"))  
    
    # -------------------------------------------------------------------------
    # % & # % & # % & # % & # % & # % & # % & # % & # % & # % & # % & # % & # %     
    # Make relevant dataframes     --------------------------------------------
    
    i_level = 0.8 # What value consititutes a parameter that is "identifiable"
                  # Note this is arbitrary, see Doherty and Hunt (2009)
    
    # Dataframe where faults are identifiable
    phantom_df_identifiable = phantom_df.copy()
    phantom_df_identifiable = phantom_df_identifiable[phantom_df_identifiable["identif"]>i_level]
    
    # Dataframe where faults are barriers only
    phantom_df_barriers = phantom_df.copy()
    phantom_df_barriers = phantom_df_barriers[round(phantom_df_barriers['K_posterior'], 2)<1]
    # I included the round so that faults with K of 0.999 would be removed.

    # Dataframe where faults are barriers only AND are identifiable
    phantom_df_barr_ident = phantom_df_barriers.copy()
    phantom_df_barr_ident = phantom_df_barr_ident[phantom_df_barr_ident["identif"]>i_level]


    # % & # % & # % & # % & # % & # % & # % & # % & # % & # % & # % & # % & # %     
    # -------------------------------------------------------------------------

    #################################
    plt.close("all")
    #################################
    
 # Plot which faults are barriers ------------------------------------------    

    plt.figure(figsize=(20, 10))
    plt.title("Phantom Faults that are barriers", fontsize=14)
    c1 = "wheat"
    c2 = "darkred"
    cols = [c1, c2]
    col_labs = ["Aquifer hk", "Barrier hk"]
    for i in range(len(all_cols_dict)):
        if (phantom_df.index[i] in phantom_df_barriers.index) == True:
            colour_i = c2
            print(phantom_df.index[i])
        else:
            colour_i = c1
        plt.plot(all_cols_dict[phantom_df.index[i]], all_rows_dict[phantom_df.index[i]], 
                 lw = 3, color = colour_i)
     
        # Legend
    for i in range(2):
        plt.plot([-10, -15], [-10, -15], lw=3, color=cols[i], 
                 label=col_labs[i])
        
    plt.xlabel("Columns", fontsize=14)
    plt.ylabel("Rows", fontsize=14)        
    
    axes = plt.gca()
    axes.set_xlim([0, param_dict["ncol"]])                
    axes.set_ylim([0, param_dict["nrow"]]) 

    legend = plt.legend(loc=4, fontsize=14) # legend_list
    frame = legend.get_frame()
    frame.set_facecolor("white")
    plt.savefig(os.path.join(figureDirectory, "ph_barriers_only")) 
           
    # -------------------------------------------------------------------------  
    # Plot which faults are identifiable
    plt.figure(figsize=(20, 10))
    plt.title("KEY Identifiabilities of phantom fault", fontsize=14)
    c1 = "wheat"
    c2 = "dodgerblue"
    cols = [c1, c2]
    col_labs = ["Not identified", "Identified"]
    for i in range(len(all_cols_dict)):
        if (phantom_df.index[i] in phantom_df_identifiable.index) == True:
            colour_i = c2
        else:
            colour_i = c1
        print("Ident value is: %5f" %identif_val)
        plt.plot(all_cols_dict[phantom_df.index[i]], all_rows_dict[phantom_df.index[i]], 
                 lw = 3, color = colour_i)
     
        # Legend
    for i in range(2):
        plt.plot([-10, -15], [-10, -15], lw=3, color=cols[i], 
                 label=col_labs[i])
        
    plt.xlabel("Columns", fontsize=14)
    plt.ylabel("Rows", fontsize=14)        
    
    axes = plt.gca()
    axes.set_xlim([0, param_dict["ncol"]])                
    axes.set_ylim([0, param_dict["nrow"]]) 

    legend = plt.legend(loc=4, fontsize=14) # legend_list
    frame = legend.get_frame()
    frame.set_facecolor("white")
    plt.savefig(os.path.join(figureDirectory, "ph_identifiability_KEY"))  
    
   # -------------------------------------------------------------------------      
   # Plot the hydraulic conductivity values of identifiable faults.
    if len(phantom_df_identifiable.index) > 0:
     
        min_value = np.floor(np.log10(phantom_df_identifiable["K_posterior"].min()))
        max_value = np.ceil(np.log10(phantom_df_identifiable["K_posterior"].max()))
        range_values = max_value-min_value 
        ranges = np.linspace(int(min_value),int(max_value),int(range_values)+1)
        
        K_leg_value = []
        for i in ranges:
            K_leg_value.append("$\geqslant$ %d" %i)
                          
        color = cm.jet(np.linspace(0,1,len(ranges)))
    
        plt.figure(figsize=(20, 10))
        plt.title("K of identifiable phantom faults", fontsize=14)
        
        for i in range(len(ph_fa_rows_list)):
            if (phantom_df.index[i] in phantom_df_identifiable.index) == True:
                K_val = np.floor(np.log10((phantom_df["K_posterior"][i])))
                K_col = color[np.where(ranges==K_val)[0][0]]
                K_leg = K_leg_value[np.where(ranges==K_val)[0][0]]
                                    
            else:
                K_col = "0.9"
                
            plt.plot(all_cols_dict[phantom_df.index[i]], all_rows_dict[phantom_df.index[i]],  
                     lw = 3, color = K_col)
        
                # Legend
        for i in range(len(K_leg_value)):
            plt.plot([-10, -15], [-10, -15], lw=3, color=color[i], 
                     label= K_leg_value[i])
        
        axes = plt.gca()
        axes.set_xlim([0, param_dict["ncol"]])                
        axes.set_ylim([0, param_dict["nrow"]]) 
    
        legend = plt.legend(loc=4, fontsize=14) # legend_list
        frame = legend.get_frame()
        frame.set_facecolor("white")
        plt.savefig(os.path.join(figureDirectory, "ph_K_identified")) 
        
        # -------------------------------------------------------------------------          
        # Plot the identifiability of these faults that are identifiable
        
        identif_colour = cm.jet(np.linspace(0,1,10))
        identif_leg_value = ["0 - 0.1", "0.1 - 0.2", "0.2 - 0.3", "0.3 - 0.4",
                             "0.4 - 0.5", "0.5 - 0.6", "0.6 - 0.7", "0.7 - 0.8",
                             "0.8 - 0.9", "0.9 - 1.0"]  
        color = cm.jet(np.linspace(0,1,101))
    
        plt.figure(figsize=(20, 10))
        plt.title("Identifiability of phantom fault - emphasised selection only", 
                  fontsize=14)
        for i in range(len(ph_fa_rows_list)):
            identif_val = phantom_df["identif"][i]*1
            if (phantom_df.index[i] in phantom_df_identifiable.index) == True:
    
                identif_color = color[int(identif_val*100)] 
                print("Ident value is: %5f" %identif_val)
                
            else:
                identif_color = "0.9"
            plt.plot(all_cols_dict[phantom_df.index[i]], all_rows_dict[phantom_df.index[i]],  
                     lw = 3, color = identif_color)
            
        # Legend
        for i in range(len(identif_leg_value)):
            plt.plot([-10, -15], [-10, -15], lw=3, color=identif_colour[i], 
                     label=identif_leg_value[i])
            
        plt.xlabel("Columns", fontsize=14)
        plt.ylabel("Rows", fontsize=14)        
        
        axes = plt.gca()
        axes.set_xlim([0, param_dict["ncol"]])                
        axes.set_ylim([0, param_dict["nrow"]]) 
    
        plt.legend(loc=4, fontsize=14) # legend_list  
        plt.savefig(os.path.join(figureDirectory, "ph_identif_emphasis"))  
        
          
        # -------------------------------------------------------------------------
        # % & # % & # % & # % & # % & # % & # % & # % & # % & # % & # % & # % & # % 
        # What is my final selection of faults?
        # % & # % & # % & # % & # % & # % & # % & # % & # % & # % & # % & # % & # % 
        # -------------------------------------------------------------------------
    
        '''
        Do I just want to look at:
            (a) Faults that are barriers
            (b) Faults that are identifiable
            (c) Faults that are both barriers and are identifiable
        
        '''
    
        key_dataframe = phantom_df_barr_ident #phantom_df_barr_ident, phantom_df_barriers, phantom_df_identifiable
        
        n_ident_faults = len(key_dataframe.index) 
        print("The number of KEY phantom faults that exist is %d" % n_ident_faults)
    
        ph_key_fault_coordinates = np.copy(ph_fault_coordinates)
        row_index = []
        for index in key_dataframe.index:
            idx = int(index[-3:])-1
            row_index.append(idx)
        ph_key_fault_coordinates = ph_key_fault_coordinates[row_index, :]
        np.save(os.path.join(overarching_model_folder, "ph_fault_coords_key"), ph_key_fault_coordinates)
        
        # -------------------------------------------------------------------------      
       # Plot the hydraulic conductivity values of key faults.
        if len(key_dataframe.index) > 0:
            min_value = np.floor(np.log10(key_dataframe["K_posterior"].min()))
            max_value = np.ceil(np.log10(key_dataframe["K_posterior"].max()))
            range_values = max_value-min_value 
            ranges = np.linspace(int(min_value),int(max_value),int(range_values)+1)
            
            K_leg_value = []
            for i in ranges:
                K_leg_value.append("$\geqslant$ %d" %i)
                              
            color = cm.jet(np.linspace(0,1,len(ranges)))
        
            plt.figure(figsize=(20, 10))
            plt.title("K of identifiable phantom faults", fontsize=14)
            
            for i in range(len(ph_fa_rows_list)):
                if (phantom_df.index[i] in key_dataframe.index) == True:
                    K_val = np.floor(np.log10((phantom_df["K_posterior"][i])))
                    K_col = color[np.where(ranges==K_val)[0][0]]
                    K_leg = K_leg_value[np.where(ranges==K_val)[0][0]]
                                        
                else:
                    K_col = "0.9"
                    
                plt.plot(all_cols_dict[phantom_df.index[i]], all_rows_dict[phantom_df.index[i]],  
                         lw = 3, color = K_col)
            
                    # Legend
            for i in range(len(K_leg_value)):
                plt.plot([-10, -15], [-10, -15], lw=3, color=color[i], 
                         label= K_leg_value[i])
            
            axes = plt.gca()
            axes.set_xlim([0, param_dict["ncol"]])                
            axes.set_ylim([0, param_dict["nrow"]]) 
        
            legend = plt.legend(loc=4, fontsize=14) # legend_list
            frame = legend.get_frame()
            frame.set_facecolor("white")
            plt.savefig(os.path.join(figureDirectory, "ph_K_key")) 
        
        # -------------------------------------------------------------------------          
        # Plot the identifiability of key faults 
        
        identif_colour = cm.jet(np.linspace(0,1,10))
        identif_leg_value = ["0 - 0.1", "0.1 - 0.2", "0.2 - 0.3", "0.3 - 0.4",
                             "0.4 - 0.5", "0.5 - 0.6", "0.6 - 0.7", "0.7 - 0.8",
                             "0.8 - 0.9", "0.9 - 1.0"]  
        color = cm.jet(np.linspace(0,1,101))
    
        plt.figure(figsize=(20, 10))
        plt.title("Identifiability of phantom fault - emphasised selection only", 
                  fontsize=14)
        for i in range(len(ph_fa_rows_list)):
            identif_val = phantom_df["identif"][i]*1
            if (phantom_df.index[i] in key_dataframe.index) == True:
    
                identif_color = color[int(identif_val*100)] 
                print("Ident value is: %5f" %identif_val)
                
            else:
                identif_color = "0.9"
            plt.plot(all_cols_dict[phantom_df.index[i]], all_rows_dict[phantom_df.index[i]],  
                     lw = 3, color = identif_color)
            
        # Legend
        for i in range(len(identif_leg_value)):
            plt.plot([-10, -15], [-10, -15], lw=3, color=identif_colour[i], 
                     label=identif_leg_value[i])
            
        plt.xlabel("Columns", fontsize=14)
        plt.ylabel("Rows", fontsize=14)        
        
        axes = plt.gca()
        axes.set_xlim([0, param_dict["ncol"]])                
        axes.set_ylim([0, param_dict["nrow"]]) 
    
        plt.legend(loc=4, fontsize=14) # legend_list  
        plt.savefig(os.path.join(figureDirectory, "ph_identif_key"))  
        
          
        #-------------------------------------------------------------------------
        # Plot metric for key only 
        
        plt.figure(figsize=(20, 10))
        plt.title("Fault metric", fontsize=14)
        i=0
        for i in range(len(ph_fa_rows_list)):
            
            if (phantom_df.index[i] in phantom_df_identifiable.index) == True:
                metric_val = phantom_df["metric"][i]
                print("Metric value is %d" %metric_val)        
                if metric_val >= metric_scale[0] and metric_val < metric_scale[1]:
                    c = metric_leg_colour[0]
                    print("c=0")
                    
                elif metric_val >= metric_scale[1] and metric_val < metric_scale[2]:
                    c = metric_leg_colour[1]
                    print("c=1")
                elif metric_val >= metric_scale[2] and metric_val < metric_scale[3]:
                    c = metric_leg_colour[2]
                    print("c=2")
                elif metric_val >= metric_scale[3] and metric_val < metric_scale[4]:
                    c = metric_leg_colour[3]    
                    print("c=3")
                elif metric_val >= metric_scale[4] and metric_val < metric_scale[5]:
                    c = metric_leg_colour[4]
                    print("c=4")
                elif metric_val >= metric_scale[5] and metric_val < metric_scale[6]:
                    c = metric_leg_colour[5]
                    print("c=5")
                elif metric_val >= metric_scale[6] and metric_val < metric_scale[7]:
                    c = metric_leg_colour[6]
                    print("c=6")
                elif metric_val >= metric_scale[7] and metric_val < metric_scale[8]:
                    c = metric_leg_colour[7]
                    print("c=7")
                elif metric_val >= metric_scale[8] and metric_val < metric_scale[9]:
                    c = metric_leg_colour[8]
                    print("c=8")
                else:
                    c = metric_leg_colour[9]
                    print("c=9")
        
                
            else:
                c = "0.9"
                
            plt.plot(ph_fa_cols_list[i], ph_fa_rows_list[i], lw = 3, 
                         color = c)
            
        plt.xlabel("Columns", fontsize=14)
        plt.ylabel("Rows", fontsize=14)
        
        # Legend
        for i in range(len(metric_leg_colour)):
            plt.plot([-10, -15], [-10, -15], lw=3, color=metric_leg_colour[i], 
                     label=leg_value[i])
          
        axes = plt.gca()
        axes.set_xlim([0, 200])                
        axes.set_ylim([0, 100])                
        
        
        plt.legend(loc=4, fontsize=14, framealpha=1.0, facecolor='w', edgecolor="k") 
    
        plt.savefig(os.path.join(figureDirectory, "ph_metric_key"))
        
        #==============================================================================
        # Trying to test how many of the faults are required and how the simualtion is improved.
        #==============================================================================    
        
        # FIRST --> Look at individually how each fault improves the RMSE, is the simulation 
        # better with these faults in it as opposed to not in it?
        # Call each run with an individual fault (or no fault, or faults combos) the "Ranking"
    #    sort_values_by = "identif" # "K_posterior" or "identif"
    #    cum_or_not = "cum" # "cum" or "ncum"   
    
        cum_optn = "cum"
        sort_optn = "inv_identif"
        f=1
        # Order these in different ways:
        for cum_optn in ["cum", "ncum"]:
            cum_or_not = cum_optn
            print("Is this plot cumulative?: %s" %cum_or_not)
            for sort_optn in ["inv_identif", "K_posterior"]:
                sort_values_by = sort_optn
                print("Sorting by: %s" %sort_values_by)
    
      
                # Make HK array with each (cumularive) ranking and put in a list.    
                
                hk_array_rankings = []
                
                key_dataframe["inv_identif"] = 1 - key_dataframe["identif"]
                
                key_dataframe = key_dataframe.sort_values(by=[sort_values_by], axis=0)
                print("Different column being sorted by %s -- check carefully" %sort_values_by)
            
                
                # Create the hk arrays for each rank of model
                for f in range(len(key_dataframe)+1):
                    print(f)
                    print("Number of faults in this simulation is %d" %(f))
                    
                    # No faults in the first simulation
                    if f == 0:
                        hk_array_x = param_dict["hk_aquifer"] * np.ones((param_dict["nlay"], 
                                param_dict["nrow"], param_dict["ncol"]), dtype=np.float32)  
                        
                    else:
                        if cum_or_not == "ncum": 
                            print("Not cumulative")
                            x_faults = key_dataframe[key_dataframe.index == key_dataframe.index[f-1]]
                        elif cum_or_not == "cum": 
                            print("Cumulative")
                            x_faults = key_dataframe.nsmallest(f, sort_values_by, 
                                                                     keep="first") # keep is if there are duplicate values
                        else:
                            "You've made a mistake, cumulative or not??"
                         
                                                 
                        hk_array_x = param_dict["hk_aquifer"] * np.ones((param_dict["nlay"], 
                                                    param_dict["nrow"], param_dict["ncol"]), dtype=np.float32) 
                        
                        
                        
                        for index in x_faults.index:
                            print(index)
                            rows = all_rows_dict[index]
                            cols = all_cols_dict[index]
                            k_posterior = phantom_df.loc[index, "K_posterior"]
    
                            for ii in range(len(rows)):
                                hk_array_x[0, int(rows[ii]), int(cols[ii])] = k_posterior
                                            
                    # Append this array to list of ranked arrays
                    hk_array_rankings.append(hk_array_x)
                                               
                    # Check: plot HK array
                    plt_name = "hk_%s_rnk_%d" %(sort_values_by, f) 
            
                    hk_array_x_log = np.log10(hk_array_x)
                    plt.figure()
                    plt.title(plt_name)
                    plt.subplot(1, 1, 1, aspect='equal')    
                
                    plt.imshow(np.flipud(hk_array_x_log[0, :, :]), extent=extent, cmap=hk_array_cmap,
                                         vmin = -5, vmax = 5) 
                    cbr = plt.colorbar()
                    cbr.set_label('Log hk (m/d')
                
                    plt.savefig(os.path.join(figureDirectory, plt_name))
                    plt.close("all")
                    
      # -----------------------------------------------------------------------------
            # What is the K array for the final posterior array?
                phantom_df.index
                            
                hk_array_posterior = param_dict["hk_aquifer"] * np.ones((param_dict["nlay"], 
                            param_dict["nrow"], param_dict["ncol"]), dtype=np.float32) 
                
                for index in phantom_df.index:
                    rows = all_rows_dict[index]
                    cols = all_cols_dict[index]
                    k_posterior = phantom_df.loc[index, "K_posterior"]
    
                    for ii in range(len(rows)):
                        hk_array_posterior[0, int(rows[ii]), int(cols[ii])] = k_posterior               
                    
            
                # Check: plot HK array
                hk_array_posterior_log = np.log10(hk_array_posterior)
                plt.figure()
                plt.subplot(1, 1, 1, aspect='equal')    
            
                plt.imshow(np.flipud(hk_array_posterior_log[0, :, :]), extent=extent, cmap=hk_array_cmap,
                                     vmin = -4, vmax = 4) 
                cbr = plt.colorbar()
                cbr.set_label('Log hk [m/d]')
            
                plt_name = "hk_array_posterior" 
                plt.savefig(os.path.join(figureDirectory, plt_name))
                
                # Compare this HK array (array_2) to final hk array I took from PEST (array_1), just to see
                # what the difference is
                
                #Giving the option to do log or not log (first is log)
                array_1 = log_interp_pest_hk # log_interp_pest_hk, hk_array_pest_log
                array_2 = hk_array_posterior_log # hk_array_posterior_log, hk_array_posterior
                
                diff_hk_test = array_2[0,:,:] - array_1
                plt.figure()
                plt.title('Difference between final PEST and optimised values')
                plt.subplot(1, 1, 1, aspect="equal")
                
                plt.imshow(np.flipud(diff_hk_test[:,:]), extent=extent, cmap='inferno')
                
                cbr = plt.colorbar()
                cbr.set_label('Difference in hk [m/d]')
                plt_name = "diff_post_pest_v_I_made" 
                plt.savefig(os.path.join(figureDirectory, plt_name))
                
                
         # THE POINT IS, THIS ARRAY IS VERY DIFFERENT, IT IS NOT OPTIMISED!!!
                  
                #--------------------------------------------------------------------------
                # Make a list of all model arrays, all rankings and the original
                # (Original is index 0)
                
                all_model_arrays = []
                
                for i in range(len(hk_array_rankings)):
                    all_model_arrays.append(hk_array_rankings[i])
            
                all_model_arrays.append(hk_array_posterior)
                
                plt.close("all")
                
                #==========================================================================
                #=====RUNNING MODELS WITH EACH RANKING AND WITH POSTERIOR==================  
                #==========================================================================
                all_model_heads = []
                all_model_ages  = []
                # How is the simulation improved for each ranking value?
                f = 1
                for f in range(len(all_model_arrays)):
                    print(f)
                    r_modelname = modelname + "_r_" + str(f+1)
                    r_modelname_mt = modelname + "_r_" + str(f+1) + "_MT3D"
                
                    mf = flopy.modflow.Modflow(r_modelname, exe_name="mfnwtdbl.exe", version='mfnwt') # MODFLOW-NWT
                
                    nper = 1 # Number of model stress periods
                    perlen = [param_dict["length_simulation"]] 
                
                    nstp = [1] # Number of time steps in each stress period
                    tsmult = [1.0]
                    steady = [True] # True = steady state
                
                    dis = flopy.modflow.ModflowDis(mf, param_dict["nlay"], param_dict["nrow"], 
                                                   param_dict["ncol"], delr=param_dict["delr"], 
                                                delc=param_dict["delc"],top=param_dict["ztop"], 
                                                botm=param_dict["botm"][1:],
                                                tsmult = tsmult,
                                                nper=nper, perlen=perlen, nstp=nstp, 
                                                steady=steady)
                 
                    print("Discretisation set up")
                       
                #        # UPSTREAM WEIGHTING PACKAGE
              
                    hk_array = all_model_arrays[f]   
            
                    uwp = flopy.modflow.mfupw.ModflowUpw(mf, hk=hk_array, vka=param_dict["vka"], 
                                                         sy=param_dict["sy"], ss=param_dict["ss"], 
                                                     laytyp=param_dict["laytyp"]) # MODFLOW- NWT
                
                    print("Upstream Weighting set up")
                
                ## NEWTON SOLVER
                    nwt = flopy.modflow.mfnwt.ModflowNwt(mf, headtol=1e-03, fluxtol=0.05, maxiterout=100, 
                            thickfact=1e-07, linmeth=2, iprnwt=1, ibotav=1, options='SPECIFIED', 
                            Continue=True, dbdtheta=0.9, dbdkappa=1e-05, dbdgamma=0.0, momfact=0.1, 
                            backflag=1, maxbackiter=30, backtol=1.05, backreduce=0.9, maxitinner=50, 
                            ilumethod=2, levfill=5, stoptol=1e-10, msdr=15, iacl=2, norder=1, level=3, 
                            north=7, iredsys=1, rrctols=0.0, idroptol=1, epsrn=0.0001, 
                            hclosexmd=0.0001, mxiterxmd=200) # MODFLOW- NWT
            
                    print("Newton Solver set up")
                
                # --- Recharge --- #
                        
                # RECHARGE (RCH) PACKAGE
                
                    rch = flopy.modflow.ModflowRch(mf, rech=recharge_array)
                
                # GENERAL HEAD BOUNDARY (GHB) PACKAGE 
                    ghb = flopy.modflow.ModflowGhb(mf, stress_period_data=ghb_spd)
                    print("General Head Boundary set up")   
                             
                # OC PACKAGE
                                       
                    spd = {}
                    for strsp in range(nper):
                        tmstp = nstp[strsp]
                        for time_ in range(tmstp):
                            spd[strsp, time_] = ['save head', 'print budget', 
                                                   'save budget'] 
                                                   
                    oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd)
                    print("OC Package set up")
                    
                #bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=300)   
                
                    bas = flopy.modflow.ModflowBas(mf,ibound=1., strt=300., 
                                                   ifrefm=True, ixsec=False, 
                                                   hnoflo=-999.99)     
                    print("Basic set up") 
                
                # --- LINKING FILE --- #
                
                    # Set up the linking file for output from flow model to be used in transport model
                    lmt = flopy.modflow.ModflowLmt(mf, output_file_name= (r_modelname + str('_mt3dLink.ftl')),
                                                   output_file_header='extended',
                                                   output_file_format='unformatted')   
                                                
                    print("Link-Mass Transport package set up") 
                  
                    mf.write_input() # Write the model input files
                    print("input written")
                    
                # What's the best way to run the command?
                    subprocess.run([mf_exe, r_modelname]) # Run command 
                   
                #=====RUNNING THE TRANSPORT MODEL==============================================  
                   
                # Instantiate MT3D-USGS object in flopy
                
                    mt = mt3d.Mt3dms(modflowmodel=mf, modelname=r_modelname_mt, model_ws=this_model_folder, 
                                     version=mt3d_version, namefile_ext='mtnam', exe_name=mt3d_exe,  
                                     ftlfilename=(r_modelname + str('_mt3dLink.ftl'))) 
                       
                #------------------------------------------------------------------------------
                # BASIC TRANSPORT PACKAGE
                
                    ncomp = 1 # Total number of chemical species in simulation  
                    mcomp = 1 # Total number of "mobile" species 
                    sconc= np.zeros((param_dict["nlay"], param_dict["nrow"], param_dict["ncol"]), 
                                    dtype=np.float32)  # initial conc. = 0 yrs   
                    nprs = 0 # Flag indicating frequency of output. = 0: results not saved except 
                    # at end of simulation; > 0: saved at times specified by timprs; < 0: saved
                    # whenever the  number of transport steps is an even multiple of nprs.
                    
                # Instantiate basic transport (BTN) package for MT3D-USGS
                    btn = mt3d.Mt3dBtn(mt, nlay=param_dict["nlay"], nrow=param_dict["nrow"], 
                                       ncol=param_dict["ncol"], nper=nper, perlen=perlen, 
                                       ncomp=ncomp, mcomp=mcomp, sconc=sconc, prsity=param_dict["prsity"],
                                       delr=param_dict["delr"], delc=param_dict["delc"], icbund=1, ifmtcn=-1, savucn=True,
                                       nprs=nprs, nprobs=1, cinact=0, ssflag='SState', laycon=1) 
            
                #------------------------------------------------------------------------------
                # ADVECTION PACKAGE
                
                # Instantiate advection (ADV) package for MT3D-USGS
                    adv = mt3d.Mt3dAdv(mt, mixelm=0)  
                
                 #------------------------------------------------------------------------------
                # GENERALISED CONJUGATE GRADIENT SOLVER for MT3D-USGS
                
                # Instatiate generalized conjugate gradient solver (GCG) package for MT3D-USGS
                    gcg = mt3d.Mt3dGcg(mt, mxiter=30, iter1=50, isolve=2, accl=1, cclose=1e-06)
                
                #------------------------------------------------------------------------------
                # REACTION PACKAGE
                
                    rc1 = np.zeros((param_dict["nlay"], param_dict["nrow"], param_dict["ncol"]), dtype=np.float32)
                    rc1[:, :, :] = -1/365.25
                    
                    isothm = 0      # 0 = no sorption
                    ireact = 100    # 100 = zeroth-order reaction option
                    rc1 = rc1       # First order reaction rate for diffolved phase of first species.
                    
                    # Setting up the Reaction package
                    rct= mt3d.Mt3dRct(mt, isothm=isothm, ireact=ireact, igetsc=0, rc1=rc1) 
                
                #------------------------------------------------------------------------------
                # SOURCE-SINK MIXING PACKAGE
                
                    crch = np.zeros((param_dict["nrow"], param_dict["ncol"]), dtype=np.float32) # The age of recharging water is 0
                
                    itype = mt3d.Mt3dSsm.itype_dict()
              
                #Instantiate source-sink mixing (SSM) package for MT3D-USGS
                    ssm = mt3d.Mt3dSsm(mt, crch=crch) #, stress_period_data=ssm_data) 
                   
                #------------------------------------------------------------------------------
                # DISPERSION PACKAGE 
                    al = 1.5 # The longitudinal dispersivity, default = 0.01
                    dmcoef = 1E-4 # Effective molecular diffusion coefficient (for water in my model), default = 1E-9
                                # 9E-6 is a value that Lenny used --> but I think this is in m^2/s not m^2/d!
                
                # Instantiate up the Dispersion package for MT3D-USGS
                    dsp = mt3d.Mt3dDsp(mt, al=al, dmcoef=dmcoef) 
                
                #----- WRITING transport MODEL ------------------------------------------------
                 
                    mt.write_input()
                    mt.write_name_file()
            
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
                # "Manual" changes to the input files
                    conc_filename_dissolved = str(r_modelname_mt) + ".ucn"
                    conc_filename_sorbed = str(r_modelname_mt) + "_S.ucn"
                    mass_filename = str(r_modelname_mt) + ".mas"
                    cnf_filename = str(r_modelname_mt) + ".cnf"
                    
                ##Add a line to the MT3D .mtnam file re-naming output UCN file + MAS,CNF
                    mt_name_file = r_modelname_mt + str('.mtnam')
                
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
                    mt_btn_file = r_modelname_mt + str('.btn')
                
                
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
                    
                        a = param_dict["length_simulation"] # '%.1E' % Decimal(length_simulation)
                        b = str(a) + "         1         1 S"
                        c = str(a) + "         1         1         SState"
                     
                        ##
                    
                        for line in lines:
                            if b in line:
                                            line = line.replace(b, c)
                            f.write(line)
            
                    mt3d_namefile = r_modelname_mt + ".mtnam"
                 
                # Running the transport model
                    subprocess.run([mt3d_exe, mt3d_namefile]) # Run command 
                
               
                    # CREATE HEADFILE OBJECTS  
                    
                    headobj = bf.HeadFile(r_modelname + '.hds')
                    data = headobj.get_alldata()
                    times = headobj.get_times()
                    head_raster = headobj.get_data(totim=times[0]) # One time period: steady-state
                    
                    all_model_heads.append(head_raster)
                    
                    # CONCENTRATION/AGE FILES - FROM TRANSPORT MODEL
                    
                    directage_file = r_modelname_mt + str('.ucn')         
                    ucnobjc_direct_age = flopy.utils.binaryfile.UcnFile(directage_file, precision="double")        
                    age_times = ucnobjc_direct_age.get_times()        
                    age_raster = ucnobjc_direct_age.get_data(totim=age_times[0])
                    
                    all_model_ages.append(age_raster)
                       
            #------------------------------------------------------------------------------
                    # END Model running
            #------------------------------------------------------------------------------
            
                    # Get head and age data for each ranking at observation wells.
                rc_df.index = rc_df.a_wellname
                
                l = 0 # layer
                      
                modelrun_name = []
                rmse_h_list = []
                rmse_a_list = []
            
                weight_age = 1.
                weight_head = 25.
            
                f = 1
                for f in range(len(all_model_arrays)):            
                    if f == 0:
                        colname = "Modelled_r0_posterior"
                        residual_colname = "Residual_r0_posterior"
    
                    else:
                        colname = "Modelled_r_%d" %(f)
                        residual_colname = "Residual_r_%d" %(f)                
                    print(colname)
            
                    pest_obs_file[colname] = np.zeros(len(pest_obs_file.index))
                    pest_obs_file[residual_colname] = np.zeros(len(pest_obs_file.index))
                    
                    modelrun_name.append("Model_r_%d" %(f+1))
                    
                    residual_head_sum = 0
                    residual_age_sum = 0
                    well = 0
                    for well in range(len(rc_df.index)):
                        wellname   = rc_df.index[well]
            
            #                print(wellname)
             
                        row = rc_df.loc[wellname, "b_wellrow"]
                        col = rc_df.loc[wellname, "c_wellcol"]
            
                        name = obs_name_format.format(wellname, row, col)
                        print(name)
                        
                        pest_obs_file.loc[(str(name)+"_h"), colname] =  all_model_heads[f][l, row, col]
                        # pest_obs_file.loc[(str(name)+"_a"), colname] =  all_model_ages[f][l, row, col]
                               
                        residual_head = pest_obs_file.loc[(str(name)+"_h"), "Measured"] - all_model_heads[f][l, row, col]
                        # residual_age = pest_obs_file.loc[(str(name)+"_a"), "Measured"] - all_model_ages[f][l, row, col]
                        
                        pest_obs_file.loc[(str(name)+"_h"), residual_colname] =  residual_head
                        # pest_obs_file.loc[(str(name)+"_a"), residual_colname] =  residual_age
                                          
                        residual_head_sum = residual_head_sum + residual_head*residual_head
                        # residual_age_sum = residual_age_sum + residual_age*residual_age
            
                    rmse_head = math.sqrt(residual_head_sum/len(rc_df.index))
                    # rmse_age = math.sqrt(residual_age_sum/len(rc_df.index))
                    
                    rmse_h_list.append(rmse_head)
                    # rmse_a_list.append(rmse_age)
                    
                # Need x axis for my plots, the first is the posterior (all 100 faults)
                n_faults_ranking_rmse = []
            
                for i in range(len(all_model_arrays)-1):
                    n_faults_ranking_rmse.append(i)
            
                n_faults_ranking_rmse.append(len(phantom_df))
                    
                # rmse_comb1ined_list = weight_head*np.asarray(rmse_h_list) + weight_age*np.asarray(rmse_a_list)
                          
                plt.figure(figsize=(10,15))
                plt.suptitle("RMSE for %s flts (order %s), i cut:%1.1f" % (cum_or_not, sort_values_by, i_level),
                             fontsize=20)
                
                ax1 = plt.subplot(3, 1, 1)
                plt.plot(n_faults_ranking_rmse, rmse_h_list, "go", label="Head (unweighted?) RMSE")
                plt.grid(which="both")
                plt.legend(loc="upper center", fontsize=15, bbox_to_anchor=(0.5, 1.18))
                plt.xticks(fontsize=14, rotation=45)
                plt.yticks(fontsize=14, rotation=0)
                plt.axhline(y=rmse_h_list[0], color="lightgreen", ls="--")
                plt.axhline(y=rmse_h_list[-1], color="lightgreen", ls="--")
                axes = plt.gca()
                axes.set_xscale("log")
                axes.get_xaxis().set_ticks([])
    #            axes.set_ylim([0, 10])
                
                # plt.subplot(3, 1, 2)
                # plt.plot(n_faults_ranking_rmse, rmse_a_list, "mo", label="Age RMSE")
                # plt.ylabel("RMSE", fontsize = 20)
                # plt.grid(which="both")
                # plt.legend(loc="upper center", fontsize=15, bbox_to_anchor=(0.5, 1.18))
                # plt.xticks(fontsize=14, rotation=45)
                # plt.yticks(fontsize=14, rotation=0)
                # plt.axhline(y=rmse_a_list[0], color="pink", ls="--")
                # plt.axhline(y=rmse_a_list[-1], color="pink", ls="--")
            
                # axes = plt.gca()
                # axes.set_xscale("log")
                # axes.get_xaxis().set_ticks([])
              
                # plt.subplot(3, 1, 3)
                # plt.plot(n_faults_ranking_rmse, rmse_combined_list, "bo", label="Combined RMSE")
                # plt.xlabel("Number of phantom faults", fontsize=20)
                # plt.grid(which="both")
                # plt.legend(loc="upper center", fontsize=15, bbox_to_anchor=(0.5, 1.18))
                # plt.xticks(fontsize=14, rotation=45)
                # plt.yticks(fontsize=14, rotation=0)
                # plt.axhline(y=rmse_combined_list[0], color="dodgerblue", ls="--")
                # plt.axhline(y=rmse_combined_list[-1], color="dodgerblue", ls="--")
                # axes = plt.gca()
                # axes.set_xscale("log")
    #            axes.set_ylim([0, 300])
            
                name_fig = "rmse_%s_%s_%d" %(cum_or_not, sort_values_by[:3], i_level*100)
                plt.savefig(os.path.join(figureDirectory, name_fig))
                 
                plt.close('all')
                
                # Save the rmse lists:
                # name = "rmse_c_%s_%s" %(cum_or_not, sort_values_by[:3]) 
                # with open(os.path.join(dataDirectory, name), 'wb') as fp:
                #     pickle.dump(rmse_combined_list, fp)
                
                # name = "rmse_a_%s_%s" %(cum_or_not, sort_values_by[:3]) 
                # with open(os.path.join(dataDirectory, name), 'wb') as fp:
                #     pickle.dump(rmse_a_list, fp)
                
                name = "rmse_h_%s_%s" %(cum_or_not, sort_values_by[:3]) 
                with open(os.path.join(dataDirectory, name), 'wb') as fp:
                    pickle.dump(rmse_h_list, fp)
                
                # END LOOP ########################################################
     
        # =============================================================================
        # Plot original head file      
        
        plt.figure(figsize=(12, 24))
        # plt.title()
        plt.subplot(3, 1, 1, aspect = 'equal')
        
        plt.imshow(np.flipud(original_head_file), extent=extent, cmap='coolwarm_r')
        cbr = plt.colorbar()
        cbr.set_label('Original head (m)')
        # plt.savefig(os.path.join(figureDirectory, "orig_head_map"))
        
        # Plot model head file    
        # plt.figure()
        # plt.title()
        plt.subplot(3, 1, 2, aspect = 'equal')
        
        plt.imshow(np.flipud(all_model_heads[-1][0, :, :]), extent=extent, cmap='coolwarm_r')
        cbr = plt.colorbar()
        cbr.set_label('Model head (m)')
        # plt.savefig(os.path.join(figureDirectory, "orig_head_map"))
        
        # Compare original files to original model files
        
        difference_head = original_head_file - all_model_heads[-1][0, :, :]
        
        # plt.figure()
        # plt.title("Note that I've cropped the colourbar - some values are larger")
        plt.subplot(3, 1, 3, aspect = 'equal')
        
        plt.imshow(np.flipud(difference_head), extent=extent, cmap='PiYG')
        cbr = plt.colorbar()
        cbr.set_label('Difference in head (m)')
        plt.savefig(os.path.join(figureDirectory, "diff_head_map"))
        
        #####--------------------------------------------------------------------------
        # Now do the same for ages
        '''
        difference_age = original_age_file - all_model_ages[-1][0, :, :]
        
        plt.figure(figsize=(12, 24))
       
        plt.subplot(3, 1, 1, aspect = 'equal')
        
        plt.imshow(np.flipud(original_age_file), extent=extent, cmap='jet', vmin=0, vmax=1500)
        cbr = plt.colorbar()
        cbr.set_label('Original age (y)')
        
        plt.subplot(3, 1, 2, aspect = 'equal')
        plt.imshow(np.flipud(all_model_ages[-1][0, :, :]), extent=extent, cmap='jet', vmin=0, vmax=1500)
        cbr = plt.colorbar()
        cbr.set_label('Model age (y)')
        
        plt.subplot(3, 1, 3, aspect = 'equal')
        plt.imshow(np.flipud(difference_age), extent=extent, cmap='PuOr', vmin=-200, vmax=200)
        cbr = plt.colorbar()
        cbr.set_label('Difference in age (y)')
        
        plt.savefig(os.path.join(figureDirectory, "diff_age_map"))
                
        '''
        # -----------------------------------------------------------------------------
        # Now compare each ranked model to the original model and get a rmse value.
        # Maybe re-do the section above where I evaluate rmse to streamline/clarify...
        original_head = np.copy(original_head_file)
        # original_age  = np.copy(original_age_file)
        
        r_values_head = []
        # r_values_age = []
        
        rank_run = -1 # This is the final run, which is the posterior
        for rank_run in range(len(all_model_heads)):
        # Compare original files to pest model files
            pest_head     = np.copy(all_model_heads[rank_run][0, :, :]) #pest_head_array
            # pest_age      = np.copy(all_model_ages[rank_run][0, :, :]) #pest_age_array
            
            filtered_arrays    = [original_head, pest_head]
        
        # Filter arrays 
            ar_index = 1
            for ar_index in range(len(filtered_arrays)):
                
                # remove values close to boundaries
                for ir in range(param_dict["nrow"]):
                    for ic in range(param_dict["ncol"]):
                        if ir < 5 or ir >= 95:
                            filtered_arrays[ar_index][ir, ic] = np.nan
                        if ic < 5 or ic >= 195:
                            filtered_arrays[ar_index][ir, ic] = np.nan
                
                # remove values close to observation wells.
                for i in range(len(obs_well_x)):
                    for row_index in range(-1, 2):
                            for col_index in range(-1, 2):
                                row_number = (obs_well_y[i]/param_dict["delc"])+row_index
                                col_number = (obs_well_x[i]/param_dict["delc"])+col_index
                                if row_number > 99 or col_number > 199:
                                    pass
                                else: 
                                    filtered_arrays[ar_index][int(row_number), int(col_number)] = np.nan
                       
               # remove values located at faults. #NOTE THIS IS ALL FAULT LOCATION 
               # --> NOT BASED ON RANKINGS (where I have fewer faults in the model)
                for i in range(len(ph_fa_rows_list)):
                    for ib in range(len(ph_fa_rows_list[i])):
                        filtered_arrays[ar_index][int(ph_fa_rows_list[i][ib]), int(ph_fa_cols_list[i][ib])] = np.nan
             
            # HEAD
            plt.figure()
            plt.title("Original versus pest head everywhere except boundaries, obs wells and faults")
            plt.plot(filtered_arrays[0], filtered_arrays[1], "b*")
            axes = plt.gca()
            plt.xlabel('Original head (m)', fontsize=14)
            plt.ylabel('PEST head (m)', fontsize=14)   
            
            # Regression analysis
            # Flatten filtered arrays
            orig_flat_h = filtered_arrays[0].flatten()
            pest_flat_h = filtered_arrays[1].flatten()
            
            # Remove nans
            orig_flat_h = orig_flat_h[np.isfinite(orig_flat_h)]
            pest_flat_h = pest_flat_h[np.isfinite(pest_flat_h)]
            
            # Check
            if len(orig_flat_h) == len(pest_flat_h):
                print("Original and pest are the same length - no issues")
            else:
                print("Original and pest arrays are different lengths - follow up!")
                
            slope_h, intercept_h, r_value_h, p_value_h, std_err_h = stats.linregress(orig_flat_h, pest_flat_h)
            print("R-squared: %f" % r_value_h**2)
            
            max_of_max_head = max(np.max(orig_flat_h), np.max(pest_flat_h))
            min_of_min_head = min(np.min(orig_flat_h), np.min(pest_flat_h))
            
            axes.set_xlim([math.floor(min_of_min_head), math.ceil(max_of_max_head)])                
            axes.set_ylim([math.floor(min_of_min_head), math.ceil(max_of_max_head)])
            
            plt.text(220, 280, "R-squared: %f" % r_value_h**2, fontsize=14)
            if rank_run == 0:
                name = "head_regression_posterior"
            else:
                name = "head_regression_%d" %(rank_run+1)
            plt.savefig(os.path.join(figureDirectory, name))
               
          
            # Save the r values for further analysis
            r_values_head.append(r_value_h)
            # r_values_age.append(r_value_a)
                
                       
                
        o_h = filtered_arrays[0] 
        p_h = filtered_arrays[1]
        # o_a = filtered_arrays[2]
        # p_a = filtered_arrays[3]
    
        plt.figure(figsize=(16, 10))
    
        plt.subplot(2, 2, 1, aspect = 'equal')
        plt.imshow(np.flipud(o_h), extent=extent, cmap='coolwarm_r')
        cbr = plt.colorbar()
        cbr.set_label('Prior head filtered (m)')       
        
        plt.subplot(2, 2, 2, aspect = 'equal')
        plt.imshow(np.flipud(p_h), extent=extent, cmap='coolwarm_r')
        cbr = plt.colorbar()
        cbr.set_label('Posterior head filtered (m)')       
        
 
        #---------------
    
        # Differences        
        diff_filtered_h = o_h - p_h 
        # diff_filtered_a = o_a - p_a
        
        plt.figure(figsize=(9, 12))
        plt.subplot(2, 1, 1)
        plt.imshow(np.flipud(diff_filtered_h), extent=extent, cmap="PiYG", vmin=-15, vmax=15)
        cbr = plt.colorbar()
        cbr.set_label("Diff filtered head (m)")
                
     
                
        #---------------
    
        max_list = []
        min_list = []
        mean_list = []
        median_list = []
        std_list = []
        
        list_filtered = [o_h, p_h, diff_filtered_h]
        
        for m in range(len(list_filtered)):
            max_list.append(abs(np.nanmax(list_filtered[m])))
            min_list.append(abs(np.nanmin(list_filtered[m])))
            mean_list.append(abs(np.nanmean(list_filtered[m])))
            median_list.append(abs(np.nanmedian(list_filtered[m])))
            std_list.append(abs(np.nanstd(list_filtered[m])))
    
        stats_filtered_arrays = pd.DataFrame({
                                          
                    "max": max_list,
                    "min": min_list,
                    "mean": mean_list,
                    "median" : median_list,
                    "st_dev": std_list}, index = ["prior_head",
                                                 "posterior_head",
                                                 "diff_head"]
                                                )
          
        #--------------------------------------------------------------------------                                           
        # RMSE for whole grid for filtered arrays
        se_head_f = 0  # Filtered square error head
        # se_age_f = 0   # Filtered square error age 
        n_cells_f = 0
        
        for ir in range(param_dict["nrow"]):
            for ic in range(param_dict["ncol"]):
                if np.isnan(p_h[ir][ic]) == True: # Don't include nan values in rmse
                    pass
                else:
                    n_cells_f = n_cells_f + 1
                    
                    hse = (p_h[ir][ic] - o_h[ir][ic])**2     # Head square error
                    se_head_f = se_head_f + hse             # Sum square error
                     
                    # ase = (p_a[ir][ic] - o_a[ir][ic])**2      # Age square error
                    # se_age_f = se_age_f + ase                 # Sum square error    
                    
        rmse_head_f = math.sqrt(se_head_f/n_cells_f)                                   
        # rmse_age_f  = math.sqrt(se_age_f/n_cells_f)    
        
        #--------------------------------------------------------------------------
        # RMSE for the whole grid... for head and age (unfiltered array)
        original_head = np.copy(original_head_file)
        # original_age  = np.copy(original_age_file)
           
        for rank_run in range(len(all_model_heads)):
        # Compare original files to pest model files
            pest_head     = np.copy(all_model_heads[rank_run][0, :, :]) #pest_head_array
            # pest_age      = np.copy(all_model_ages[rank_run][0, :, :]) #pest_age_array
        
        # Check they are all the same
        # np.shape(original_head), np.shape(original_age), np.shape(pest_head), np.shape(pest_age)
    
        se_head_uf = 0  # Unfiltered square error head
        se_age_uf = 0   # Unfiltered square error age 
        n_cells = 0
        
        for ir in range(param_dict["nrow"]):
            for ic in range(param_dict["ncol"]):
                n_cells = n_cells + 1
                
                hse = (pest_head[ir][ic] - original_head[ir][ic])**2 # Head square error
                se_head_uf = se_head_uf + hse                        # Sum square error
                 
                # ase = (pest_age[ir][ic] - original_age[ir][ic])**2  # Age square error
                # se_age_uf = se_age_uf + ase                        # Sum square error                   
            
        rmse_head_uf = math.sqrt(se_head_uf/n_cells)                                   
        # rmse_age_uf  = math.sqrt(se_age_uf/n_cells)   
        
        list_of_rmse_whole_model = [rmse_head_f, rmse_head_uf]
        
        name = "rmse_f_uf_whole_mdl"
        with open(os.path.join(dataDirectory, name), 'wb') as fp:
            pickle.dump(list_of_rmse_whole_model, fp)    
                                                  
        # -------------------------------------------------------------------------
        # RMSE for the whole grid... for hk (unfiltered array)
        
        # Posterior hk
        hk_array_posterior
            
        # Set up prior hk  
        hk_array_prior = param_dict["hk_aquifer"] * np.ones((param_dict["nlay"], 
                                                               param_dict["nrow"], 
                                                               param_dict["ncol"]), 
                                                               dtype=np.float32)     
        for barr_number in range(n_real_faults):
            for i in range(re_fa_length_list[barr_number]):
                hk_array_prior[0, int(re_fa_rows_list[barr_number][i]), 
                int(re_fa_cols_list[barr_number][i])] = hk_real_fault
    
        diff_hk_prior_posterior = hk_array_posterior - hk_array_prior
        
        # Plot difference in hk
        plt.figure()
        plt.subplot(1, 1, 1, aspect='equal')    
           
        plt.imshow(np.flipud(diff_hk_prior_posterior[0, :, :]), extent=extent, cmap="RdGy") 
        cbr = plt.colorbar()
        cbr.set_label('Diff in hk (m/d')
        
        plt.savefig(os.path.join(figureDirectory, "diff_hk_prr_postr"))
        
        # Calculate the hk rmse
        
        se_hk_uf = 0  # Unfiltered square error head
        n_cells = 0
        
        for ir in range(param_dict["nrow"]):
            for ic in range(param_dict["ncol"]):
                n_cells = n_cells + 1
                hkse = (hk_array_posterior[0][ir][ic] - hk_array_prior[0][ir][ic])**2 # Head square error
                se_hk_uf = se_hk_uf + hkse     
        
        rmse_hk_uf = math.sqrt(se_hk_uf/n_cells)          
    
        list_of_whole_model_rmses = [rmse_head_uf, rmse_hk_uf]    # head, age, hk                     
    
    
    #--------------------------------------------------------------------------
        # Save the statistics index
        fileName = os.path.join(dataDirectory, "filtered_arrays_stats.csv")
        stats_filtered_arrays.to_csv(fileName, encoding='utf-8', index=True)
        
        #Save the key dataframe that just has the key results
        fileName = os.path.join(dataDirectory, "key_dataframe.csv")
        key_dataframe.to_csv(fileName, encoding='utf-8', index=True)
    
        # Save the rmse lists for the whole model:
        name = "rmse_whole_model_hd_ag_hk"
        with open(os.path.join(dataDirectory, name), 'wb') as fp:
            pickle.dump(list_of_whole_model_rmses, fp)
            
        
    else:
        print("No phantom faults were identifiable")
        name = "rmse_c_cum_inv"
        rmse_combined_list = [0]
        with open(os.path.join(dataDirectory, name), 'wb') as fp:
            pickle.dump(rmse_combined_list, fp)
        
        name = "rmse_a_cum_inv" 
        rmse_a_list = [0]
        with open(os.path.join(dataDirectory, name), 'wb') as fp:
            pickle.dump(rmse_a_list, fp)
        
        name = "rmse_h_cum_inv"
        rmse_h_list = [0]
        with open(os.path.join(dataDirectory, name), 'wb') as fp:
            pickle.dump(rmse_h_list, fp)    
            
    #--------------------------------------------------------------------------
       
    # Save the phantom dataframe with all of these results
    fileName = os.path.join(dataDirectory, "phantom_df.csv")
    phantom_df.to_csv(fileName, encoding='utf-8', index=True)
    
    
    plt.close('all')
print("Run finished for " + overallName)
