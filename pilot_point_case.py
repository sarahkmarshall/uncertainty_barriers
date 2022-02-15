# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:13:58 2020

@author: mars0451
"""
#=====IMPORTS==================================================================

import flopy
import numpy as np
import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf
import pandas as pd
import subprocess
import os, shutil
import time
from flopy import mt3d
import pyemu
import sys
import string
import pickle
import random 

functions_path = "C:\workspace\sarahs_functions"

if any(functions_path in s for s in sys.path):
    print("Functions path already in list")
else:
    sys.path.append(r"C:\workspace\sarahs_functions")

import functions_proj_3

#=====SETTING UP MULTIPLE RUNS=================================================

phantom_or_pilot = "Pilot_points" # "Phantom_faults" or "Pilot_points"

t1 = time.time()

already_run_pest = "NO"

all_master_files_saved = True
#### !! Don't change these for UB-18 
already_set_up_obs_wells =      True   #
add_additional_obs_wells =      False
subsample_original_obs_wells =  True
already_saved_params =          True  #
already_saved_ph_faults =       True  #
have_no_fault =                 False
include_real_with_ph =          False
putting_in_key_faults =         False
#### !! Don't change these for UB-18 

n_subsampled_wells = 25

n_real_faults = 1 

n_phantom_faults = 100

is_normal_score_transform_used = "NO" # "YES"

hk_array_cmap = 'BrBG'

obs_optn_list = [1]*6 + [2]*6 + [3]*6
ph_optn_list = [0]*2 + [1]*2 + [2]*2
ph_optn_list = ph_optn_list*3

n_runs = len(obs_optn_list) # Doesn't matter what I say for this if running individually
run_list = list(range(n_runs))

idxs_for_ppt = [0, 1, 6, 7, 12, 13] # Two for each ow config (still doing high phimlim and low phimlim)

if phantom_or_pilot == "Pilot_points":
    run_list = [run_list[i] for i in idxs_for_ppt]
    obs_optn_list = [obs_optn_list[i] for i in idxs_for_ppt]
    ph_optn_list = [ph_optn_list[i] for i in idxs_for_ppt]
else:
    pass
#=====DIRECTORIES==============================================================

overallName = "UB_27"

proj_3_folder       = r'C:\workspace\Proj3_PilbaraBarriers'

MODFLOW_folder      = r'C:\workspace\modflow_dbl\mfnwtdbl.exe' # double precision MODFLOW NWT
mfnwt_exe_name = "mfnwtdbl.exe" 

MT3D_USGS_folder      = r'C:\workspace\modflow_dbl\mt3dusgsdbl.exe' # Double precision MT3D-USGS
mt3d_version = 'mt3d-usgs' 
mt3d_exe_name = "mt3dusgsdbl.exe"

# SET UP OVER-ARCHING FOLDER FOR SAVING OBSERVATION WELLS

overarching_model_folder = os.path.join(proj_3_folder, overallName)
if not os.path.exists(overarching_model_folder):
    os.makedirs(overarching_model_folder) 


#==============================================================================
#COPY OVER THE FAULTS AND OBS WELLS FROM ub_18 = the "MASTER"


if all_master_files_saved == False:
    # OBSERVATION WELL FILES
    
    o_w_1_masters = [os.path.join(r'C:\workspace\Proj3_PilbaraBarriers\UB_18', 'obs_rowcol_1.csv'),
                     os.path.join(r'C:\workspace\Proj3_PilbaraBarriers\UB_18', 'obs_rowcol_2.csv'),
                     os.path.join(r'C:\workspace\Proj3_PilbaraBarriers\UB_18', 'obs_rowcol_3.csv')]
    
    o_w_1_new     = [os.path.join(r'C:\workspace\Proj3_PilbaraBarriers', str(overallName), 'obs_rowcol_1.csv'),
                     os.path.join(r'C:\workspace\Proj3_PilbaraBarriers', str(overallName), 'obs_rowcol_2.csv'),
                     os.path.join(r'C:\workspace\Proj3_PilbaraBarriers', str(overallName), 'obs_rowcol_3.csv')]
    
    
    for f in range(len(o_w_1_masters)):
        if not os.path.isfile(o_w_1_new[f]):
            print("Obs files not yet in folder so they are being copied from MASTER (UB_18)")
            shutil.copy2(o_w_1_masters[f], o_w_1_new[f])
        else:
            print("My Master obs well files are already in this folder")
            
        
        
     
    # PHTANTOM FAULT FILES
    
    p_f_1_masters = [os.path.join(r'C:\workspace\Proj3_PilbaraBarriers\UB_18', 'ph_f_coords_1.npy'),
                     os.path.join(r'C:\workspace\Proj3_PilbaraBarriers\UB_18', 'ph_f_coords_2.npy'),
                     os.path.join(r'C:\workspace\Proj3_PilbaraBarriers\UB_18', 'ph_f_coords_3.npy')]
    
    p_f_1_new     = [os.path.join(r'C:\workspace\Proj3_PilbaraBarriers', str(overallName), 'ph_f_coords_1.npy'),
                     os.path.join(r'C:\workspace\Proj3_PilbaraBarriers', str(overallName), 'ph_f_coords_2.npy'),
                     os.path.join(r'C:\workspace\Proj3_PilbaraBarriers', str(overallName), 'ph_f_coords_3.npy')]
    
    
    for f in range(len(p_f_1_masters)):
        if not os.path.isfile(p_f_1_new[f]):
            print("Phantom faults not yet in folder so they are being copied from MASTER (UB_18)")
            shutil.copy2(p_f_1_masters[f], p_f_1_new[f])
        else:
            print("My Master phantom faults are already in this folder")
        
       
        
        
    # PARAMETER DICTIONARY    
        
    param_dict_master = os.path.join(r'C:\workspace\Proj3_PilbaraBarriers\UB_18', 'param_dict.p')    
        
    param_dict_new = os.path.join(r'C:\workspace\Proj3_PilbaraBarriers', str(overallName), 'param_dict.p')
    
    if not os.path.isfile(param_dict_new):
        print("Parameter dictionary not yet in folder so it is being copied from MASTER (UB_18)")
        shutil.copy2(param_dict_master, param_dict_new)
    else:
        print("My Master parameter dictionary is already in this folder")    
    
else:
    print("Master files already set up")   

run = run_list[0]
obs_optn = obs_optn_list[0]
ph_optn = ph_optn_list[0]

for run, obs_optn, ph_optn in zip(run_list, obs_optn_list, ph_optn_list):
    print("...................")
    print("Run # %d" %run)
    print("Obs option is: %d" %obs_optn)
    print("Ph option is: %d" %ph_optn)

    # a - f = obs 1
    # g - l = obs 2
    # m - r = obs 3
    # RUNNING MODELS FOR PAPER #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
    which_obs_optn = obs_optn     # Out of 1, 2, 3
    phantom_optn =   ph_optn        # Out of 0, 1, 2
    
    
    # DEFINE FUNCTIONS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    # Defined externally in sys path I have appended.
    
    #=====MODEL DISCRETISATION=====================================================
    
    ## FIXED PARAMETERS --> Use an external script where all of this information is set...
    
    param_dict = {}
    
    param_dict["Lx"]    = 20000.  # Length of the model sides in metres.
    param_dict["Ly"]    = 10000.  # Length of the model sides in metres.
    param_dict["delr"]  = 100.  # Spacing length across rows in metres.
    param_dict["delc"]  = 100.  # Spacing length across columns in metres.
    
    param_dict["sHead"] = 300  # Starting head across aquifer.
    param_dict["nlay"]  = 1     # Number of layers.
    param_dict["ztop"]  = 300.  # Top of model.
    param_dict["zbot"]  = 0.    # Base of model.
    param_dict["delv"]  = (param_dict["ztop"] - param_dict["zbot"]) / param_dict["nlay"] # Length of cells in z direction.
    param_dict["botm"]  = np.linspace(param_dict["ztop"], param_dict["zbot"], param_dict["nlay"] + 1) 
    param_dict["ncol"]  = int(param_dict["Lx"]/param_dict["delr"]) # Number of columns
    param_dict["nrow"]  = int(param_dict["Ly"]/param_dict["delc"]) # Number of rows
    param_dict["sy"]    = 0.1 ## Specific yield
    param_dict["ss"]    = param_dict["sy"]/param_dict["ztop"] ## Specific storage
    param_dict["laytyp"] = 1 ## Layer type (1 = convertible; 0 = confined)
    param_dict["laybarr"] = 0
    param_dict["prsity"] = param_dict["sy"]
    param_dict["length_simulation"] = 1
    param_dict["recharge_flux"] = 1.3699e-5 # m/d HIGH This is equivalent ot 5 mm per year.
    
    param_dict["hk_aquifer"] = 1.  # Hydraulic conductvity along rows (m/day) It worked ar 0.09
    param_dict["vka"]   = param_dict["hk_aquifer"]/10. # Vertical hydraulic conductivity
    
    os.chdir(overarching_model_folder)
    
    if already_saved_params == False:
        with open("param_dict.p", "wb") as fp:
            pickle.dump(param_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("param_dict.p", "rb") as fp:
            param_dict = pickle.load(fp)
    
    #=====BOUNDARY CONDITIONS==================================================
    
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
    
    #=====STARTING HEADS AND CELL TYPE I.E IBOUND PACKAGE======================
    
    if param_dict["nlay"] == 1:
        ibound = np.ones((param_dict["nrow"], param_dict["ncol"]), dtype=np.int32)
        strt = param_dict["sHead"] * np.ones((param_dict["nrow"], param_dict["ncol"]), dtype=np.float32)
    else:
        ibound = np.ones((param_dict["nlay"], param_dict["nrow"], param_dict["ncol"]), dtype=np.int32)
        strt = param_dict["sHead"] * np.ones((param_dict["nlay"], 
                param_dict["row"], param_dict["ncol"]), dtype=np.float32)
    
    #=====PHANTOM FAULT LOCATIONS==============================================
    # Phantom fault set up names and K values
    if putting_in_key_faults == True:
        ph_key_fault_coordinates = np.load(os.path.join(overarching_model_folder, "ph_fault_coords_key.npy"))
        n_phantom_faults = len(ph_key_fault_coordinates)
    
    else:
        print("Using all phantom faults") 
    
    n_ph_fault_setups = 3  
    
      
    ph_f_coords = []
    
    # Set up various barriers at different sizes and locations    
    for i in range(n_ph_fault_setups):
        name = "ph_f_coords_%d" %(i+1)
        if already_saved_ph_faults == False:
            print("Making new phantom faults")
        
            f_coords = functions_proj_3.barriers.makeFaultCoords(n_phantom_faults, 
                                                                 param_dict["nrow"], 
                                                                 param_dict["ncol"],
                                                                 param_dict["delr"])
            
            ph_f_coords.append(f_coords)        
            
            np.save(os.path.join(overarching_model_folder, name), f_coords)    
    
        else:
            # Overwrite the previous fault coordinates if I am just using "Key" faults from previous inversion
            if putting_in_key_faults == True: 
                print("Phantom fault coordinates are just the key coordinates from previous inversion")
                ph_fault_coordinates = ph_key_fault_coordinates
            else:
                print("Already saved phantom faults so loading them")
                ph_f_coords.append(np.load(os.path.join(overarching_model_folder, name+".npy")))
    
    ph_f_rows       = []
    ph_f_cols       = []
    ph_f_lengths    = []
    
    for ph_optn in range(n_ph_fault_setups):
    
        ph_fa_rows_list, ph_fa_cols_list, ph_fa_length_list = (functions_proj_3.barriers.makeFaultRowCols(
                                        ph_f_coords[ph_optn], 
                                        param_dict["nrow"], 
                                        param_dict["ncol"]))
        
        # Remove any faults that turned out not on the grid
        for i in range(len(ph_fa_length_list)):
            if ph_fa_length_list[i] == 0:
                print("Length of barrier is 0 --> no barrier on grid")
                while ph_fa_length_list[i] == 0:
                    fault_coordinate_redone = functions_proj_3.barriers.makeFaultCoords(1, 
                                                            param_dict["nrow"], 
                                                            param_dict["ncol"],
                                                            param_dict["delr"])    
                    ph_f_coords[ph_optn][i] = fault_coordinate_redone[0]
                    ph_fa_rows_list, ph_fa_cols_list, ph_fa_length_list = (
                            functions_proj_3.barriers.makeFaultRowCols(ph_f_coords[ph_optn], 
                                                                       param_dict["nrow"], 
                                                                       param_dict["ncol"]))
                    ph_fa_length_list[0]
        
            else:
                print("Length of barrier is fine, there is a barrier on the grid")
                
        ph_f_rows.append(ph_fa_rows_list)
        ph_f_cols.append(ph_fa_cols_list)
        ph_f_lengths.append(ph_fa_length_list)
    
    # -------------------------------------------------------------------------
    # PLOT THE LOCATION OF ALL PHANTOM FAULTS
    
    hk_ph = param_dict["hk_aquifer"] * np.ones((param_dict["nlay"], 
                                                           param_dict["nrow"], 
                                                           param_dict["ncol"]), 
                                                           dtype=np.float32)    
            
    for ph_optn in range(n_ph_fault_setups):
        for barr_number in range(n_phantom_faults):
            for i in range(ph_f_lengths[ph_optn][barr_number]):
                hk_ph[0, int(ph_f_rows[ph_optn][barr_number][i]), 
                int(ph_f_cols[ph_optn][barr_number][i])] = 1e-3  
                
                               
    # Check: plot HK array
    hk_ph_log = np.log10(hk_ph)
    plt.figure()
    plt.subplot(1, 1, 1, aspect='equal')    
    
    extent = (param_dict["delr"]/2., 
              param_dict["Lx"] - param_dict["delr"]/2., 
              param_dict["delc"]/2., 
              param_dict["Ly"] - param_dict["delc"]/2.)
    
    plt.imshow(np.flipud(hk_ph_log[0, :, :]), extent=extent, cmap="Greys") 
    cbr = plt.colorbar()
    cbr.set_label('Log hk (m/d')
    
    plt.savefig(os.path.join(overarching_model_folder, "all_ph_faults"))    
    
    # Array with all values of python faults are equal to True
    
    hk_ph = param_dict["hk_aquifer"] * np.ones((param_dict["nlay"], 
                                                           param_dict["nrow"], 
                                                           param_dict["ncol"]), 
                                                           dtype=bool)    
            
    for ph_optn in range(n_ph_fault_setups):
        for barr_number in range(n_phantom_faults):
            for i in range(ph_f_lengths[ph_optn][barr_number]):
                hk_ph[0, int(ph_f_rows[ph_optn][barr_number][i]), 
                int(ph_f_cols[ph_optn][barr_number][i])] = 0        
                               
    # How many cells have a fault?
    total_number_cells = hk_ph.size
    total_number_cells_without_phantom = np.sum(hk_ph)
    total_number_cells_with_phantom =   total_number_cells - total_number_cells_without_phantom
    
    #==============================================================================
    #=====SETTING UP OBS WELLS LOCATIONS ==========================================  
    #==============================================================================     
           
    if already_set_up_obs_wells == False: 
        
        no_phantom_coords = []
        for ir in range(param_dict["nrow"]):
            for ic in range(param_dict["ncol"]):
                if hk_ph[0, ir, ic] == 1:
                    no_phantom_coords.append([ir, ic])
                    
        all_obs_wells = random.sample(no_phantom_coords, k=1500)
        n_obs_wells = 500  
        obs_wells_1 = all_obs_wells[:n_obs_wells]
        obs_wells_2 = all_obs_wells[n_obs_wells:(n_obs_wells*2)]
        obs_wells_3 = all_obs_wells[(n_obs_wells*2):]
        
        obs_wells = [obs_wells_1, obs_wells_2, obs_wells_3]
        
        wellnames_list = []
        for i in range(n_obs_wells):
            well_name = "well_" + str(1+i)
            print(well_name)
            wellnames_list.append(well_name)
        
        well_rows = []
        well_cols = []
        for optn in range(len(obs_wells)):
            row_list = []
            col_list = []
            for well in range(n_obs_wells):
                row_list.append(obs_wells[optn][well][0])
                col_list.append(obs_wells[optn][well][1])
            well_rows.append(row_list)
            well_cols.append(col_list)
            
        
        row_cols_obswells_py = []
        
        for optn in range(len(obs_wells)):
              
            row_cols_obswells_py.append(pd.DataFrame({  # I.e. "py" = python = 0-based
                                          
                    "a_wellname": wellnames_list,
                    "b_wellrow": well_rows[optn],
                    "c_wellcol": well_cols[optn],
                    "d_welllay": np.zeros(n_obs_wells),
                                                })  )
            
        for optn in range(len(obs_wells)):
            name =  "obs_rowcol_%d.csv"%(1+optn)
            fileName = os.path.join(overarching_model_folder, name)
            row_cols_obswells_py[optn].to_csv(fileName, encoding='utf-8', index=False)
        
    
    else:   # The obs well locations are already saved and I need to download them
        pass   
    
    # Now that I know I have the obs file in the overarching model folder, I can
    # download it to use in the script. 
    
    
    fileName = os.path.join(overarching_model_folder, "obs_rowcol_%d.csv"%which_obs_optn)
    
    #If I want to download a specific database.
    #fileName = os.path.join(r'C:\workspace\Proj3_PilbaraBarriers\UB_16_b\Data', "obs_rowcol_f.csv")
    
    rc_df = pd.read_csv(fileName) # This has python-based numbering
    original_n_wells = len(rc_df)
    
    rc_df.head()    
    rc_df.index = rc_df.a_wellname
    
    if subsample_original_obs_wells == True:
        rc_df = rc_df[:n_subsampled_wells] # Since they are all randomly chosen I can just select in order
    else:
        pass
    
    # Make x and y JUST for plots --> corresponding to middle location of cells
    obs_wells_x = (np.asarray(rc_df["c_wellcol"]))*param_dict["delr"]+0.5*param_dict["delr"]
    obs_wells_y = (np.asarray(rc_df["b_wellrow"]))*param_dict["delc"]+0.5*param_dict["delc"]
    
    number_obs_wells = len(rc_df)
    
    print("The number of observation wells is: %d" %number_obs_wells)
    
    #=====RUNNING PEST MULTIPLE TIMES==============================================
    


    if putting_in_key_faults == True:
        modelName = overallName + "_" + string.ascii_lowercase[run] + "_key"
    else:
        modelName = overallName + "_" + string.ascii_lowercase[run] + "_plt"
    print("Model name is: %s"%modelName)
    
    modelname_mt = modelName + "_MT3D" 

    this_model_folder = os.path.join(proj_3_folder, modelName)
    if not os.path.exists(this_model_folder):
        os.makedirs(this_model_folder)
    
    dataDirectory = os.path.join(this_model_folder, 'Data')
    if not os.path.exists(dataDirectory):
        os.makedirs(dataDirectory)
    
    figureDirectory = os.path.join(this_model_folder, 'Figures')
    if not os.path.exists(figureDirectory):
        os.makedirs(figureDirectory)
    
    # Set working directory
    os.chdir(this_model_folder)
     
    # Copy modflow to folder
    this_model_modflow = os.path.join(this_model_folder, mfnwt_exe_name)
    
    if not os.path.isfile(this_model_modflow):
        print("modflow nwt dbl not yet in folder")
        shutil.copy2(MODFLOW_folder, this_model_modflow)
    else:
        print("modflow nwt dbl is already in folder")
        
    # Copy mt3d to folder
    this_model_mt3d = os.path.join(this_model_folder, mt3d_exe_name)
    
    if not os.path.isfile(this_model_mt3d):
        print("mt3d usgs dbl not yet in folder")
        shutil.copy2(MT3D_USGS_folder, this_model_mt3d)
    else:
        print("mt3d usgs dbl is already in folder")
        
        
            
    modelname = modelName 
    print(modelname)
    modelname_mt = modelname_mt
    print(modelname_mt)
    
 
    #==========================================================================
    #=====SETTING UP MODEL=====================================================  
    #==========================================================================
    
    #=====FAULT HK VALUES======================================================
       
    # Real fault set up name(s) and K value(s)
    
    fault_name_format = "hkph_%03d"
    rl_fault_name_format = "hkre_%03d"
    
    real_fault_K = param_dict["hk_aquifer"]/1000 # param_dict["hk_aquifer"]/1000
    
    rl_fault_name_list = []
    
    for i in range(n_real_faults):
        rl_fault_name_list.append(rl_fault_name_format % (i+1))   
    
    rl_fault_K_list = []    
    
    for i in range(n_real_faults):
        # Randomly allocating K of fault:
        #phantom_fault_K_list.append(10**(-6 + (7*sp.rand())))
        # Not randomly allocating K, using "phantom fault" method:
        rl_fault_K_list.append(real_fault_K)  
    
    phantom_fault_K = param_dict["hk_aquifer"] # prior K same as the aquifer
    
    ph_fault_name_list = []
    
    for i in range(n_phantom_faults):
        ph_fault_name_list.append(fault_name_format % (i+1))
    
    phantom_fault_K_list = [] 
    
    for i in range(n_phantom_faults):
        phantom_fault_K_list.append(phantom_fault_K)
    
    K_barr_df = pd.DataFrame({
                                  
            "barr_name": ph_fault_name_list,
            "K_barr_orig": phantom_fault_K_list
                                        })
    
    fileName = os.path.join(this_model_folder, "real_K_barr_values.csv")
    K_barr_df.to_csv(fileName, encoding='utf-8', index=False)
         
    #=====REAL FAULT LOCATIONS=================================================
    re_r_start  = int(2500/param_dict["delc"])
    re_c_start  = int(9200/param_dict["delr"])
    re_r_end    = int(6400/param_dict["delc"])
    re_c_end    = int(13100/param_dict["delr"])
    re_fault_coordinates = [ # format [row start, col start, row end, col end]
                           [re_r_start, re_c_start, re_r_end, re_c_end] # RE 01
                           ] 
                                                      
    re_fa_rows_list, re_fa_cols_list, re_fa_length_list =  (
            functions_proj_3.barriers.makeFaultRowCols(re_fault_coordinates, 
                                                       param_dict["nrow"], 
                                                       param_dict["ncol"]))
    
    # But make sure the fault is actually on the grid.
    for i in range(len(re_fa_length_list)):
        if re_fa_length_list[i] == 0:
            print("Length of barrier is 0 --> no barrier on grid")
            while re_fa_length_list[i] == 0:
                fault_coordinate_redone = functions_proj_3.barriers.makeFaultCoords(1)    
                re_fault_coordinates[i] = fault_coordinate_redone[0]
                re_fa_rows_list, re_fa_cols_list, re_fa_length_list = (
                    functions_proj_3.barriers.makeFaultRowCols(re_fault_coordinates, 
                                                       param_dict["nrow"], 
                                                       param_dict["ncol"]))
                re_fa_length_list[0]
    
        else:
            print("Length of barrier is fine, there is a barrier on the grid")
            
    np.save(os.path.join(dataDirectory, "re_fault_coords"), re_fault_coordinates)
    
    # -------------------------------------------------------------------------
    # Plot the location of real faults and save 
 
    # First make the colour list for the legend
    real_colourlist, real_legendlist = functions_proj_3.barriers.makeColourList(rl_fault_K_list)
    
    plt.figure(figsize=(20, 10))
    
    for i in range(len(re_fa_rows_list)):
        plt.plot(re_fa_cols_list[i], re_fa_rows_list[i], lw = 3, 
                 color = real_colourlist[i])
        
    plt.xlabel("Columns", fontsize=14)
    plt.ylabel("Rows", fontsize=14)
    
    # I need to re-do this code to plot the legend
#    for j in range(n_categories):
#        plt.plot([-1, -2], [-5, -6], color = orig_colour_list[j], label = category_list[j]) 
#    
    axes = plt.gca()
    axes.set_xlim([0, param_dict["ncol"]])                
    axes.set_ylim([0, param_dict["nrow"]])         
#    
#    plt.legend(loc=4) # legend_list  
    plt.savefig(os.path.join(figureDirectory, "barrs_hk_real_orig"))
    
    # -------------------------------------------------------------------------
    # Plot phantom fault locations
    
    ph_fa_cols_list =    ph_f_cols[phantom_optn]
    ph_fa_rows_list =    ph_f_rows[phantom_optn]
    ph_fa_length_list = ph_f_lengths[phantom_optn]
    phantom_f_coords =  ph_f_coords[phantom_optn]


    # Save the phantom coords for this model run specifically     
    np.save(os.path.join(dataDirectory, "ph_fault_coords.npy"), phantom_f_coords)
    
    # First make the colour list for the legend
    ph_colourlist, ph_legendlist = functions_proj_3.barriers.makeColourList(phantom_fault_K_list)

    plt.figure(figsize=(20, 10))
    for i in range(len(ph_fa_rows_list)):
        plt.plot(ph_fa_cols_list[i], ph_fa_rows_list[i], lw = 3, 
                 color = ph_colourlist[i])
    plt.xlabel("Columns", fontsize=14)
    plt.ylabel("Rows", fontsize=14)
      
    axes = plt.gca()
    axes.set_xlim([0, param_dict["ncol"]])                
    axes.set_ylim([0, param_dict["nrow"]])                
    
    plt.legend(loc=4) # legend_list  
    plt.savefig(os.path.join(figureDirectory, "barrs_hk_phantom_orig"))
    
    # Now add the real fault
    for i in range(len(re_fa_rows_list)):
        plt.plot(re_fa_cols_list[i], re_fa_rows_list[i], lw = 3, 
                 color = real_colourlist[i])
        
    plt.savefig(os.path.join(figureDirectory, "barrs_hk_ph_r_orig"))
          
    # -------------------------------------------------------------------------
    # Save a back-up of the obs file in this model folder for back up
    
    fileName = os.path.join(dataDirectory, "obs_rowcol.csv") 
    rc_df.to_csv(fileName, encoding='utf-8', index=False)

    # Determine which observation wells are lying on a phantom fault cell
    ph_fa_rows_list # Ph rows by cell - 0-based (python-based)
    ph_fa_cols_list # Ph cols by cell - 0-based (python-based)
    
    wells_intersect_ph = []
    for well in rc_df.index:
        print(well)
        wellrowcol = [rc_df.loc[well, "b_wellrow"], rc_df.loc[well, "c_wellcol"]]
        print(wellrowcol)
        for n_fault in range(len(ph_fa_rows_list)):
            for j in range(len(ph_fa_rows_list[n_fault])):
                wellrowcol_ph = [int(ph_fa_rows_list[n_fault][j]), int(ph_fa_cols_list[n_fault][j])]
                if wellrowcol_ph == wellrowcol:
                    print(wellrowcol_ph)
                    print("Intersection in n_fault= % d, j = %d" %(n_fault, j))
                    wells_intersect_ph.append(well)
                else:
                    pass

    print("Number of insersections (including doubles): %d" %len(wells_intersect_ph))
    
    #=====MAKE HYDRAULIC CONDUCTIVITY VECTOR===================================
    
    # HYDRAULIC CONDUCTIVITY VECTOR
    
    hk_array_barrier = param_dict["hk_aquifer"] * np.ones((param_dict["nlay"], 
                                                           param_dict["nrow"], 
                                                           param_dict["ncol"]), 
                                                           dtype=np.float32)     
    if have_no_fault == True:
        pass
    else:        
        for barr_number in range(n_real_faults):
            for i in range(re_fa_length_list[barr_number]):
                hk_array_barrier[0, int(re_fa_rows_list[barr_number][i]), 
                int(re_fa_cols_list[barr_number][i])] = rl_fault_K_list[barr_number]  
    
                               
    # Check: plot HK array
    hk_array_log = np.log10(hk_array_barrier)
    plt.figure()
    plt.subplot(1, 1, 1, aspect='equal')    
    
    extent = (param_dict["delr"]/2., 
              param_dict["Lx"] - param_dict["delr"]/2., 
              param_dict["delc"]/2., 
              param_dict["Ly"] - param_dict["delc"]/2.)
    
    plt.imshow(np.flipud(hk_array_log[0, :, :]), extent=extent, cmap=hk_array_cmap) 
    cbr = plt.colorbar()
    cbr.set_label('Log hk (m/d')
    
    plt.savefig(os.path.join(figureDirectory, "real_barrs_hk_array"))
    
    hk_array_no_barr = param_dict["hk_aquifer"] * np.ones((param_dict["nlay"], 
                                                           param_dict["nrow"], 
                                                           param_dict["ncol"]), 
                                                           dtype=np.float32)     
    
    # Save the original hk file - so I can compare it to output with pest
    fname = "original_hk_file"
    np.savetxt(os.path.join(dataDirectory, fname), hk_array_barrier[0, :, :], fmt='%8.6e', delimiter='   ')
    
    
    # HK Array with barriers ---> print it here and save the fig, 
    # compare to my plot of hk array with lines as I drew above.
    
    hk_array_ph_prior = param_dict["hk_aquifer"] * np.ones((param_dict["nlay"], 
                                                            param_dict["nrow"], 
                                                            param_dict["ncol"]), dtype=np.float32)     
    

    for round in range(2):    
        if round == 0:
            hk_ph_faults = phantom_fault_K_list # The actual prior
            name_img = "ph_barrs_prior"
        else:
            hk_ph_faults = n_phantom_faults*[1e-3] # Do this just to show where they are
            name_img = "where_are_ph_faults_prior"
            
        for barr_number in range(n_phantom_faults):
            barr_name = ph_fault_name_list[barr_number]
            print(barr_name)
            for i in range(ph_fa_length_list[barr_number]):
                hk_array_ph_prior[0, int(ph_fa_rows_list[barr_number][i]), 
                         int(ph_fa_cols_list[barr_number][i])] = hk_ph_faults[barr_number]                                   
            
       
        # Check: plot HK array - check prior
        hk_array_log_barrier = np.log10(hk_array_ph_prior)
        plt.figure()
        plt.subplot(1, 1, 1, aspect='equal')    
        
        extent = (param_dict["delr"]/2., 
                  param_dict["Lx"] - param_dict["delr"]/2., 
                  param_dict["delc"]/2., 
                  param_dict["Ly"] - param_dict["delc"]/2.)
        
        plt.imshow(np.flipud(hk_array_log_barrier[0, :, :]), extent=extent, cmap=hk_array_cmap,
                   vmin=-3, vmax=3) 
        cbr = plt.colorbar()
        cbr.set_label('Log hk (m/d')
        
        axes = plt.gca()
        axes.set_xlim([0, param_dict["Lx"]])
        axes.set_ylim([0, param_dict["Ly"]])
        
        plt.savefig(os.path.join(figureDirectory, name_img))
                               
    #==== RECHARGE CHARACTERISTICS ================================================
    
    number_of_recharge_scenarios = 1
    
    # Recharge array 1 = diffuse recharge across the whole aquifer in equal amounts.
    recharge_array = param_dict["recharge_flux"] * np.ones((param_dict["nrow"], 
                     param_dict["ncol"]), dtype=np.float32)
    
    #==============================================================================
    #=====RUNNING THE FLOW MODEL===================================================  
    #==============================================================================

    
    mf = flopy.modflow.Modflow(modelname, exe_name=mfnwt_exe_name, version='mfnwt') # MODFLOW-NWT
    #mf = flopy.modflow.Modflow(modelname, exe_name=MODFLOW_folder) # MODFLOW-2005
    
    nper = 1 # Number of model stress periods
    perlen = [param_dict["length_simulation"]] 
    
    nstp = [1] # Number of time steps in each stress period
    tsmult = [1.0]
    steady = [True] # True = steady state
    
    dis = flopy.modflow.ModflowDis(mf, param_dict["nlay"], param_dict["nrow"], 
                                   param_dict["ncol"], delr=param_dict["delr"], 
                                   delc=param_dict["delc"], top=param_dict["ztop"], 
                                   botm=param_dict["botm"][1:],
                                   tsmult = tsmult,
                                   nper=nper, perlen=perlen, nstp=nstp, 
                                   steady=steady)
    
    print("Discretisation Module Set Up")
            
    #        # UPSTREAM WEIGHTING PACKAGE
    
    hk_array = hk_array_barrier
    
    uwp = flopy.modflow.mfupw.ModflowUpw(mf, hk=hk_array, vka=param_dict["vka"], 
                                         sy=param_dict["sy"], 
                                         ss=param_dict["ss"], 
                                         laytyp=param_dict["laytyp"]) # MODFLOW- NWT
    
    print("Upstream Weighting Package Set Up")
    
    ## NEWTON SOLVER
    nwt = flopy.modflow.mfnwt.ModflowNwt(mf, headtol=1e-03, fluxtol=0.05, maxiterout=100, 
                thickfact=1e-07, linmeth=2, iprnwt=1, ibotav=1, options='SPECIFIED', 
                Continue=True, dbdtheta=0.9, dbdkappa=1e-05, dbdgamma=0.0, momfact=0.1, 
                backflag=1, maxbackiter=30, backtol=1.05, backreduce=0.9, maxitinner=50, 
                ilumethod=2, levfill=5, stoptol=1e-10, msdr=15, iacl=2, norder=1, level=3, 
                north=7, iredsys=1, rrctols=0.0, idroptol=1, epsrn=0.0001, 
                hclosexmd=0.0001, mxiterxmd=200) # MODFLOW- NWT
    
    print("Newton Solver Set Up")
    
    # --- Recharge --- #
            
    # RECHARGE (RCH) PACKAGE
    
    rch = flopy.modflow.ModflowRch(mf, rech=recharge_array)
    
    # GENERAL HEAD BOUNDARY (GHB) PACKAGE 
    ghb = flopy.modflow.ModflowGhb(mf, stress_period_data=ghb_spd)
    
    print("General Head Boundary Package Set Up")   
                 
    # OUTPUT CONTROL (OC) PACKAGE
                           
    spd = {}
    for strsp in range(nper):
        tmstp = nstp[strsp]
        for time_ in range(tmstp):
            spd[strsp, time_] = ['save head', 'print budget', 
                                   'save budget'] 
                                   
    oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd)
    
    print("Output Control Set Up")
    
    #bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=300)   
    
    bas = flopy.modflow.ModflowBas(mf, ibound=1., strt=300., 
                                   ifrefm=True, ixsec=False, 
                                   hnoflo=-999.99)     
    
    print("Basic Package Set Up") 
    
    # --- LINKING FILE --- #
    
    # Set up the linking file for output from flow model to be used in transport model
    lmt = flopy.modflow.ModflowLmt(mf, output_file_name= (modelname + str('_mt3dLink.ftl')),
                                   output_file_header='extended',
                                   output_file_format='unformatted')   
                                    
    print("Link-Mass Transport Package Set Up") 
      
    mf.write_input() # Write the model input files
    print("input written")
    
    # What's the best way to run the command?
    subprocess.run([mfnwt_exe_name, modelname]) # Run command 
       
    #==============================================================================
    #=====RUNNING THE TRANSPORT MODEL==============================================  
    #==============================================================================
    
    #==== TRACER PARAMETERS =======================================================
    
    # Instantiate MT3D-USGS object in flopy
    
    mt = mt3d.Mt3dms(modflowmodel=mf, modelname=modelname_mt, model_ws=this_model_folder, 
                     version=mt3d_version, namefile_ext='mtnam', exe_name=mt3d_exe_name,  
                     ftlfilename=(modelname + str('_mt3dLink.ftl'))) 
    
    # ftlfree=True. This was on the online code but I can't find anywhere what it means.
    
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
                       delr=param_dict["delr"], delc=param_dict["delc"], 
                       icbund=1, ifmtcn=-1, savucn=True,
                       nprs=nprs, nprobs=1, cinact=0, ssflag='SState', laycon=1) 
    
    #------------------------------------------------------------------------------
    # ADVECTION PACKAGE
    
    # Instantiate advection (ADV) package for MT3D-USGS
    adv = mt3d.Mt3dAdv(mt, mixelm=0)  
    
    # mixelm = 0 means that I am using the standard finite-difference method with upstream or 
    # central-in-stream weighting, depending on the value of NADVFD. 
    # Other options, MOC (1); MMOC (2); HMOC (3); TVD (-1). 
    
    #------------------------------------------------------------------------------
    # GENERALISED CONJUGATE GRADIENT SOLVER for MT3D-USGS
    
    # Instatiate generalized conjugate gradient solver (GCG) package for MT3D-USGS
    gcg = mt3d.Mt3dGcg(mt, mxiter=30, iter1=50, isolve=2, accl=1, cclose=1e-06)
    
    #------------------------------------------------------------------------------
    # REACTION PACKAGE
    
    rc1 = np.zeros((param_dict["nlay"], param_dict["nrow"], param_dict["ncol"]), 
                   dtype=np.float32)

    rc1[:, :, :] = -1/365.25
    
    isothm = 0      # 0 = no sorption
    ireact = 100    # 100 = zeroth-order reaction option
    rc1 = rc1       # First order reaction rate for diffolved phase of first species.
    
    # Setting up the Reaction package
    rct= mt3d.Mt3dRct(mt, isothm=isothm, ireact=ireact, igetsc=0, rc1=rc1) 
    
    #------------------------------------------------------------------------------
    # SOURCE-SINK MIXING PACKAGE
    
    crch = np.zeros((param_dict["nrow"], param_dict["ncol"]), 
                    dtype=np.float32) # The age of recharging water is 0
    
    itype = mt3d.Mt3dSsm.itype_dict()
    #ssm_data = {}
    #ssm_data[0] = [(9, 0, -1, 0., itype['CHD'])]
    #
    #for i in range(nrow):
    #    ssm_data[0].append((9, i, -1, 0., itype['CHD']))
    #
    
    #Instantiate source-sink mixing (SSM) package for MT3D-USGS
    ssm = mt3d.Mt3dSsm(mt, crch=crch) #, stress_period_data=ssm_data) 
          
    #------------------------------------------------------------------------------
    # DISPERSION PACKAGE 
    al = 1.5 # The longitudinal dispersivity, default = 0.01
    #        trpt = 0.1  #ratio of horizontal transverse: longitudinal dispersivity, default = 0.1
    #        trpv = 0.01 #ratio of vertical transverse dispersivity: longitudinal dispersivity, default = 0.01
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
    conc_filename_dissolved = str(modelname_mt) + ".ucn"
    conc_filename_sorbed = str(modelname_mt) + "_S.ucn"
    mass_filename = str(modelname_mt) + ".mas"
    cnf_filename = str(modelname_mt) + ".cnf"
    
    ##Add a line to the MT3D .mtnam file re-naming output UCN file + MAS,CNF
    mt_name_file = modelname_mt + str('.mtnam')
    
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
    mt_btn_file = modelname_mt + str('.btn')
    
    
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
    
    mt3d_namefile = modelname_mt + ".mtnam"
    
    # Running the transport model
    subprocess.run([mt3d_exe_name, mt3d_namefile]) # Run command 
    
    #==============================================================================
    #=====CREATE DATA OUTPUTS======================================================  
    #==============================================================================
     
    # CREATE HEADFILE OBJECTS  
    
    headobj = bf.HeadFile(modelname + '.hds')
    data = headobj.get_alldata()
    times = headobj.get_times()
    head_raster = headobj.get_data(totim=times[0]) # One time period: steady-state
    
    # CONCENTRATION/AGE FILES - FROM TRANSPORT MODEL
    
    directage_file = modelname_mt + str('.ucn') 
    
    ucnobjc_direct_age = flopy.utils.binaryfile.UcnFile(directage_file, precision="double")
    
    age_times = ucnobjc_direct_age.get_times()
    
    age_raster = ucnobjc_direct_age.get_data(totim=age_times[0])
    
    # Plotting options
    head_cmap = 'coolwarm'
    age_cmap = 'jet'
   
    #==============================================================================
    #=====PLOTTING HEAD DATA=======================================================  
    #==============================================================================
     
    # CONTOUR PLOTS 
    
    head_1 = headobj.get_data(totim=times[0]) # One time period: steady-state
    
    levels = np.linspace(0, 400, 51)
    
    # Plot steady state hydraulic heads
    
    plt.figure()
    plt.subplot(1, 1, 1, aspect='equal')    

    plt.imshow(np.flipud(head_1[0, :, :]), extent=extent, cmap='coolwarm_r', 
                         vmin = 200, vmax = 340) # vmax=np.amax(head_1[0, :, :]), vmin = np.amin(head_1[0, :, :])
    
    cbr = plt.colorbar()
    cbr.ax.tick_params(labelsize=14) 
    cbr.set_label('Head in mAHD', fontsize = 14)
    cs = plt.contour((head_1[0, :, :]), levels=levels, extent=extent, 
                               colors = "k") 
    plt.clabel(cs, inline=1, fontsize=14, fmt='%1i')
    
    #for i in range(len(pilot_point_x)):
    #    plt.plot(pilot_point_x[i], pilot_point_y[i], "wo")
    
    # Plot location of observation wells
    plt.savefig(figureDirectory + "\head_real_ss")
    
    for i in range(len(obs_wells_x)):
        plt.plot(obs_wells_x[i], obs_wells_y[i], "m*", markersize=10)
    
    plt.savefig(figureDirectory + "\head_real_ss_withobs")
    
    # Save the original head file - so I can compare it to output with pest
    fname = "original_head_file"
    np.savetxt(os.path.join(dataDirectory, fname), head_1[0, :, :], fmt='%8.6e', delimiter='   ')
       
    #==============================================================================
    #=====PLOTTING K DATA==========================================================  
    #==============================================================================    
    
    levels_k = np.linspace(0, 1, 11)
    
    # Plot K data originally used for the model
    
    plt.figure()
    plt.subplot(1, 1, 1, aspect='equal')    

    plt.imshow(np.flipud(hk_array[0, :, :]), extent=extent, cmap=hk_array_cmap, 
                         ) # vmax=np.amax(head_1[0, :, :]), vmin = np.amin(head_1[0, :, :])
    cbr = plt.colorbar()
    cbr.ax.tick_params(labelsize=14) 
    cbr.set_label('HK (m/d)', fontsize = 14)
    #
    #for i in range(len(pilot_point_x)):
    #    plt.plot(pilot_point_x[i], pilot_point_y[i], "wo")
    
    # Plot location of observation wells
    
    for i in range(len(obs_wells_x)):
        plt.plot(obs_wells_x[i], obs_wells_y[i], "m*", markersize=10)
        
    plt.grid()
    plt.xlabel("x axis (m)", fontsize = 14)
    plt.ylabel("y axis (m)", fontsize = 14)
    
    plt.savefig(figureDirectory + "\hk_real")
    
    #------------------------------------------------------------------------------
    ## Plot "real" hk field as log
    hk_array_log = np.log10(hk_array)
    
    plt.figure()
    plt.subplot(1, 1, 1, aspect='equal')    

    plt.imshow(np.flipud(hk_array_log[0, :, :]), extent=extent, cmap='terrain', 
                         ) # vmax=np.amax(head_1[0, :, :]), vmin = np.amin(head_1[0, :, :])
    cbr = plt.colorbar()
    cbr.ax.tick_params(labelsize=14) 
    cbr.set_label('Log HK (m/d)', fontsize = 14)
    
    for i in range(len(obs_wells_x)):
        plt.plot(obs_wells_x[i], obs_wells_y[i], "m*", markersize=10)
    #    
    plt.grid()
    plt.xlabel("x axis (m)", fontsize = 14)
    plt.ylabel("y axis (m)", fontsize = 14)
    
    plt.savefig(figureDirectory + "\hk_real_log")
    
    #------------------------------------------------------------------------------
    ## Plot "real" hk field as log -> symmetrical colour scale
    hk_array_log = np.log10(hk_array)
    
    plt.figure()
    plt.subplot(1, 1, 1, aspect='equal')    

    plt.imshow(np.flipud(hk_array_log[0, :, :]), extent=extent, cmap='seismic', 
                         vmin = -6, vmax = 6) # vmax=np.amax(head_1[0, :, :]), vmin = np.amin(head_1[0, :, :])
    cbr = plt.colorbar()
    cbr.ax.tick_params(labelsize=14) 
    cbr.set_label('Log HK (m/d)', fontsize = 14)
    
    for i in range(len(obs_wells_x)):
        plt.plot(obs_wells_x[i], obs_wells_y[i], "m*", markersize=10)
    #    
    plt.grid()
    plt.xlabel("x axis (m)", fontsize = 14)
    plt.ylabel("y axis (m)", fontsize = 14)
    
    plt.savefig(figureDirectory + "\hk_real_log_2")
    
    #==============================================================================
    # PLOT AGE WITHOUT BARRIER
    #==============================================================================
    
    plt.figure()
    plt.subplot(1, 1, 1, aspect='equal')    
    
    plt.imshow(np.flipud(age_raster[0, :, :]), extent=extent, cmap=age_cmap, vmin = 0, vmax = 1750) # vmax=np.amax(head_1[0, :, :]), vmin = np.amin(head_1[0, :, :])
    cbr = plt.colorbar()
    cbr.set_label('Age (years)')
    
    plt.savefig(figureDirectory + "\plotage_real")
    
    #
    #for i in range(len(pilot_point_x)):
    #    plt.plot(pilot_point_x[i], pilot_point_y[i], "wo")
    
    for i in range(len(obs_wells_x)):
        plt.plot(obs_wells_x[i], obs_wells_y[i], "m*", markersize=10)
    
    plt.savefig(figureDirectory + "\plotage_real_withobs")
    
    # Save the original age file - so I can compare it to output with pest
    fname = "original_age_file"
    np.savetxt(os.path.join(dataDirectory, fname), age_raster[0, :, :], fmt='%8.6e', delimiter='   ')
    
    #==============================================================================
    #=====PEST=====================================================================  
    #==============================================================================
    
    # PEST SET UP EXAMPLE
    # FROM: https://github.com/jtwhite79/pyemu/blob/develop/examples/
    # notest_modflow_to_pest_like_a_boss.ipynb
    
    # I am using two types of observations:
    #    (1) HEADS DATA
    #    (2) AGE DATA
    
    namefile = modelname + str(".nam")
    headsfile = modelname + str(".hds")
    agefile = modelname_mt + str('.ucn') 
    
    #------------------------------------------------------------------------------
    # Load the existing model
    model_ws = this_model_folder
    ml = flopy.modflow.Modflow.load(namefile, model_ws=model_ws, verbose=False)
    ml.exe_name = "mfnwtdbl.exe" 
    ml.write_input()
    ml.run_model() # I don't think this is the best way to run the model...
       
    #------------------------------------------------------------------------------
    # Instruction Files  
    #------------------------------------------------------------------------------
    # Getting head data for each observation well
    
    # Check row data of observation wells from csv 
    rc_df.head()
    
    # Get all of the head data
    hds = bf.HeadFile(os.path.join(ml.model_ws, headsfile)) 
    ages = flopy.utils.binaryfile.UcnFile(agefile, precision="double")
    
    heads_data = hds.get_alldata()
    heads_data.shape # (time period, layer, row, column)
    ages_data = ages.get_alldata()
    ages_data.shape # (time period, layer, row, column)
    
    obs_names = []
    obs_names_head = []
    obs_names_age = []
    obs_times = []
    obs_heads = []
    obs_ages_noise = []
    obs_ages_nonoise = []
    obs = []
    
    roff = 0.0 # ml.dis.delc.array[0] / 2.0
    coff = 0.0 # ml.dis.delr.array[0] / 2.0
    # roff is the fractional offset from the centre of the cell in Y direction (between rows)
    # coff is the fractional offset from the centre of the cell in the X direction (between cols)
    
    obs_name_format = "i{1:03d}j{2:03d}"
    n = rc_df.a_wellname[0]
    r = rc_df.b_wellrow[0]
    c = rc_df.c_wellcol[0]

    for n, r, c in zip(rc_df.a_wellname, rc_df.b_wellrow, rc_df.c_wellcol):
        
        # Don't need to minus 1 from col and row because i've converted from 0-based
        # above (it's currently in the correct row, col format for modflow)
        # String formatting: {field_name:conversion} d = decimal integer.
        
        name = obs_name_format.format(n, r, c) 
        print(name)
        obs_names.append(name)
        obs_head_name = str(name)+"_h"
        obs_names_head.append(obs_head_name)
        obs_names_age.append(str(name)+"_a")
        
        # Create empty data shape to put the head data into.
        # heads_data.shape is referring to my heads data above, so it's the same format. 
        # So one row (only one time period = steady-state, and two columns for time and head)
        d = np.zeros((heads_data.shape[0], 2)) 
        print(d)
        
        # Now put the times into the first column of empty matrix.
        # From 1 onwards, get rid of steady-state data.
        d[:,0] = (hds.times[0] - 1) # I subtracted 1 because need to start at time period 0.
        obs_times.append(hds.times[0] - 1)
    
        # Now add the head data to the second column of the empty matrix.
        # and add some random noise to the observations.
        
        d[:, 1] = heads_data[0, 0, r, c] #+ np.random.randn(d.shape[0]) 
        obs_heads.append(heads_data[0, 0, r, c]) # #+ np.random.randn(d.shape[0])[0]
    
        # Now add the age data to the second column of the empty matrix.
        # and add some random noise to the observations.
        obs_ages_noise.append(ages_data[0, 0, r, c]) # + np.random.randn(d.shape[0])[0]) 
        obs_ages_nonoise.append(ages_data[0, 0, r, c])
        
        # Now create a list of each obs - these are zero based and flopy converts them to 1-based (for modflow)
        obs.append(flopy.modflow.mfhob.HeadObservation(ml, obsname=obs_head_name, 
                                                       layer=0, row=r,
                                                       column=c, roff=roff, coff=coff,
                                                       time_series_data=d)) 
        
    flopy.modflow.mfhob.ModflowHob(ml, obs_data=obs, iuhobsv=600) 
    
    ml.write_input()
    
    #os.system("mfnwtdbl.exe " + str(namefile))
    
    ml.run_model() # To get hob.out file. If this isn't working, check names of hob obs - they can't have'-' in them.
    
    all_observations = pd.DataFrame({
                                      
                "obs_name": obs_names,
                "obs_time": obs_times,
                "obs_heads": obs_heads,
                "obs_ages": obs_ages_noise
                                            })
    
    head_obs = pd.DataFrame({
                                      
                "a_obs_name": obs_names_head,
                "b_obs_time": obs_times,
                "c_obs_heads": obs_heads 
                                            })
    
    age_obs = pd.DataFrame({
                                      
                "a_obs_ages": obs_ages_noise,            
                "b_obs_name": obs_names_age
    
                                            })
    
    # Export as csv
    fileName = os.path.join(this_model_folder, "all_obs.csv")
    all_observations.to_csv(fileName, encoding='utf-8', index=False)
    all_observations.index = all_observations.obs_name
    
    # Export as text file --> 
    # THESE ARE THE OBSERVATION DATA AS BASIS FOR INSTRUCTION FILES
    # OR USE HOB FILE 
    
    fileName_text = os.path.join(this_model_folder, "all_obs.txt")
    all_observations.to_csv(fileName_text, header=None, index=None, mode='a', sep=' ')
    
    fileName_text = os.path.join(this_model_folder, "head_obs.txt")
    head_obs.to_csv(fileName_text, header=None, index=None, mode='a', sep=' ')
    
    fileName_text_out = os.path.join(this_model_folder, "age_obs.out")
    # Delete previous instances, because it adds to them rather than writing a new one.
    if os.path.exists(fileName_text_out):
        print("deleting old age output file")
        os.remove(fileName_text_out) 
        
    age_obs.to_csv(fileName_text_out, header=None, index=None, mode='a', sep=' ')
    
    # Plot head & age obs -----------------------------------------------------------
    h_plotting = []
    a_plotting = []
    for n, r, c in zip(rc_df.a_wellname, rc_df.b_wellrow, rc_df.c_wellcol):
        name = obs_name_format.format(n, r, c) 
        h_plotting.append(all_observations.loc[name, "obs_heads"])
        a_plotting.append(all_observations.loc[name, "obs_ages"])
        
    plt.figure(figsize=(10,10))
    plt.suptitle("Head and age data used for obs", fontsize=20)
    plt.subplot(2, 1, 1)
    axes1 = plt.gca()

    rc_df.plot.scatter(x="c_wellcol", y="b_wellrow", c=h_plotting, s=30, 
                       colormap="coolwarm_r", linewidths=0.0, ax=axes1)

    axes1.set_xlim([0, param_dict["Lx"]/param_dict["delr"]])
    axes1.set_ylim([0, param_dict["Ly"]/param_dict["delc"]])
    plt.xlabel("Columns", fontsize=20)
    plt.ylabel("Rows", fontsize=20)

    
    plt.subplot(2, 1, 2)
    axes2 = plt.gca()

    rc_df.plot.scatter(x="c_wellcol", y="b_wellrow", c=a_plotting, s=30, 
                       colormap=age_cmap, linewidths=0.0, ax=axes2)

    axes2.set_xlim([0, param_dict["Lx"]/param_dict["delr"]])
    axes2.set_ylim([0, param_dict["Ly"]/param_dict["delc"]])
    plt.xlabel("Columns", fontsize=20)
    plt.ylabel("Rows", fontsize=20)
    
    plt.savefig(os.path.join(figureDirectory, "a_h_obs_w_noise"))
    
    # - - - - - - - - - - - - - - - - - - - - -
    # Make PEST Instruction files.
    # Instruction files: for every model output file containing observations, you need
    # to provide an instruction file containing directions which PEST must follow in order
    # to read that file.
    
    # Make instruction file for heads from hob.out file.
    # this also will load observations into dataframe
    hob_df = pyemu.gw_utils.modflow_hob_to_instruction_file(os.path.join(ml.model_ws,ml.name+".hob.out"))
    
    hob_df.head() # Prints the first five lines to check it out
    
    # Change observation names in hob df
    
    hob_df.index = hob_df.obsnme
    hob_df.head()
    
    # Change parameter group name in hob df
    
    head_group_name = "head"
    head_group_name_list = []
    
    for i in range(len(obs_names_head)):
        head_group_name_list.append(head_group_name)
    
    hob_df["obgnme"] = head_group_name_list
    hob_df.head()
    
    
    weight_head = 25.0
    hob_df["weight"] = weight_head
    
    # - - - - - - - - - - - - - - - - - - - - -
    
    # Make instruction file for age_obs out file
    # Also load obs into a dataframe
    age_weight = 1.0
    age_group_name = "age" 
    
    ins_line_age = []
    weight_age = []
    obgnme_age = [] 
    for i in range(len(obs_names_age)):
        ins_line_age.append("l1 !%s!" % obs_names_age[i])
        weight_age.append(age_weight)
        obgnme_age.append(age_group_name)
        
    
    age_df = pd.DataFrame({                                  
                           "simval": obs_ages_noise,            
                           "obsval": obs_ages_nonoise, 
                           "obsnme": obs_names_age,
                           "ins_line": ins_line_age,
                           "weight": weight_age,
                           "obgnme": obgnme_age
                                                    })
    
    fileName_text = os.path.join(this_model_folder, "age_obs.out.ins")
    age_df.to_csv(fileName_text, columns=["ins_line"], header=True, index=False)
    
    # Now I need to change the header "ins_line" to "pif~"
    
    fpath = fileName_text
    with open(fpath, 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
    
        a = "ins_line" 
        b = "pif ~"
     
        ##
    
        for line in lines:
            if a in line:
                            line = line.replace(a, b)
            f.write(line)
    
    # Now I have automatically made my instruction file for age data, assuming that the 
    # output is in the same order as the instruction file! I'm happy with this.
    
    #--------------------------------------------------------------------------
    # TEMPLATE FILES ---> for each parameter
    #--------------------------------------------------------------------------
    
    # Name the new external path: "ref"
    # This is the path that my model will access when it runs the model.
    ext_path = os.path.join(ml.model_ws, "ref")
    
    # Resetting my input files so that they're getting external output during PEST runs.
    # Put all external files in directory: "ref".
    
    # See page 8-57 of MODFLOW-2005 manual:
        
    '''
        FMTIN—is the format for reading array elements. The format must contain 20 characters or less. The format must
    either be a standard Fortran format that is enclosed in parentheses, "(FREE)" which indicates free format, or
    "(BINARY)" which indicates binary (unformatted) data. When using a free-format control line, the format must be
    enclosed in apostrophes if it contains one or more blanks or commas.
    '''
    
    # Recursively delete a directory tree if it already exists so I'm not using old data.
    if os.path.exists(ext_path):
        shutil.rmtree(ext_path) 
    
    # Now make a new external directory:
    os.mkdir(ext_path)
    
    # Now set the external path associated with my model (don't need the full path just the final "ref")
    ml.external_path = os.path.split(ext_path)[-1]
    
    # Add same thing but for mt3d model
    # ---> for the time being I don't need this as not estimating mt3d parameters
    mt.external_path = os.path.split(ext_path)[-1]
    
    # Set the format for readying array elements to external for parameters of interest.
    # Can check the ref folder to see that all the parameter files are formed.
    ml.upw.hk.fmtin = "(FREE)" 
    #ml.upw.sy.fmtin = "(FREE)" # Only relevant for transient sims.
    ml.rch.rech.fmtin = "(FREE)"
    
    # This will write the external files in the "ref" folder.
    ml.write_input()
    
    # - - - - - - - - - - - - - - - - - - - - -
    
    # FIRST: RESET HK ARRAY TO JUST ONES AS THE BASIS FOR STARTING PEST.
    # Make a hk file that is all just ones. This will be changed with each pest run.
    hk_array_bak = param_dict["hk_aquifer"] * np.ones((param_dict["nlay"], 
                                         param_dict["nrow"], 
                                         param_dict["ncol"]), dtype=np.float32)  
    
    ## UPSTREAM WEIGHTING PACKAGE
    uwp = flopy.modflow.mfupw.ModflowUpw(ml, hk=hk_array_bak, vka=param_dict["vka"], 
                                         sy=param_dict["sy"], ss=param_dict["ss"], 
                                         laytyp=param_dict["laytyp"]) # MODFLOW- NWT
    
    
    ml.write_input()
    
    # - - - - - - - - - - - - - - - - - - - - -
    
    # Set up a backup of all original model files (in directory "bak" ) - > 
    # these will be copied to "ref" diretory, then altered with pest model run.
    
    ext_files = [f for f in os.listdir(ext_path)]
    
    bak_path = os.path.join(ml.model_ws,"bak")
    
    # Delete any old versions of this folder
    if os.path.exists(bak_path):
        shutil.rmtree(bak_path)
        
    # Create a new version of the folder
    os.mkdir(bak_path)
    
    # copy2 is identical to copy (copies file to a directory) but attempts to preserve file metadate
    for f in ext_files:
        shutil.copy2(os.path.join(ext_path, f),os.path.join(bak_path, f))
    
    # It is these files in bak that I will alter during pest model run...
    
    # - - - - - - - - - - - - - - - - - - - - -
    """
    Not sure if I need to do this...


    # SECOND: RESET HK ARRAY TO HAVE BARRIER - FOR HOB FILE...
    # I think I need to do this? This should not change the files in bak
    
    ## UPSTREAM WEIGHTING PACKAGE
    uwp = flopy.modflow.mfupw.ModflowUpw(ml, hk=hk_array, vka=param_dict["vka"], 
                                         sy=param_dict["sy"], ss=param_dict["ss"], 
                                         laytyp=param_dict["laytyp"]) # MODFLOW- NWT
    
    ml.write_input()
    
    # Re-run the model using external files --> is this a check to see it's worked maybe?
    ml.run_model()
    """
    
    # See if I can do the same thing for mt3d model
    ### Actually I don't need to --> because there are no parameters in mt3d it's
    ### only output
    
    ## - - - - - - - - - - - - - - - - - - - - -
    if phantom_or_pilot == "Phantom_faults":
        print("Using phantom faults")
        
        if include_real_with_ph == True:
            # If I am including real fault in the inversion
            par_names = ["hkaq"] + rl_fault_name_list + ph_fault_name_list
            par_values = [param_dict["hk_aquifer"]] + [phantom_fault_K] + phantom_fault_K_list
        
        elif include_real_with_ph == False:        
            # If not including real fault in the inversion
            par_names  = ["hkaq"] + ph_fault_name_list
            par_values = [param_dict["hk_aquifer"]] + phantom_fault_K_list
       
        else:
            print("Something funky - choose if you want the real fault included")
        # Make template file for hk of barriers
        
        fileName_templ = os.path.join(this_model_folder, "hk1pp.dat.tpl")
        
        template_file = open(fileName_templ, "w+")
        
        template_file.write("ptf ~")
        
        for i in range(len(par_names)):
            template_file.write("\n~%s     ~" % par_names[i])
                
        template_file.close()
        
        # Make input file for hk of barriers
        
        fileName_input = os.path.join(this_model_folder, "hk1pp.dat")
        
        input_file = open(fileName_input, "w+")
        
        for i in range(len(par_values)):
            input_file.write("%.2f \n" % par_values[i])
            
        input_file.close()
        plt.close("all")
    
    elif phantom_or_pilot == "Pilot_points":
        print("Using pilot points")
    
        # - - - - - - - - - - - - - - - - - - - - -
        # PILOT POINTS 
        
        # first specify what pilot point names we want to use for each model layer (counting from 0). 
        prefix_dict= {0:["hk1"]} # , "rech1" Can also include "sy1", --> but only relevant for transient sima.
        
        # How regularly do we want the pilot points to be spaced? See "every_n_cell" in function below.
        pp_cells = 10
        
        # Set up a regular-spaced (gridded) pilot point parameterisation
        pp_df = pyemu.utils.pp_utils.setup_pilotpoints_grid(ml=ml, prefix_dict=prefix_dict, 
                                                            every_n_cell=pp_cells, 
                                                            use_ibound_zones=False, pp_dir='.', 
                                                            tpl_dir='.', shapename='pp.shp') 
        
        '''
        NOTE ON ZONES! : This function 'pyemu.utils.pp_utils.setup_pilotpoints_grid' 
        has a "use_ibound zones option"– a flag to use the greater-than-zero values in the ibound as pilot point zones. 
        If False,ibound values greater than zero are treated as a single zone. Default is False.
        
        I can use this to test zones with my low-permeability zones
        '''
        
        # Change index to the name of the parameter
        pp_df.index = pp_df.parnme
    
    else:
        print("There's an error - what type of regularisation do you want?")
    #--------------------------------------------------------------------------
    # GEOSTATS AND KRIGING --> Setting up factor file
    #--------------------------------------------------------------------------
    if phantom_or_pilot == "Phantom_faults":
        print("Don't need to do any kriging")
        
    elif phantom_or_pilot == "Pilot_points":
        print("Doing kriging because I am using pilot points")

        '''
         --> Need to get value of parameters from an array of whole grid to pp locations
         and into template input files
         So we...
         need to set up the kriging factors to solve kriging equations for each 
         model cell using pilot point locations.
        '''
        
        # Read the pilot point file to a pandas Dataframe for hk 
        # (we only need to get kriging factors once)
        hk_pp = pyemu.pp_utils.pp_file_to_dataframe(os.path.join(ml.model_ws, "hk1pp.dat"))
        hk_pp.head()
        
        # Set up geostatistical structure
        # Since all pilot points will be log transformed, good idea to use log-based 
        # geostatistical structure
        a =  pp_cells * ml.dis.delr.array[0] * 3.0
        v =  pyemu.utils.ExpVario(contribution=1.0, a=a)
        gs = pyemu.utils.GeoStruct(variograms=v,transform="log")
        
        # Create OrdinaryKrige object
        ok = pyemu.utils.OrdinaryKrige(geostruct=gs,point_data=hk_pp)
        
        # Use helper function to solve kriging factors for each active model cell
        ok.calc_factors_grid(ml.sr, zone_array=ml.bas6.ibound[0].array,
                             var_filename=os.path.join(ml.model_ws,"layer1_var.dat"))
        
        # Write the kriging factors to a factors file
        ok.to_grid_factors_file(os.path.join(ml.model_ws,"pp.fac"))
        
        # Check out the kriging variance array
        # ma = masked array, get's rid of NaNs
        var_arr = np.ma.masked_invalid(np.loadtxt(os.path.join(ml.model_ws,"layer1_var.dat")))
        
        # Plot the kriging variance array
        fig = plt.figure(figsize=(20,20))
        ax = plt.subplot(111,aspect="equal")
        ax.pcolormesh(ml.sr.xcentergrid,ml.sr.ycentergrid,var_arr,alpha=0.5)
        ax.scatter(hk_pp.x, hk_pp.y,marker='.',s=10)
        plt.xlim(0, param_dict["Lx"])
        plt.ylim(0, param_dict["Ly"])

        ml.sr.xcentergrid[0,0],ml.sr.ycentergrid[0,0]
        hk_pp.iloc[0,:].values
        plt.savefig(os.path.join(figureDirectory, "kriging_array"))

        plt.close("all")
    
    else:
        pass
    #--------------------------------------------------------------------------
    # Control File  
    #--------------------------------------------------------------------------
    
    ## Building the PEST control file
    # Use the template and instruction files to construct a control file
    
    # Make sure template/input file pairs have consistent naming.
    
    ## FOR SOME REASON PEST ++ HAS AN ISSUE WITH LONG VERSIONS OF THE FILE PATH, JUST USE THE END PART
    
    # For pest - whole file path
#    tpl_files = [os.path.join(ml.model_ws,f) for f in os.listdir(ml.model_ws) if f.endswith(".tpl")]           
#    input_files = [f.replace(".tpl",'') for f in tpl_files]
#    
#    # Same thing for instruction/output file pairs
#    ins_files = [os.path.join(ml.model_ws,f) for f in os.listdir(ml.model_ws) if f.endswith(".ins")]
#    output_files = [f.replace(".ins",'') for f in ins_files]
    
    # For PEST++ - shortened file path
    tpl_files = [os.path.join(f) for f in os.listdir(ml.model_ws) if f.endswith(".tpl")]           
    input_files = [f.replace(".tpl",'') for f in tpl_files]
    
    # Same thing for instruction/output file pairs
    ins_files = [os.path.join(f) for f in os.listdir(ml.model_ws) if f.endswith(".ins")]
    output_files = [f.replace(".ins",'') for f in ins_files]
    
    # - - - - - - - - - - - - - - - - - - - - -
    # MAKING PEST INSTANCE (using template and instruction files)
    
    # Copy pest inschek into folder
    inschek_folder = os.path.join('C:\workspace\Proj3_PilbaraBarriers\pest_files', 'inschek.exe')
    this_model_inschek = os.path.join(ml.model_ws,"inschek.exe")
    
    if not os.path.isfile(this_model_inschek):
        print("inschek not yet in folder")
        shutil.copy2(inschek_folder, this_model_inschek)
    else:
        print("inschek is already in folder")
    
    
    # Can include pst_filename here, which will automatically write the pest file,
    # but if none is provided, the default is None and pest file not written.
    pst = pyemu.utils.helpers.pst_from_io_files(tpl_files,
                                                input_files,
                                                ins_files,
                                                output_files)
    
    # Here at this step - I am getting the generation of a different hob output file - 
    # the UB_15_y.hob.out.obf --> why is this being formed???
    
    
    #I am getting this error, don't know why...: 
    #    error using inschek for instruction file C:\workspace\Proj3_PilbaraBarriers\UB_01_h\age_obs.ins:run() 
    #    returned non-zero observations in this instruction file will havegeneric values.
    #    
    #BUT,, if I run inschek in command line with: inschek age_obs.ins age_obs.out
    #Then I don't get any errors and I can check the instruction file was read correctly...
    
    
    # Check the parameter data section - all naming is the same as PEST manual
    
    # ------------------------------------
    # Normal score transform
    # ------------------------------------
    
    # Here, I am doing things a bit differently from other models, I want the parameters to be between 0 and 1
    # because I am going to convert the value that PEST determines to a K value using a 
    # cumulative distribution function.
    
    pst.parameter_data.head()
    
    # Change parameter transformation, because I no longer want it to be log
    pst.parameter_data.loc[:, "partrans"] = "none"
    
    # Change upperboundary to be same as aquifer --> This means that all faults have to be barriers
    pst.parameter_data.loc[:,"parubnd"] = 1
    pst.parameter_data.loc[:,"parlbnd"] = 1e-10 

    # Check the observation data section - all naming same as PEST manual
    pst.observation_data.head()
    
    # - - - - - - - - - - - - - - - - - - - - -
    # Changing observation group names
     
    '''
    Made an age dataframe (above) and combine them...
    '''
    hob_and_age_df = hob_df.append(age_df)
    hob_and_age_df.index = hob_and_age_df.obsnme
    
    hob_and_age_df.index        # Testing, these should be equal
    indexes = pst.observation_data.index  # Testing, these should be equal
    
    # Change the obs group name in the pest instance
    pst.observation_data.loc[hob_and_age_df.index,"obgnme"] = hob_and_age_df.obgnme
    pst.observation_data.loc[hob_and_age_df.index, :].head()
    
    # Change the weights in the pest instance
    
    pst.observation_data.loc[hob_and_age_df.index,"weight"] = hob_and_age_df.weight
    
    # - - - - - - - - - - - - - - - - - - - - -
    
    # At the moment the observation data in pst is the simulated equivalent, not the observed data
    # We need to get the actual observed data into obsval 
    
    pst.observation_data.loc[hob_and_age_df.index, "obsval"] = hob_and_age_df.obsval
    
    # - - - - - - - - - - - - - - - - - - - - -
    
    # Need to set initial values for parameters, these are given default values of 1.0
    
    # Default values of 1.0 are probably fine for me for now as the model is set up
    # to have a default hk of 1.0
    
    # - - - - - - - - - - - - - - - - - - - - -
    # Changing parameter group names
    par = pst.parameter_data #just a pointer to the full, long-named attribute
    
    if phantom_or_pilot == "Phantom_faults":
        # Set parameter group names using the pilot point file we created earlier.
        for i in range(len(par)):
            par.loc[par.parnme[i], "pargp"] = "hk1"

    elif phantom_or_pilot == "Pilot_points":            
        # Set parameter group names using the pilot point file we created earlier.
        par.loc[pp_df.parnme, "pargp"] = pp_df.pargp
        pst.parameter_data.head()

    else:
        print("Not naming parameters - you need to choose a regularisation type")
    # - - - - - - - - - - - - - - - - - - - - -
    # Change PEST model command 
    # Make sure that I write and have a file "forward_run.py"
    
    forward_run_name = modelname + "_forward_run.py"
    pst.model_command = ["python " + str(forward_run_name)]
    
    # - - - - - - - - - - - - - - - - - - - - -
    # MAKE FINAL CHANGES TO CONTROL FILE
    
    # Save final version of the control file
    control_file_name = modelname + ".pst"
    pst.write(os.path.join(ml.model_ws, control_file_name))
       
    #--------------------------------------------------------------------------
    # Forward Run File  
    #--------------------------------------------------------------------------
    
    # Copy and paste forward run file into folder and change the model name in the script
    if phantom_or_pilot == "Phantom_faults":
        # Copy forward run file
        if include_real_with_ph == True:
            stock_forward_run_file = os.path.join(r'C:\SMarshall_PhD\PythonScripts\Proj3_PilbaraBarrier\forward_run_files', 
                                                  'UB_stock_forward_run_real.py')
        else: # include_real_with_ph == False:
            stock_forward_run_file = os.path.join(r'C:\SMarshall_PhD\PythonScripts\Proj3_PilbaraBarrier\forward_run_files', 
                                        'UB_stock_forward_run.py')
    
    elif phantom_or_pilot == "Pilot_points":
        # Copy forward run file
        stock_forward_run_file = os.path.join(r'C:\SMarshall_PhD\PythonScripts\Proj3_PilbaraBarrier\forward_run_files', 
                                                  'UB_stock_forward_run_plt.py')    
    else:
        print("No forward run file if regularisation type isn't assigned")
    
    new_forward_run_file = os.path.join(this_model_folder, forward_run_name)
    
    if not os.path.isfile(new_forward_run_file):
        print("forward run file being copied into folder")
        shutil.copy2(stock_forward_run_file, new_forward_run_file)
    else:
        print("forward run file already in folder")
        
    fpath = new_forward_run_file
    with open(fpath, 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
    
        a = "insert_name" 
        b = modelName
        
        for line in lines:
            if a in line:
                line = line.replace(a, b)
            f.write(line)
                    
    with open(fpath, 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
    
        a = "insert_obs_format"
        b = obs_name_format
        
        for line in lines:
            if a in line:
                line = line.replace(a, b)
            f.write(line)
            
    with open(fpath, 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
    
        a = "insert_ph_format"
        b = fault_name_format
        
        for line in lines:
            if a in line:
                line = line.replace(a, b)
            f.write(line)
            
    with open(fpath, 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
    
        a = "insert_normal_score_transform_used"
        b = is_normal_score_transform_used
        
        for line in lines:
            if a in line:
                line = line.replace(a, b)
            f.write(line)
                               
    ### Ensure in forward run file that the number of phantom faults,
    # number of rows, columns and layers, etc. are correct.
      
    #--------------------------------------------------------------------------
    # Setting up Tikhonov regularisaion 
    #--------------------------------------------------------------------------
    
    # Set prior information for hydraulic conductivity and recharge
    # (By putting in no param groups it defaults to all adjustable params)
    pyemu.utils.helpers.zero_order_tikhonov(pst) # , par_groups=["hk1"]
    
    # Look at the prior information
    pst.prior_information.head()

   
    # I am not sure that I want to do this! --> This is setting information 
    # about how similar each value next to each other should be.
    '''
    # Now, let's set preferred difference equations for pilot point groups. 
    # We will use the Pearson coef as the weight...
    pp_groups = pp_df.groupby("pargp").groups
    
    for pargp,par_names in pp_groups.items():
        this_pp_df = pp_df.loc[par_names,:]
        cov = gs.covariance_matrix(this_pp_df.x,this_pp_df.y,this_pp_df.parnme)
        pyemu.helpers.first_order_pearson_tikhonov(pst,cov,reset=False,abs_drop_tol=0.2)
    
    pst.prior_information
    '''
    
    # Change the pest mode to regularisation
    pst.control_data.pestmode = "regularization"

    # Change PHIMLIM
    prev_phi_optimised = 5.4E+06 # ub17a
    phimlim = 1e-10  # prev_phi_optimised*1.1 # 1e-10 # 1e6
    
    pst.reg_data.phimlim = phimlim 
    pst.reg_data.phimaccept = phimlim*1.1

    # BUT NOTE THAT I AM NOT SURE HOW TO ROUND THESE AND SO I JUST CUT TO 1 OR 2 DP IN THE CONTROL FILE...
    
    
    # Some other params to change
    pst.control_data.phiredstp = 0.01
    pst.control_data.nphistp = 3                  
    
    # Now save the new control file that has regularisation (then you can compare)
    control_file_name_reg = modelname + "_reg" + ".pst"
    pst.write(control_file_name_reg)
    
    
    '''
    Check all the input files, run pest check and now you're good to run your pest run!
    '''
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Change the options in the control file for interim saving --> too many files!
    
    fpath = os.path.join(this_model_folder, control_file_name_reg)
    with open(fpath, 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
    
        a = "jcosaveitn" 
        b = "nojcosaveitn"
        
        for line in lines:
            if a in line:
                line = line.replace(a, b)
    
            f.write(line)
            
    fpath = os.path.join(this_model_folder, control_file_name_reg)
    with open(fpath, 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
    
        a = "reisaveitn" 
        b = "noreisaveitn"
        
        for line in lines:
            if a in line:
                line = line.replace(a, b)
    
            f.write(line)
            
    fpath = os.path.join(this_model_folder, control_file_name_reg)
    with open(fpath, 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
    
        a = "parsaveitn" 
        b = "noparsaveitn"
        
        for line in lines:
            if a in line:
                line = line.replace(a, b)
    
            f.write(line)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            
    # Copy pestchek into folder
    pestchek_folder = os.path.join('C:\workspace\Proj3_PilbaraBarriers\pest_files', 'pestchek.exe')
    this_model_pestchek = os.path.join(ml.model_ws,"pestchek.exe")
    
    if not os.path.isfile(this_model_pestchek):
        print("pestchek not yet in folder, so it's being copied")
        shutil.copy2(pestchek_folder, this_model_pestchek)
    else:
        print("pestchek is already in folder")
    
    # Run PEST chek    
    pestchek = subprocess.run([this_model_pestchek, control_file_name_reg],
                              stdout=subprocess.PIPE, universal_newlines=True) # Run command 
    print(pestchek.stdout)    
    
    # Copy pest into folder
    pest_folder = os.path.join('C:\workspace\Proj3_PilbaraBarriers\pest_files', 'pest.exe')
    this_model_pest = os.path.join(ml.model_ws,"pest.exe")
    
    if not os.path.isfile(this_model_pest):
        print("pest not yet in folder, so it's being copied")
        shutil.copy2(pest_folder, this_model_pest)
    else:
        print("pest is already in folder")
        
    # Copy pest++ into folder
    pestpp_folder = os.path.join('C:\workspace\Proj3_PilbaraBarriers\pest_files', 'pest++.exe')
    this_model_pestpp = os.path.join(ml.model_ws,"pest++.exe")
    
    if not os.path.isfile(this_model_pestpp):
        print("pest++ not yet in folder, so it's being copied")
        shutil.copy2(pestpp_folder, this_model_pestpp)
    else:
        print("pest is already in folder")      


    print("...................")
    print("Run # %d" %run)
    print("Obs option is: %d" %obs_optn)
    print("Ph option is: %d" %ph_optn)        
    print("Finished set up for: %s" %modelname)
    
    ## Note that if I am putting in key faults from previous inversion only,
    # I need to change the overall folder and number of phantom faults in the forward run file"
    
    ###########################################################################
    ###########################################################################
    '''
    # # # RUN PEST
    #
    pestrun = subprocess.run([this_model_pest, control_file_name_reg],
                              stdout=subprocess.PIPE, universal_newlines=True) # Run command 
    #
    print(pestrun.stdout)
    '''
    ###########################################################################
    ###########################################################################
    
        
        
    

