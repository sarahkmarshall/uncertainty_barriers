
"""
This is a script to post-process the results for the PEST run that I sent up in 
the corresponding script (without the 'PP')
"""

#=====IMPORTS==================================================================

import flopy
import flopy.utils.binaryfile as bf
import matplotlib.pyplot as plt
import numpy as np
import os, shutil
import pandas as pd
import pyemu
import sys
import string
import matplotlib.cm as cm
import subprocess
import time
from flopy import mt3d
import math 
from scipy import stats

import pickle

functions_path = "C:\workspace\sarahs_functions"

if any(functions_path in s for s in sys.path):
    print("Functions path already in list")
else:
    sys.path.append(r"C:\workspace\sarahs_functions")

import functions_proj_3

#=====SETTING UP MULTIPLE RUNS=================================================

already_run_this_script = False

savedata_question = 'NO' # 'YES' or 'NO'

n_runs = 3

### VARIABLES 

n_real_faults = 1 

hk_real_fault = 1e-3

hk_array_cmap = 'BrBG'

#=====DIRECTORIES==============================================================

proj_3_folder       = r'C:\workspace\Proj3_PilbaraBarriers'

overallName = "UB_27"

overarching_model_folder = os.path.join(proj_3_folder, overallName)

nruns = [0, 1, 6, 7, 12, 13]
run = nruns[0] 
for run in nruns:
    print(run)
    
    modelname = overallName + "_" + string.ascii_lowercase[run] + "_plt"
    print(modelname)
    
    modelname_mt = modelname + "_MT3D" 
    print(modelname_mt)
    
    savedata_question = 'NO' # 'YES' or 'NO'
    
    MODFLOW_folder      = r'C:\workspace\modflow_dbl\mfnwtdbl.exe' # double precision MODFLOW NWT
    mf_exe              = "mfnwtdbl.exe" 
    
    MT3D_USGS_folder     = r'C:\workspace\modflow_dbl\mt3dusgsdbl.exe' # Double precision MT3D-USGS
    mt3d_version = 'mt3d-usgs'  
    mt3d_exe            = 'mt3dusgsdbl.exe'
    
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
    
    pst_name_reg = str(modelname) + '_reg.pst'
    
    os.chdir(this_model_folder)
    
    modelName_noPLT = overallName + "_" + string.ascii_lowercase[run] 
    dataDirectory_noPLT = os.path.join(proj_3_folder, modelName_noPLT, 'Data')
    
    #====='REAL' MODEL ARRAYS==================================================
        
    # Load original age file
    original_age_file = os.path.join(dataDirectory_noPLT, "original_age_file")
    original_age_file = np.loadtxt(original_age_file)
    
    # Load original head file
    original_head_file = os.path.join(dataDirectory_noPLT, "original_head_file")
    original_head_file = np.loadtxt(original_head_file)
    
    # Load original hk file
    original_hk_file = os.path.join(dataDirectory_noPLT, "original_hk_file")
    original_hk_file = np.loadtxt(original_hk_file)
    
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
    pest_par_file.index = pest_par_file.param_name
    
    
    pest_par_file.index[0] # Check that it hasn't got a weird index - I fixed this with encoding 'utf-8-sig'
    
    
    # Get obs well locations-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    
    # These are modflow based
    
    # I think I should change the name of "pest_obswell_file"! It's not really from PEST
    rc_df_filename  = os.path.join(dataDirectory, 'obs_rowcol.csv') #filtered
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
    
    
    # Get pilot point locations-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    pest_pphk_filename = os.path.join(this_model_folder, 'hk1pp.dat')
    
    pest_pphk_file = pd.read_csv(pest_pphk_filename, header=None, 
                                delim_whitespace=True, skiprows=0, encoding='utf-8-sig') 
    
    pest_pphk_file.columns = ["pp_identifier", "easting", "northing", "zone", "pp_value"]
    
    pilot_point_x = pest_pphk_file["easting"]
    pilot_point_y = pest_pphk_file["northing"]
    
    #==========================================================================
    #  EXPLORING THE FINAL PARAMETER VALUES
    #==========================================================================
    # Importing kriged parameter files from PEST (the file where pilot point estimates
    # have been interpolated using fac2real)
    
    # Determine the final files to access
    #hk_file = os.path.join(this_model_folder, "ref", "hk_Layer_1.ref")
    hk_file = os.path.join(final_pest_files_folder, "hk_Layer_1.ref")
    #rech_file = os.path.join(this_model_folder, "ref", "rech_0.ref")
    #rech_file = os.path.join(final_pest_files_folder, "rech_0.ref")
    
    # Load hk file
    hk_final_array = np.loadtxt(hk_file)
    
    # Load rech file
    #rech_final_array = np.loadtxt(rech_file)
    
    # Plotting hk as a log ####################################################
    
    log_interp_pest_hk = np.log10(hk_final_array)
    
    plt.figure()
    plt.subplot(1, 1, 1, aspect='equal')    
    extent = (param_dict["delr"]/2., param_dict["Lx"] - param_dict["delr"]/2., 
              param_dict["delc"]/2., param_dict["Ly"] - param_dict["delc"]/2.)
    plt.imshow(np.flipud(log_interp_pest_hk[ :, :]), extent=extent, cmap='terrain',
                         vmin = -3, vmax = 0) # vmax=np.amax(head_1[0, :, :]), vmin = np.amin(head_1[0, :, :])
    cbr = plt.colorbar()
    cbr.ax.tick_params(labelsize=14) 
    cbr.set_label('LOG Estimated K (m/d)', fontsize = 14)
    
    for i in range(len(pilot_point_x)):
        plt.plot(pilot_point_x[i], pilot_point_y[i], "ko", markersize=1)
        
    plt.savefig(figureDirectory + "\hk_pest_interp_log_noobs")
    
    # Plot location of observation wells
    
    for i in range(len(obs_well_x)):
        plt.plot(obs_well_x[i], obs_well_y[i], "m*", markersize=10)
        
    plt.grid()
    plt.xlabel("x axis (m)", fontsize = 14)
    plt.ylabel("y axis (m)", fontsize = 14)
    
    plt.savefig(figureDirectory + "\hk_pest_interp_log")
    
    
    # Plotting recharge ###########################################################
    # no recharge modelled
    #convert_recharge = rech_final_array*365*1000
    #
    #plt.figure()
    #plt.subplot(1, 1, 1, aspect='equal')    
    #extent = (delr/2., Lx - delr/2., delc/2., Ly - delc/2.)
    #plt.imshow(np.flipud(convert_recharge[ :, :]), extent=extent, cmap='cool') 
    ## vmax=np.amax(head_1[0, :, :]), vmin = np.amin(head_1[0, :, :])
    #cbr = plt.colorbar()
    #cbr.ax.tick_params(labelsize=14) 
    #cbr.set_label('Estimated Recharge (mm/yr)', fontsize = 14)
    #
    #for i in range(len(pilot_point_x)):
    #    plt.plot(pilot_point_x[i], pilot_point_y[i], "wo")
    #
    ## Plot location of observation wells
    #
    #for i in range(len(obs_well_x)):
    #    plt.plot(obs_well_x[i]*delr, obs_well_y[i]*delc, "m*", markersize=10)
    #    
    #plt.grid()
    #plt.xlabel("x axis (m)", fontsize = 14)
    #plt.ylabel("y axis (m)", fontsize = 14)
    #
    #plt.savefig(figureDirectory + "\pest_rech_interp")
    
    #==============================================================================
    # Plot model results with final hk and recharge array
    #==============================================================================
    final_pest_files = [f for f in os.listdir(final_pest_files_folder)]
    
    #------------------------------------------------------------------------------
    # Load the existing model
    model_ws = this_model_folder
    namefile = modelname + str(".nam")
    ml = flopy.modflow.Modflow.load(namefile, model_ws=model_ws,verbose=False)
    ml.exe_name = "mfnwtdbl.exe" 
    
    ml.external_path = os.path.split(ext_path)[-1]
    
    # Set the format for readying array elements to external for parameters of interest.
    # Can check the ref folder to see that all the parameter files are formed.
    ml.upw.hk.fmtin = "(FREE)" 
    #ml.upw.sy.fmtin = "(FREE)" # Only relevant for transient sims.
    ml.rch.rech.fmtin = "(FREE)"
    
    # This will write the external files in the "ref" folder.
    ml.write_input()
    
    # Now I've written new ref files so I need to delete them
    for f in ext_files:
        file = os.path.join(ext_path, f)
        print(file)
        os.remove(file) 
    
    # Copy all of the final pest files from "final pest files folder" to ref folder so they're 
    # used when I run the model
    
    for f in final_pest_files:
        shutil.copy2(os.path.join(final_pest_files_folder, f),  os.path.join(ext_path, f))
    
    # Now run the model with the estimated parameter files from pest
    ml.run_model()
    
    # Load mt3d model
    mt3d_namefile = modelname_mt + ".mtnam"
    mt = flopy.mt3d.mt.Mt3dms.load(mt3d_namefile)
    mt.exe_name = "mt3dusgsdbl.exe" 
    mt.run_model()
    
    #==============================================================================
    #=====CREATE DATA OUTPUTS======================================================  
    #==============================================================================
     
    # CREATE HEADFILE OBJECTS  
    
    headobj = bf.HeadFile(modelname + '.hds')
    data = headobj.get_alldata()
    times = headobj.get_times()
    
    # CONCENTRATION/AGE FILES - FROM TRANSPORT MODEL
    
    directage_file = modelname_mt + str('.ucn') 
    ucnobjc_direct_age = flopy.utils.binaryfile.UcnFile(directage_file, precision="double")
    age_times = ucnobjc_direct_age.get_times()
    age_raster = ucnobjc_direct_age.get_data(totim=age_times[0])
    
    #==============================================================================
    #=====PLOT HEADS DATA WITH PEST RESULTS========================================  
    #==============================================================================
    
    head_2 = headobj.get_data(totim=times[0]) # One time period: steady-state
    
    levels = np.linspace(0, 400, 101)
    
    # Plot steady state hydraulic heads
    
    plt.figure()
    plt.subplot(1, 1, 1, aspect='equal')    
    plt.imshow(np.flipud(head_2[0, :, :]), extent=extent, cmap='coolwarm_r', 
                         ) # vmax=np.amax(head_1[0, :, :]), vmin = np.amin(head_1[0, :, :])
    cbr = plt.colorbar()
    cbr.ax.tick_params(labelsize=14) 
    cbr.set_label('Head in mAHD', fontsize = 14)
    cs = plt.contour(np.flipud(head_2[0, :, :]), levels=levels, extent=extent, 
                               colors = "k") 
    plt.clabel(cs, inline=1, fontsize=14, fmt='%1i')
    
    #for i in range(len(pilot_point_x)):
    #    plt.plot(pilot_point_x[i], pilot_point_y[i], "wo")
    
    # Plot location of observation wells
    
    for i in range(len(obs_well_x)):
        plt.plot(obs_well_x[i], obs_well_y[i], "m*", markersize=10)
    
    plt.savefig(figureDirectory + "\head_pest_ss")
    
    #==============================================================================
    #   PLOT AGES DATA WITH PEST RESULTS  
    #==============================================================================
    
    plt.figure()
    plt.subplot(1, 1, 1, aspect='equal')    
    
    plt.imshow(np.flipud(age_raster[0, :, :]), extent=extent, cmap='jet') # vmax=np.amax(head_1[0, :, :]), vmin = np.amin(head_1[0, :, :])
    cbr = plt.colorbar()
    cbr.set_label('Age (years)')
    #
    #for i in range(len(pilot_point_x)):
    #    plt.plot(pilot_point_x[i], pilot_point_y[i], "wo")
    
    for i in range(len(obs_well_x)):
        plt.plot(obs_well_x[i], obs_well_y[i], "m*", markersize=10)
    
    plt.savefig(figureDirectory + "\plotage_pest")

    plt.close('all')
    
    
    #==========================================================================
    # Calculating RMSE - at observation wells only
        
    rc_df.index = rc_df.a_wellname
    
    pest_obs_file["Modelled_r"] = np.zeros(len(pest_obs_file.index))
    pest_obs_file["Residual_r"] = np.zeros(len(pest_obs_file.index))
       
    residual_head_sum = 0
    residual_age_sum = 0
    
    rmse_h_list = []
    rmse_a_list = []
    
    weight_age = 1.
    weight_head = 25.
    
    well = 0
    for well in range(len(rc_df.index)):
        wellname   = rc_df.index[well]
        
        #                print(wellname)
         
        row = rc_df.loc[wellname, "b_wellrow"]
        col = rc_df.loc[wellname, "c_wellcol"]
        
        name = obs_name_format.format(wellname, row, col)
        print(name)
        
        pest_obs_file.loc[(str(name)+"_h"), "Modelled_r"] =  head_2[0, row, col]
        pest_obs_file.loc[(str(name)+"_a"), "Modelled_r"] =  age_raster[0, row, col]
               
        residual_head = pest_obs_file.loc[(str(name)+"_h"), "Measured"] - head_2[0, row, col]
        residual_age = pest_obs_file.loc[(str(name)+"_a"), "Measured"] - age_raster[0, row, col]
        
        pest_obs_file.loc[(str(name)+"_h"), "Residual_r"] =  residual_head
        pest_obs_file.loc[(str(name)+"_a"), "Residual_r"] =  residual_age
                          
        residual_head_sum = residual_head_sum + residual_head*residual_head
        residual_age_sum = residual_age_sum + residual_age*residual_age
        
    rmse_head = math.sqrt(residual_head_sum/len(rc_df.index))
    rmse_age = math.sqrt(residual_age_sum/len(rc_df.index))
    
    rmse_h_list.append(rmse_head)
    rmse_a_list.append(rmse_age)
        
    rmse_combined_list = weight_head*np.asarray(rmse_h_list) + weight_age*np.asarray(rmse_a_list)

    # Save the rmse lists:
    name = "rmse_c" 
    with open(os.path.join(dataDirectory, name), 'wb') as fp:
        pickle.dump(rmse_combined_list, fp)
    
    name = "rmse_a" 
    with open(os.path.join(dataDirectory, name), 'wb') as fp:
        pickle.dump(rmse_a_list, fp)
    
    name = "rmse_h"  
    with open(os.path.join(dataDirectory, name), 'wb') as fp:
        pickle.dump(rmse_h_list, fp)
        
    # -----------------------------------------------------------------------------
    # PLot original heads and ages to check
       
    plt.figure()
    plt.subplot(2, 1, 1)    
    plt.imshow(np.flipud(original_head_file[:, :]), extent=extent, cmap='coolwarm_r') 
    cbr = plt.colorbar()
    cbr.ax.tick_params(labelsize=14) 
    cbr.set_label('Real model head in mAHD')
    
    plt.subplot(2, 1, 2, aspect='equal')    
    
    plt.imshow(np.flipud(original_age_file[:, :]), extent=extent, cmap='jet') # vmax=np.amax(head_1[0, :, :]), vmin = np.amin(head_1[0, :, :])
    cbr = plt.colorbar()
    cbr.set_label('Real model age (years)')


    plt.close('all')
    
    # -----------------------------------------------------------------------------
    # RMSE FOR FILTERED MODEL (ALL FILTERED CELLS)

    original_head = np.copy(original_head_file)
    original_age  = np.copy(original_age_file)
    
    r_values_head = []
    r_values_age = []
    
        # Compare original files to pest model files
    pest_head     = np.copy(head_2[0]) #pest_head_array
    pest_age      = np.copy(age_raster[0]) #pest_age_array
        
    filtered_arrays    = [original_head, pest_head, original_age, pest_age]
    
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
               
       # Didn't remove values located at faults because there are no faults

     
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

    name = "head_regression" 
    plt.savefig(os.path.join(figureDirectory, name))
       
    # AGE
    plt.figure()
    plt.title("Original versus pest age everywhere except boundaries, obs wells and faults, more filtered")
    plt.plot(filtered_arrays[2], filtered_arrays[3], "r*")
    axes = plt.gca()
    plt.xlabel('Original age (y)', fontsize=14)
    plt.ylabel('PEST age (y)', fontsize=14)
    
    # Regression analysis
    # Flatten filtered arrays
    orig_flat_a = filtered_arrays[2].flatten()
    pest_flat_a = filtered_arrays[3].flatten()
    
    # Remove nans
    orig_flat_a = orig_flat_a[np.isfinite(orig_flat_a)]
    pest_flat_a = pest_flat_a[np.isfinite(pest_flat_a)]
    
    # Check
    if len(orig_flat_a) == len(pest_flat_a):
        print("Original and pest are the same length - no issues")
    else:
        print("Original and pest arrays are different lengths - follow up!")
    
    slope_a, intercept_a, r_value_a, p_value_a, std_err_a = stats.linregress(orig_flat_a, pest_flat_a)
    print("R-squared: %f" % r_value_a**2)
    
    max_of_max_age = max(np.max(orig_flat_a), np.max(pest_flat_a))
    min_of_min_head = min(np.min(orig_flat_a), np.min(pest_flat_a))
   
    axes.set_xlim([math.floor(min_of_min_head), math.ceil(max_of_max_age)])                
    axes.set_ylim([math.floor(min_of_min_head), math.ceil(max_of_max_age)])
    
    plt.text(1000, 500, "R-squared: %f" % r_value_a**2, fontsize=14)

    name = "age_regression" 
    # At this stage I don't want to save them all.
    # plt.savefig(os.path.join(figureDirectory, name))
    
    # Save the r values for further analysis
    r_values_head.append(r_value_h)
    r_values_age.append(r_value_a)
        
                   
            
    o_h = filtered_arrays[0] 
    p_h = filtered_arrays[1]
    o_a = filtered_arrays[2]
    p_a = filtered_arrays[3]

    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1, aspect = 'equal')
    plt.imshow(np.flipud(o_h), extent=extent, cmap='coolwarm_r')
    cbr = plt.colorbar()
    cbr.set_label('Prior head filtered (m)')       
    
    plt.subplot(2, 2, 2, aspect = 'equal')
    plt.imshow(np.flipud(p_h), extent=extent, cmap='coolwarm_r')
    cbr = plt.colorbar()
    cbr.set_label('Posterior head filtered (m)')       
    
    plt.subplot(2, 2, 3, aspect = 'equal')
    plt.imshow(np.flipud(o_a), extent=extent, cmap='jet')
    cbr = plt.colorbar()
    cbr.set_label('Prior age filtered (y)')       
    
    plt.subplot(2, 2, 4, aspect = 'equal')
    plt.imshow(np.flipud(p_a), extent=extent, cmap='jet')
    cbr = plt.colorbar()
    cbr.set_label('Posterior age filtered (y)')       
            
    #---------------

    # Differences        
    diff_filtered_h = o_h - p_h 
    diff_filtered_a = o_a - p_a
    
    plt.figure(figsize=(9, 12))
    plt.subplot(2, 1, 1)
    plt.imshow(np.flipud(diff_filtered_h), extent=extent, cmap="PiYG", vmin=-15, vmax=15)
    cbr = plt.colorbar()
    cbr.set_label("Diff filtered head (m)")
            
    plt.subplot(2, 1, 2)
    plt.imshow(np.flipud(diff_filtered_a), extent=extent, cmap="PuOr", vmin = -200, vmax=200)
    cbr = plt.colorbar()
    cbr.set_label("Diff filtered age (y)")            
            
    #---------------

    max_list = []
    min_list = []
    mean_list = []
    median_list = []
    std_list = []
    
    list_filtered = [o_h, p_h, diff_filtered_h, o_a, p_a, diff_filtered_a]
    
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
                                             "diff_head",
                                             "prior_age",
                                             "posterior_age",
                                             "diff_age"]
                                            )
      
    #--------------------------------------------------------------------------                                           
    # RMSE for whole grid for filtered arrays
    se_head_f = 0  # Filtered square error head
    se_age_f = 0   # Filtered square error age 
    n_cells_f = 0
    
    for ir in range(param_dict["nrow"]):
        for ic in range(param_dict["ncol"]):
            if np.isnan(p_h[ir][ic]) == True: # Don't include nan values in rmse
                pass
            else:
                n_cells_f = n_cells_f + 1
                
                hse = (p_h[ir][ic] - o_h[ir][ic])**2     # Head square error
                se_head_f = se_head_f + hse             # Sum square error
                 
                ase = (p_a[ir][ic] - o_a[ir][ic])**2      # Age square error
                se_age_f = se_age_f + ase                 # Sum square error    
                
    rmse_head_f = math.sqrt(se_head_f/n_cells_f)                                   
    rmse_age_f  = math.sqrt(se_age_f/n_cells_f)    
    
    #--------------------------------------------------------------------------
    # RMSE for the whole grid... for head and age (unfiltered array)
    original_head = np.copy(original_head_file)
    original_age  = np.copy(original_age_file)
       
    # Compare original files to pest model files
    pest_head     = np.copy(head_2[0]) #pest_head_array
    pest_age      = np.copy(age_raster[0]) #pest_age_array
    
    # Check they are all the same
    np.shape(original_head), np.shape(original_age), np.shape(pest_head), np.shape(pest_age)

    se_head_uf = 0  # Unfiltered square error head
    se_age_uf = 0   # Unfiltered square error age 
    n_cells = 0
    
    for ir in range(param_dict["nrow"]):
        for ic in range(param_dict["ncol"]):
            n_cells = n_cells + 1
            
            hse = (pest_head[ir][ic] - original_head[ir][ic])**2 # Head square error
            se_head_uf = se_head_uf + hse                        # Sum square error
             
            ase = (pest_age[ir][ic] - original_age[ir][ic])**2  # Age square error
            se_age_uf = se_age_uf + ase                        # Sum square error                   
        
    rmse_head_uf = math.sqrt(se_head_uf/n_cells)                                   
    rmse_age_uf  = math.sqrt(se_age_uf/n_cells)   
    
    list_of_rmse_whole_model = [rmse_head_f, rmse_age_f, rmse_head_uf, rmse_age_uf]
    
    name = "rmse_f_uf_whole_mdl"
    with open(os.path.join(dataDirectory, name), 'wb') as fp:
        pickle.dump(list_of_rmse_whole_model, fp)    
                                              
    plt.close("all")
