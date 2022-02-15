# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:56:41 2019

@author: mars0451
"""

import os
import numpy as np
import pandas as pd
import flopy 
import pylab as plt
import pickle
import sys
import math

functions_path = "C:\workspace\sarahs_functions"

if any(functions_path in s for s in sys.path):
    print("Functions path already in list")
else:
    sys.path.append(r"C:\workspace\sarahs_functions")

import functions_proj_3

# DIRECTORIES =================================================================

modelname = "insert_name"
modelname_mt = modelname + "_MT3D" 
proj_3_folder       = r'C:\workspace\Proj3_PilbaraBarriers'
this_model_folder = os.path.join(proj_3_folder, modelname)
dataDirectory = os.path.join(this_model_folder, "Data")

modelname_overall = modelname[:-2]

overall_folder = os.path.join(proj_3_folder, modelname_overall) 
os.chdir(overall_folder)

with open("param_dict.p", "rb") as fp:
    param_dict = pickle.load(fp)

os.chdir(this_model_folder)
print("Current working directory is %s" %(os.getcwd()))

bak_dir = "bak"
if not os.path.exists(bak_dir):
	os.mkdir(bak_dir)
 
out_dir = "ref"
if not os.path.exists(out_dir):
	os.mkdir(out_dir)

# SETTING SOME PARAMS FROM MODEL ==============================================
normal_score_transform_used = "insert_normal_score_transform_used"

nrow = param_dict["nrow"]
ncol = param_dict["ncol"]
nlay = param_dict["nlay"]

n_phantom_faults = 100

ph_fault_name_list = []

for i in range(n_phantom_faults):
    ph_fault_name_list.append("insert_ph_format" % (i+1))

# Upload fault coordinates from folder

ph_fault_coordinates = np.load(os.path.join(dataDirectory, "ph_fault_coords.npy"))

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

# WELL FILES ==================================================================
# Getting from this model folder, filtered from removing duplicates
fileName = os.path.join(dataDirectory, "obs_rowcol.csv") # f for filtered
rc_df = pd.read_csv(fileName)

# BEFORE RUNNING MODEL ======================================================== 
# DELETE
pp_file = "hk1pp.dat"
prefix = "hk"
##

# ----------------------
# Normal score transform
#-----------------------
''' 
If normal score transform is used, I am converting the parameter values that PEST determines, "hk values" (or more like probabilities
between 0 and 1) to hk values using a probability density function (cumulative).
The reason for doing this is to try and "force" the model to EITHER choose values of K 
that are close to the aquifer itself, OR close to the estimated value of the fault, 10^-3.

Otherwise, PEST doesn't like particularly low values as they don't minimise the regularisation objective function. 
'''
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
    
    #plt.plot(p1, c = "cornflowerblue", label="Barrier K prior")
    #plt.plot(p2, c = "pink", label = "Aquifer K prior")
    #plt.xlabel("X value", fontsize=14)
    #plt.ylabel("Density", fontsize=14)
    
    p = 0.5*p1 + 0.5*p2 # Combined distribution
    plt.plot(p, c = "purple", label = "Bimodal K prior")
    
    P = np.cumsum(p) * (x[1]-x[0]) # Cumulative frequency plot

else:
    pass # No transform to data except log which can be built into PEST

def prepare():
    
    # apply pilot point parameters as multipliers
    pp_files = ["hk1pp.dat"] # "sy1pp.dat", , "rech1pp.dat"
    prefixes = ["hk"] # "sy", , "rech"
    
    for pp_file, prefix in zip(pp_files, prefixes): # Remove index [0] --> I'm just testing with hk
        print("PP_file is %s" % (pp_file))
        print("Prefix is %s" % (prefix))
        
        # Open hk input file, this has the values of hk for each zone.
        current_run_parvals         = pd.read_csv(pp_file, header=None)
        current_run_parvals.columns = ["par_value"]
        current_run_parvals.index   = ["hkaq"] + ph_fault_name_list
        
        if normal_score_transform_used == "YES":               
    #=#= Normal score transformation using my cumulative frequency distribution, P
            for i in current_run_parvals.index:
                print(i)
                
                PP = current_run_parvals.loc[i, "par_value"] # = 1.0
                n = np.argmin((P-PP)**2)
                K = 10**x[n]
                current_run_parvals.loc[i, "par_value"] = K
    
                print(current_run_parvals.loc[i, "par_value"])
        else:
            pass # No transform to data except log which can be built into PEST
      
        # HYDRAULIC CONDUCTIVITY VECTOR
        hk_aquifer = current_run_parvals.loc["hkaq", "par_value"]
      
        hk_array = hk_aquifer * np.ones((nlay, nrow, ncol), dtype=np.float32)     
        
        for barr_number in range(n_phantom_faults):
            barr_name = ph_fault_name_list[barr_number]
            print(barr_name)
            for i in range(ph_fa_length_list[barr_number]):
                hk_array[0, int(ph_fa_rows_list[barr_number][i]), 
                         int(ph_fa_cols_list[barr_number][i])] = current_run_parvals.loc[barr_name, "par_value"]                                   
        
        # Set the original base array --> this is the one that I created with my original model 
        # Here it is saved in the "bak" directory
        base_arr_files = [f for f in os.listdir(bak_dir) if prefix in f.lower()] 
        
        for fname in base_arr_files: # I don't know why this needs to be a loop... only one file per prefix...
            print(fname)
            
            # Now save the changed array file into the "ref" folder, where the model will access it during model run.
            np.savetxt(os.path.join(out_dir, fname), hk_array[0, :, :], fmt='%8.6e', delimiter='   ') # 
            
        return current_run_parvals


# RUNNING MODEL =============================================================== 
           
def run_flow():
    namefile = modelname + ".nam"
    os.system("mfnwtdbl.exe " + str(namefile)) # Use to execute system commands
 
def run_transport():
    mt_namefile = modelname_mt + ".mtnam"
    os.system("mt3dusgsdbl.exe " + str(mt_namefile)) # Use to execute system commands

# AFTER RUNNING MODEL ========================================================= 

# rc_df is in python-indexing (base 0)

def post_process():
    agefile = modelname_mt + str('.ucn') 
    ages = flopy.utils.binaryfile.UcnFile(agefile, precision='double') 
    ages_data = ages.get_alldata()
    obs_ages_nonoise = []
    obs_names_age = []
    
    for n, r, c in zip(rc_df.a_wellname, rc_df.b_wellrow, rc_df.c_wellcol):
        
        # Minus 1 from col and row because python is 0-based, modflow isn't.
        # String formatting: {field_name:conversion} d = decimal integer.
        name = "insert_obs_format".format(n, r, c) 
        print(name)
        obs_names_age.append(str(name)+"_a")
    
        # Now add the age data to the second column of the empty matrix.
        # and add some random noise to the observations.
        obs_ages_nonoise.append(ages_data[0, 0, r, c])
        
    age_obs = pd.DataFrame({                                  
            "a_obs_ages": obs_ages_nonoise,            
            "b_obs_name": obs_names_age
                                        })
    
    fileName_text_out = os.path.join(this_model_folder, "age_obs.out")
    # Delete previous instances, because it adds to them rather than writing a new one.
    if os.path.exists(fileName_text_out):
        print("deleting old age output file")
        os.remove(fileName_text_out) 
    else:
        pass
        
    age_obs.to_csv(fileName_text_out, header=None, index=None, mode='a', sep=' ')
    print("make new age output file")
        
 # ------------------------------------------------------------------------
    # Make an instruction file for my phantom fault locations
    
def postprocess_reg(current_run_parvals):
    Kaq   = current_run_parvals.loc["hkaq", "par_value"] 
    
    prior_vals_pf   = []
    obs_vals_pf     = []
    obs_name_pf_reg = []
    ins_line_pf     = []
    weight_pf_reg   = []
    obsgnme_pf_reg  = []
   
    pf_name_format = "i{0:03d}j{1:03d}_{2:03d}"
          
    for barr_number in range(n_phantom_faults):
        
        barr_name = ph_fault_name_list[barr_number]
        Kbarr = current_run_parvals.loc[barr_name, "par_value"]
        diff_with_aquifer = Kaq - Kbarr
        x_value = math.exp(diff_with_aquifer)
        
        for row, col in zip(ph_fa_rows_list[barr_number], ph_fa_rows_list[barr_number]):
            prior_vals_pf.append(1.0)
            obs_vals_pf.append(x_value)
            name = pf_name_format.format(int(row), int(col), int(barr_number))
            obs_name_pf_reg.append(name)
            ins_line_pf.append("l1 !%s!" % name)
            weight_pf_reg.append(1.0)
            obsgnme_pf_reg.append("regul_ehk")
    
    ph_reg_df = pd.DataFrame({
                              "simval": prior_vals_pf,
                              "obsval": obs_vals_pf,
                              "obsnme": obs_name_pf_reg,
                              "ins_line": ins_line_pf,
                              "weight": weight_pf_reg,
                              "obsgnme": obsgnme_pf_reg})
    
    fileName_text = os.path.join(this_model_folder, "regul_ehk_obs.out")
    ph_reg_df.to_csv(fileName_text, columns=["obsval"], header=False, index=False)
        
#--------------------------------------------------------------------------
        
# RUN THIS SCRIPT ============================================================= 

# __name__ is a built-in variable which evaluates to the name of the current module
#  it can be used to check whether the current script is being run on its own or being imported somewhere else

if __name__ == "__main__": # Script is being run directly...(?)
    current_run_parvals = prepare()
    run_flow()
    run_transport()
    post_process()
    postprocess_reg(current_run_parvals)
