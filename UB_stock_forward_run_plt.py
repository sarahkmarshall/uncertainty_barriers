
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
import pyemu

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

modelname_overall = modelname[:-6]

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

nrow = param_dict["nrow"]
ncol = param_dict["ncol"]
nlay = param_dict["nlay"]

# WELL FILES ==================================================================
# Getting from this model folder, filtered from removing duplicates
fileName = os.path.join(dataDirectory, "obs_rowcol.csv") # f for filtered
rc_df = pd.read_csv(fileName)

# BEFORE RUNNING MODEL ======================================================== 
# DELETE
pp_file = "hk1pp.dat"
prefix = "hk"
##

def prepare():
    
    # apply pilot point parameters as multipliers
    pp_files = ["hk1pp.dat"] # "sy1pp.dat", , "rech1pp.dat"
    prefixes = ["hk"] # "sy", , "rech"
    
    for pp_file, prefix in zip(pp_files, prefixes): # Remove index [0] --> I'm just testing with hk
        print("PP_file is %s" % (pp_file))
        print("Prefix is %s" % (prefix))
    
        # Make a spatial interpolation from pp points to rest of model domain, this is based on 
        # the pilot points file, and the factors file, both which are created in original 
        # pest set up python script.
        arr = pyemu.utils.geostats.fac2real(pp_file=pp_file, factors_file="pp.fac", out_file=None)    
        
        # Set the original base array --> this is the one that I created with my original model 
        # Here it is saved in the "bak" directory
        base_arr_files = [f for f in os.listdir(bak_dir) if prefix in f.lower()] 
        print("Base array file is %s" % (base_arr_files))
        
        for fname in base_arr_files: # I don't know why this needs to be a loop... only one file per prefix...
            print(fname)
            base_arr = np.loadtxt(os.path.join(bak_dir, fname)) # This loads the original base array
            base_arr *= arr # Multiply this original array by the one created above. K of original file should all be 1.0. Recharge is a multiplication factor.
            # *= means "[expression on the left] = [itself] * [expression on the right]"    
            # This doesn't completely make sense for my hkfile because it shouldn't be a multiplication of that original, I should just be able to replace it.
        
            np.savetxt(os.path.join(out_dir, fname), base_arr, fmt='%8.6e', delimiter='   ') # 


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
    
# RUN THIS SCRIPT ============================================================= 

# __name__ is a built-in variable which evaluates to the name of the current module
#  it can be used to check whether the current script is being run on its own or being imported somewhere else

if __name__ == "__main__": # Script is being run directly...(?)
    prepare()
    run_flow()
    run_transport()
    post_process()
