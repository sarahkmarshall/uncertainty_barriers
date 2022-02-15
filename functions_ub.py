# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:05:49 2020

@author: mars0451
"""

#=====IMPORTS==================================================================

import random
import numpy as np
import math
import sys
sys.path.append("..")
import matplotlib.cm as cm
import pandas as pd
import os

#=====IMPORTS==================================================================

'''
These are functions to support my coding in "Project 3". I want them as a 
separate script so that I can make changes to the scripts, yet ensure I am
using the most updated script as possible.
'''

class barriers:
#    def __init__(self):
#        """
#        Initialise the class and add first parameters.
#           Good to do this if there are parameters that get updated in the other functions.
#        """
#        pass
        
    
    def makeFaultCoords(n_faults, nrow, ncol, delr):
        """
        Can make fault coordinates for either the phantom faults or the real fault(s).
        """
        fault_coordinates = []
        
        for i in range(n_faults):
            start_row = random.randint(0, nrow-1) 
            start_col = random.randint(0, ncol-1) 
            # Start fault gen code
            start_coord = [start_row, start_col] # Any x y coord on the grid, this is the left-most point of fault
            
            alpha = random.randint(0, 180) # Any angle between 0 and 180
            
            length_min = 100 #(m)
            length_max = 10000 #(m)
            length_fault = random.randint(length_min/delr, length_max/delr) # multiply length by delc/delr
            length_fault
            if alpha == 0:
                end_coords = [start_coord[0]+length_fault, start_coord[1]]
            elif alpha == 180:
                end_coords = [start_coord[0]-length_fault, start_coord[1]]
            elif alpha == 90:
                end_coords = [start_coord[0], start_coord[1]+length_fault]
            elif alpha > 90:
                Ly = round(length_fault*math.cos(math.radians(180 - alpha)))
                Lx = round(length_fault*math.sin(math.radians(180 - alpha)))
                end_coords = [start_coord[0]-Ly, start_coord[1]+Lx]
            elif alpha < 90:
                Ly = round(length_fault*math.cos(math.radians(alpha)))
                Lx = round(length_fault*math.sin(math.radians(alpha)))
                end_coords = [start_coord[0]+Ly, start_coord[1]+Lx]
            else:
                print("error, angle not between 0 - 180 degr")
                
            fault_coordinates.append([start_coord[0], start_coord[1], end_coords[0], end_coords[1]])
            
        return fault_coordinates
    
    def makeFaultCoords2(n_faults, nrow, ncol, delr, startrow, endrow,
                         startcol, endcol, startornt, endornt,
                         startlen, endlen):
        """
        Can make fault coordinates for either the phantom faults or the real fault(s).
        """
        fault_coordinates = []
        
        for i in range(n_faults):
            start_row = random.randint(startrow, endrow) 
            start_col = random.randint(startcol, endcol) 
            # Start fault gen code
            start_coord = [start_row, start_col] # Any x y coord on the grid, this is the left-most point of fault
            
            alpha = random.randint(startornt, endornt) # Any angle between 0 and 180
            
            length_min = 100 #(m)
            length_max = 10000 #(m)
            length_fault = random.randint(length_min/delr, length_max/delr) # multiply length by delc/delr
            
            if alpha == 0:
                end_coords = [start_coord[0]+length_fault, start_coord[1]]
            elif alpha == 180:
                end_coords = [start_coord[0]-length_fault, start_coord[1]]
            elif alpha == 90:
                end_coords = [start_coord[0], start_coord[1]+length_fault]
            elif alpha > 90:
                Ly = round(length_fault*math.cos(math.radians(180 - alpha)))
                Lx = round(length_fault*math.sin(math.radians(180 - alpha)))
                end_coords = [start_coord[0]-Ly, start_coord[1]+Lx]
            elif alpha < 90:
                Ly = round(length_fault*math.cos(math.radians(alpha)))
                Lx = round(length_fault*math.sin(math.radians(alpha)))
                end_coords = [start_coord[0]+Ly, start_coord[1]+Lx]
            else:
                print("error, angle not between 0 - 180 degr")
                
            fault_coordinates.append([start_coord[0], start_coord[1], end_coords[0], end_coords[1]])
            
        return fault_coordinates
            
    def makeFaultRowCols(fault_coordinates, nrow, ncol):
        
        length_barr_list = []
        barrier_rows_list = []
        barrier_cols_list = []
        
        for barr_number in range(len(fault_coordinates)):    
            
            first_row_b = fault_coordinates[barr_number][0]
            last_row_b = fault_coordinates[barr_number][2]
            len_row_b = abs(last_row_b - first_row_b) 
            barr_rows_b = np.linspace(first_row_b, last_row_b, len_row_b+1)
            
            first_col_b = fault_coordinates[barr_number][1]
            last_col_b = fault_coordinates[barr_number][3]
            len_col_b = abs(last_col_b - first_col_b)
            barr_cols_b = np.linspace(first_col_b, last_col_b, len_col_b+1)
               
            if len_row_b != len_col_b: # Are the number of columns and rows unequal?
            
                if len(barr_rows_b) > len(barr_cols_b): # More rows than columns
                    ratio_b = len(barr_rows_b)/len(barr_cols_b)
                    dir_b = 1 
                #    barr_rows_b = np.linspace(first_row_b, last_row_b, len_row_b+1) 
                
                else: # More columns than rows
                    ratio_b = len(barr_cols_b)/len(barr_rows_b)
                    dir_b = 2
                #    barr_cols_b = np.linspace(first_col_b, last_col_b, len_col_b+1)
                    
                cols_list = []
                rows_list = []
                
                modulus_ratio = ratio_b % 1
                
                if modulus_ratio != 0:
                    number_repeat = int(1/modulus_ratio)
                else:
                    number_repeat = 1
                    
                if dir_b == 1:    # This means that there are more rows than columns
                    for i in range(len(barr_rows_b)): # Add more values to col list, so it's the same length as the rows
                        if i % number_repeat != 0 and i != 0: # Add additional vals at frequency determined by number_repeat
        #                    print("1 is ON")
                            for n in range(int(ratio_b)):
                                if first_col_b < last_col_b:
                                    cols_list.append(first_col_b+i)
                                else:
                                    cols_list.append(first_col_b-i)
                        else:
        #                    print("2 is ON")
                            for n in range(int(ratio_b+1)):
                                if first_col_b < last_col_b:
                                    cols_list.append(first_col_b+i)
                                else:
                                    cols_list.append(first_col_b-i)
                                
                        barr_cols_b = np.asarray(cols_list[:len(barr_rows_b)])
                    
                else: # I.e. there are more columns than rows
                    for i in range(len(barr_cols_b)):
                        if i % number_repeat != 0 and i != 0:
                            for n in range(int(ratio_b)):
                                if first_row_b < last_row_b:
                                    rows_list.append(first_row_b+i)
                                else:
                                    rows_list.append(first_row_b-i)    
                        else:
                            for n in range(int(ratio_b+1)):
                                if first_row_b < last_row_b:
                                    rows_list.append(first_row_b+i)
                                else:
                                    rows_list.append(first_row_b-i)
                                
                        barr_rows_b = np.asarray(rows_list[:len(barr_cols_b)])
            
            else: # If the number of rows and cols are equal --> no changes reqd
                pass
               
            if len(barr_cols_b) == len(barr_rows_b):
                print("Length of rows and columns of barrier are equal")
                length_barr_cells = len(barr_cols_b)
            else:
                print("Rows and columns for barrier aren't equal, you have a problem")
                
            length_barr_list.append(length_barr_cells)
            barrier_rows_list.append(barr_rows_b)
            barrier_cols_list.append(barr_cols_b)
            
        ## Now I need to get rid of the cells that are outside the model domain
        
        length_barr_cells_updated = []
        
        for barr_number in range(len(fault_coordinates)):    # n_phantom_faults
            check_barr_in_range = np.asarray([True]*length_barr_list[barr_number])
            for i in range(length_barr_list[barr_number]):
                if (barrier_rows_list[barr_number][i] >= nrow or 
                    barrier_rows_list[barr_number][i] < 0     or
                    barrier_cols_list[barr_number][i] >= ncol or
                    barrier_cols_list[barr_number][i] < 0):
                    check_barr_in_range[i] = False
                else:
                    pass
            # Now the check acts as a mask, should remove false values from array
            barrier_rows_list[barr_number] = barrier_rows_list[barr_number][check_barr_in_range] 
            barrier_cols_list[barr_number] = barrier_cols_list[barr_number][check_barr_in_range]
        
            if len(barrier_cols_list[barr_number]) == len(barrier_rows_list[barr_number]):
                print("Length of rows and columns of barrier are equal")
                length_barr_cells = len(barrier_cols_list[barr_number])
            else:
                print("Rows and columns for barrier aren't equal, you have a problem")
                
            length_barr_cells_updated.append(length_barr_cells)
                       
        return barrier_rows_list, barrier_cols_list, length_barr_cells_updated
        
    def makeColourList(K_list):
        
        n_categories = 10
        orig_colour_list = cm.inferno(np.linspace(0,1,n_categories)) # cool_r
        category_list = ["< 1e-6 m/d", "1e-6 to 1e-5 m/d", "1e-5 to 1e-4 m/d", "1e-4 to 1e-3 m/d",
                         "1e-3 to 1e-2 m/d", "1e-2 to 1e-1 m/d", "1e-1 to 1 m/d", "1 m/d to 10 m/d",
                         "10 m/d to 100 m/d", "> 100 m/d"]
                               
        colours_list = []
        legend_list = []                      
        # Assign colours for the barriers based on their conductance:
        for i in range(len(K_list)):
            if K_list[i] < 10**-6:
                colours_list.append(orig_colour_list[0])
                legend_list.append(category_list[0])
            elif K_list[i] >= 10**-6 and K_list[i] < 10**-5:
                colours_list.append(orig_colour_list[1])
                legend_list.append(category_list[1])
            elif K_list[i] >= 10**-5 and K_list[i] < 10**-4:
                colours_list.append(orig_colour_list[2]) 
                legend_list.append(category_list[2])
            elif K_list[i] >= 10**-4 and K_list[i] < 10**-3:
                colours_list.append(orig_colour_list[3])  
                legend_list.append(category_list[3])
            elif K_list[i] >= 10**-3 and K_list[i] < 10**-2:
                colours_list.append(orig_colour_list[4])  
                legend_list.append(category_list[4])         
            elif K_list[i] >= 10**-2 and K_list[i] < 10**-1:
                colours_list.append(orig_colour_list[5])   
                legend_list.append(category_list[5])
            elif K_list[i] >= 10**-1 and K_list[i] < 1:
                colours_list.append(orig_colour_list[6])   
                legend_list.append(category_list[6])
            elif K_list[i] >= 10**0 and K_list[i] < 10**1:
                colours_list.append(orig_colour_list[7])
                legend_list.append(category_list[7])
            elif K_list[i] >= 10**1 and K_list[i] < 10**2:
                colours_list.append(orig_colour_list[8])
                legend_list.append(category_list[8])
            else:
                colours_list.append(orig_colour_list[9])
                legend_list.append(category_list[9])
                
        return colours_list, legend_list
        
        
        
class obsWells:
    
    
    def setUpObs(nrow, ncol, delc, delr, sampling_density_optn, overarching_model_folder):
    # What is my sampling density 
        
        surface_area = (ncol*delc)*(nrow*delr) # m2
        
        number_samples_list = [50, 100, 150, 250, 350, 500, 650]
               
        sample_coordinates_list = []
        sampling_density_list = []
        
        for i in range(len(number_samples_list)):
            sample_coords = []
            sampling_density = surface_area/number_samples_list[i]
            sampling_density_list.append(sampling_density)
            
            for j in range(number_samples_list[i]):
                sample_row = random.randint(0,(nrow-1))
                sample_col = random.randint(0,(ncol-1))
                coords_for_sampling = [sample_row, sample_col]
                
                while sample_coords.count(coords_for_sampling) > 0:
                    print("It's already in the list, need to re-sample")
                    sample_row = random.randint(0,(nrow-1))
                    sample_col = random.randint(0,(ncol-1))
                    coords_for_sampling = [sample_row, sample_col]
        
                sample_coords.append(coords_for_sampling)
                
            sample_coordinates_list.append(sample_coords)
            
        # Choose sampling density
#        sampling_density_optn = 1 # I.e. this is the option in "number_samples_list"
        print("Sampling density is %.1f" %sampling_density_list[sampling_density_optn]) 
        
        obs_wells_x_cols = []
        obs_wells_y_rows = []
        
        for i in range(len(sample_coordinates_list[sampling_density_optn])):
            obs_wells_y_rows.append(sample_coordinates_list[sampling_density_optn][i][0]) # Rows
            obs_wells_x_cols.append(sample_coordinates_list[sampling_density_optn][i][1]) # Cols
            
        print("Max row is %d" %max(obs_wells_x_cols))
        print("Max col is %d" %max(obs_wells_y_rows))
        
        print("Min row is %d" %min(obs_wells_x_cols))
        print("Min col is %d" %min(obs_wells_y_rows))
           
        number_samples = number_samples_list[sampling_density_optn]
    
        #------------------------------------------------------------------------------
        # Setting up observation wells to get the data for 
        #
        wellnames_list = []
        for i in range(number_samples):
            well_name = "well_" + str(1+i)
            print(well_name)
            wellnames_list.append(well_name)
        #
        #obs_wells_x = np.array([300., 700., 900.]*4)
        #obs_wells_y = np.array([200.]*3 + [400.]*3 + [600.]*3 + [800.]*3)
        
        well_rows = obs_wells_y_rows
        well_rows = [ int(x) for x in well_rows ]
        
        well_cols = obs_wells_x_cols
        well_cols = [ int(x) for x in well_cols ]
        
        layer_wells = 1
        wells_layer = []
        for i in range(len(well_rows)):
            wells_layer.append(layer_wells)
        
        ## Come up with a better way of distributing the wells - determining which rows
        ## and columns the wells are located.
    
    
    # MAKE ROWS AND COLUMNS PYTHON BASED (NOT MODFLOW!)    
    #    
        well_rows = np.asarray(well_rows) # + 1
        well_cols = np.asarray(well_cols) # + 1
        
        row_cols_obswells_py = pd.DataFrame({  # I.e. "py" = python = 0-based
                                          
                    "a_wellname": wellnames_list,
                    "b_wellrow": well_rows,
                    "c_wellcol": well_cols,
                    "d_welllay": wells_layer,
                                                })
        
        fileName = os.path.join(overarching_model_folder, "obs_rowcol.csv")
        fileName_text = os.path.join(overarching_model_folder, "obs_rowcol.txt")
        row_cols_obswells_py.to_csv(fileName, encoding='utf-8', index=False)
        row_cols_obswells_py.to_csv(fileName_text, header=None, index=None, mode='a', sep=' ')
        
        
    def addMoreObs(nrow, ncol, delc, delr, n_additional_wells, overarching_model_folder):        
                             
        sample_coords = []
        
        for j in range(n_additional_wells):
            sample_row = random.randint(0,(nrow-1))
            sample_col = random.randint(0,(ncol-1))
            coords_for_sampling = [sample_row, sample_col]
            
            while sample_coords.count(coords_for_sampling) > 0:
                print("It's already in the list, need to re-sample")
                sample_row = random.randint(0,(nrow-1))
                sample_col = random.randint(0,(ncol-1))
                coords_for_sampling = [sample_row, sample_col]
    
            sample_coords.append(coords_for_sampling)
                                
        obs_wells_x_cols = []
        obs_wells_y_rows = []
        
        for i in range(n_additional_wells):
            obs_wells_y_rows.append(sample_coords[i][0]) # Rows
            obs_wells_x_cols.append(sample_coords[i][1]) # Cols
            
        print("Max row is %d" %max(obs_wells_x_cols))
        print("Max col is %d" %max(obs_wells_y_rows))
        
        print("Min row is %d" %min(obs_wells_x_cols))
        print("Min col is %d" %min(obs_wells_y_rows))
               
        #------------------------------------------------------------------------------
        # Setting up observation wells to get the data for 
        #
        wellnames_list = []
        for i in range(n_additional_wells):
            well_name = "well_" + str(1+i)
            print(well_name)
            wellnames_list.append(well_name)
        #
        #obs_wells_x = np.array([300., 700., 900.]*4)
        #obs_wells_y = np.array([200.]*3 + [400.]*3 + [600.]*3 + [800.]*3)
        
        well_rows = obs_wells_y_rows
        well_rows = [ int(x) for x in well_rows ]
        
        well_cols = obs_wells_x_cols
        well_cols = [ int(x) for x in well_cols ]
        
        layer_wells = 1
        wells_layer = []
        for i in range(len(well_rows)):
            wells_layer.append(layer_wells)
        
        ## Come up with a better way of distributing the wells - determining which rows
        ## and columns the wells are located.
    
    # MAKE ROWS AND COLUMNS PYTHON BASED (NOT MODFLOW!)    
    #    
        well_rows = np.asarray(well_rows) # + 1
        well_cols = np.asarray(well_cols) # + 1
        
        row_cols_obswells_py = pd.DataFrame({  # I.e. "py" = python = 0-based
                                          
                    "a_wellname": wellnames_list,
                    "b_wellrow": well_rows,
                    "c_wellcol": well_cols,
                    "d_welllay": wells_layer,
                                                })
        
        fileName = os.path.join(overarching_model_folder, "obs_rowcol_e.csv") # "e" for extra
        fileName_text = os.path.join(overarching_model_folder, "obs_rowcol_e.txt")
        row_cols_obswells_py.to_csv(fileName, encoding='utf-8', index=False)
        row_cols_obswells_py.to_csv(fileName_text, header=None, index=None, mode='a', sep=' ')
        

class barrierPP:  # Post-processing results from barriers
    
    # Function to get angle (alpha) and length (hypotenuse length) of fault based on 
    # the row and column coordinates.
    
    def getAngleAndLength(n_faults, fault_rows_list, fault_cols_list, delr, delc):
        alpha_vals = []
        len_hypotenuse_vals = []
        for fault_n in range(n_faults):
            print("Start col: %i, End col: %i" % (fault_cols_list[fault_n][0], fault_cols_list[fault_n][-1]))
            print("Start row: %i, End row: %i" % (fault_rows_list[fault_n][0], fault_rows_list[fault_n][-1]))
            if fault_cols_list[fault_n][-1] > fault_cols_list[fault_n][0]: 
                #Ascending order cols
                ncol_barr = fault_cols_list[fault_n][-1] - fault_cols_list[fault_n][0]
                Lx_barr = ncol_barr*delr
                
                if fault_rows_list[fault_n][-1] > fault_rows_list[fault_n][0]:
                    print("Ascending order rows")
                    nrow_barr = fault_rows_list[fault_n][-1] - fault_rows_list[fault_n][0]
                    Ly_barr = nrow_barr*delc
                    alpha = 90 - math.degrees(math.atan((Ly_barr)/(Lx_barr)))
                    len_hypotenuse = Lx_barr / (math.cos(math.radians(90 - alpha)))
                else:
                    print("Descending order rows")
                    nrow_barr =  fault_rows_list[fault_n][0] - fault_rows_list[fault_n][-1]
                    Ly_barr = nrow_barr*delc
                    alpha =  180 - math.degrees(math.atan((Lx_barr)/(Ly_barr)))
                    len_hypotenuse = Ly_barr/(math.cos(math.radians(180-alpha)))
                        
           
            elif fault_cols_list[fault_n][-1] == fault_cols_list[fault_n][0]:
                print("Stationary order cols")
                if fault_rows_list[fault_n][-1] > fault_rows_list[fault_n][0]:
                    alpha = 180
                    len_hypotenuse =  fault_rows_list[fault_n][-1] - fault_rows_list[fault_n][0]
                else:
                    alpha = 0
                    len_hypotenuse = fault_rows_list[fault_n][0] - fault_rows_list[fault_n][-1]
                
            else:
                print("There shouldn't be any other option?")
                
            alpha_vals.append(alpha)
            len_hypotenuse_vals.append(len_hypotenuse)    
        return alpha_vals, len_hypotenuse_vals

            