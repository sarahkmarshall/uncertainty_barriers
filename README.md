# uncertainty_barriers

This study investigates ways to include sharp, low-permeability linear features in groundwater models where their presence is suspected, but unknown.

This study uses PEST to include many more low-permeability features than are likely to occur and then calibrated the model and analyses the output to determine which of these "phantom" structures are probable. 

There are several scripts that interact. I have tried to make them as intuitive as possible but since the work involved many iterations and testing lots of different things there may be some code in there that is relic. 

I will outline the flow for how the codes interact here, but please email me on s.marshall@csiro.au if you have questions or suggestions.

## A note on folder structure

The way that the code runs will make folders for you to store model output. The idea is that there is an Overarching name. I use "UB_01, _02 etc" In the overarching folder, this will contain the obs well locations, phantom structure  coordinates, parameter dictionary, key phantom structures dataframe

Once you have an overarching structure, a separate folder will be set up for each case study: 

- (a) Opn 1 obs wells - optn 0 phantom faults - LOW phimlim
- (b) Opn 1 obs wells - optn 0 phantom faults - HIGH phimlim
- (c) Optn 1 obs wells - optn 1 phantom faults - LOW phimlim
- (d) Optn 1 obs wells - optn 1 phantom faults - HIGH phimlim
- (e) Optn 1 obs wells - optn 2 phantom faults - LOW phimlim
- (f) Optn 1 obs wells - optn 2 phantom faults - HIGH phimlim

- (g) Opn 2 obs wells -  optn 0 phantom faults - LOW phimlim
- (h) Opn 2 obs wells -  optn 0 phantom faults - HIGH phimlim
- (i) Optn 2 obs wells -  optn 1 phantom faults - LOW phimlim
- (j) Optn 2 obs wells -  optn 1 phantom faults - HIGH phimlim
- (k) Optn 2 obs wells - optn 2 phantom faults - LOW phimlim
- (l) Optn 2 obs wells -  optn 2 phantom faults - HIGH phimlim

- (m) Opn 3 obs wells -  optn 0 phantom faults - LOW phimlim
- (n) Opn 3 obs wells -  optn 0 phantom faults - HIGH phimlim
- (o) Optn 3 obs wells -  optn 1 phantom faults - LOW phimlim
- (p) Optn 3 obs wells -  optn 1 phantom faults - HIGH phimlim
- (q) Optn 3 obs wells - optn 2 phantom faults - LOW phimlim
- (r) Optn 3 obs wells -  optn 2 phantom faults - HIGH phimlim

In this folder, it will contain all of the input and output files from the model runs and from PEST. A separate "bak" folder is created to save a copy of the model params, the "Data" folder contains all of the output created during post-processing, the "Figures" folder contains all figures, "final from pest" folder contains final params from PEST inversion, and "ref" folder is another model parameter folder.

## Model functions

Some functions were developed for the scripts to run. These are saved here as "functions_ub". This may be referred to as paper_3_functions in the code, I will work to improve this. These functions are not perfectly parsimonious, the method for generating random features could be improved.

## Setting up the case studies

The functions for setting up the phantom structures case studies and pilot point case studies are called: "setting_up_case_study.py" and "pilot_point_case.py" respectively. In reality the scripts are very similar. This is the first script I suggest you play with this script does the following:
- Imports necessary packages
- Sets up directories
- Pulls in any saved parameters from previous runs that you wish to re-use
- Sets up (and runs if you like) inversions for the 18 different cases (a-r) above 
- Sets up (or pulls in) phantom structure locations + K values
- Plots phantom structure locations
- Sets up (or pulls in) obs well locations, be sure to set the obs well density in the top part of the code under n_subsampled_wells 
- Copies over in your executable files for modflow and mt3d if they are saved elsewhere on your computer
- Sets up the names and K values for the structures
- Sets up (or pulls in) the real structure locations
- Plots the location of the real structure & phantom structures and saves it
- Sets up and runs the "real" model
- Plots the results for age and head from the real model
- Sets up PEST instruction files, setting up obs well names and how they sample age and head files (can add error here)
- Make instruction file for phantom structure locations (or for pilot points)
- Make template files for each parameter
- Setting up the factor file for geostats
- Set up the control file
- Alter a forward run file (Saved "forward_run_file") based on your simulation
- Set up the Tikhonov regularisation.
- ** Important manual step ** here it is important if you are running the second case study where the PHIMLIM is 10* higher than previous run, you need to manually enter the previous value this is the parameter "prev_phi_optimised" I have not figured out how to pull that value out.
- Run PESTCHEK and run PEST (however, I usually run it separately, outside of Spyder).

## Post processing results

There are three steps really to post-processing the results. The script known as "post-processing" (Or "post_processing_pilotpoint" for the pilot point option) will run through all of the plots from the inversion. The ultimate output is generating the RMSE values and also the key structure statistics.

The next step is the "all_cases_statistics" this will pull together all the results from a-r and compare them/make overall statistics.

The final step is to generate the figures, this script I have called "paper_figures".

