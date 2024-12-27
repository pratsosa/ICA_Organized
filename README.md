# ICA_Organized
All up-to-date scripts, notebooks, tables, and data needed to recreate ICA and CIV plots on quasar spectra. Additionally, these can be used to recreate the reconstructions and morphed spectra files.  

## Instructions

The scripts in HST_rebin_all_2 are used to rebin HST and SDSS spectra. This code was used 
to create the files in RebinnedSpec_2022Aug11.  

The notebook ICA_HST_2023code reads in the spectra from RebinnedSpec_2022Aug11 and creates the files in reconstructions_final and spectra_morphed.   

The notebook ICAplot_HST_2023code reads in spectra from reconstructions_final and spectra_morphed to create ICA and CIV plots.  

The notebook HST_CIV_March2023 reads in spectra from reconstructions_final and spectra_morphed to create only CIV plots.

## Packages used

The following packages must be installed using pip or through a conda environment in order for the scripts above to run.   

Numpy  
Matplotlib  
Astropy  
Pandas  
lmfit  

## Contributors

This repository was created for Professor Richards' research group at Drexel University.  
The code in this repository was written by Trevor McCaffrey (TVM) and minor edits/comments were added by Alexandros Pratsos (AP)