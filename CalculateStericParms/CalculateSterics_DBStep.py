# Author: George Logan Bartholomew
# Date Finalized: August 3rd, 2023
# Email: logan_bartholomew@berkeley.edu

'''
This program calculates Sterimol parameters and % Buried Volume about
atoms specified in an Excel spreadsheet 
for molecules represented as .xyz files.
'''

# changes the working directory to the directory this script is located in
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
 
import dbstep.Dbstep as db
import numpy as np
import pandas as pd

# generates an array of filenames and atom indices for molecules to analyze
# worksheet format = [filename, atom1, atom2]
worksheet = pd.read_excel('file_names_for_sterics.xlsx')
array = np.array(worksheet)

# these are useful later
num_rows =  array.shape[0]
num_columns = array.shape[1]
                
# initializes list of parameters
parameters = []

# initializes file_names list
file_names = []
for row in range(num_rows):
    file_names.append(array[row, 0])
    
# changes the working directory to the directory of xyz files
files_dir = os.path.dirname(os.path.abspath(__file__)) + '/xyz'
os.chdir(files_dir)

# calculates parameters and adds them to a list
for file in file_names:

    # finds which atoms to use for parameter calculation
    atom1 = array[file_names.index(file), 1]
    atom2 = array[file_names.index(file), 2]

    # Create DBSTEP object
    mol = db.dbstep(file,
                    atom1=atom1,
                    atom2=atom2,
                    commandline=True,
                    verbose=True,
                    sterimol=True,
                    volume=True,
                    measure='classic') 
    
    # Grab Sterimol parameters along N–(para-C) axis
    L = mol.L
    Bmin = mol.Bmin
    Bmax = mol.Bmax
    
    # Grab buried volume about nitrogen atom
    buried_volume = mol.bur_vol
    
    # round parameters to 2 decimals points to make parameters csv look nicer
    L = round(L, 2)
    Bmin = round(Bmin, 2)
    Bmax = round(Bmax, 2)
    buried_volume = round(buried_volume, 2)
    
    # removes '.xyz' from file name to make parameters csv look nicer
    base, ext = os.path.splitext(file)
    
    # generates list of parameters to append to parameters list
    list_to_add = [base, L, Bmin, Bmax, buried_volume]
    
    parameters.append(list_to_add)
    
# changes the working directory to the directory this script is located in
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# adds headers to parameters
headers = ['molecule',
           'L',
           'Bmin',
           'Bmax',
           'BuriedVolume']

parameters.insert(0, headers)

# prints  results after conversion to numpy array
parameters_array = np.array(parameters)

# Convert the array to a DataFrame using pandas
parms_df = pd.DataFrame(parameters_array)

# Specify the file path for saving the Excel spreadsheet
file_path = "output_steric_data.xlsx"

# Export the DataFrame to an Excel spreadsheet using pandas
parms_df.to_excel(file_path, index=False, header=False)

print("\n")
print("Data exported as an Excel spreadsheet successfully.")
