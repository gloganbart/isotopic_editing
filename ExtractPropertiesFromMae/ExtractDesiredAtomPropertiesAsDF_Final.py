# Author: George Logan Bartholomew
# Date Finalized: August 3rd, 2023
# Email: logan_bartholomew@berkeley.edu

'''
This program extracts desired atom properties from Maestro structure
files that have been changed to .txt files and exports them as an 
Excel spreadsheet with headers, molecule names, and atom indices.
'''

# changes the working directory to the directory this script is located in
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# generates an array of filenames and atom indices for molecules to analyze
# worksheet format = [filename, atom1, atom2]
mol_ids_ws = pd.read_excel('file_names_for_properties.xlsx')
mol_ids_ar = np.array(mol_ids_ws)

molecule_names_ar = mol_ids_ar[0:,0]
nitrogen_atom_nums = mol_ids_ar[0:,1]
carbon_atom_nums = mol_ids_ar[0:,2]

# generates a list of molecule names
molecules = []
for molecule in molecule_names_ar:
    molecules.append(molecule)
    
# specify desired atom properties 
desired_properties = ['  r_j_EPN', 
                      '  r_j_ESP_Charges',
                      '  r_j_Max_surface_ESP', 
                      '  r_j_Min_surface_ESP', 
                      '  r_j_f-NN-HOMO', 
                      '  r_j_f-NN-LUMO', 
                      '  r_j_f-NS-HOMO', 
                      '  r_j_f-NS-LUMO', 
                      '  r_j_f-SN-HOMO', 
                      '  r_j_f-SN-LUMO', 
                      '  r_j_f-SS-HOMO', 
                      '  r_j_f-SS-LUMO'
                      ]
    
# initializes result list
output_data = []

# changes working directory to txt files location
txt_files_dir = os.path.dirname(os.path.abspath(__file__)) + '/mae'
os.chdir(txt_files_dir)

for molecule in molecules:
    
    # Read the sample data from the text file
    file_name = molecule
    with open(file_name, 'r') as file:
        input_text = file.read()

    # Split the input text into lines
    lines = input_text.strip().split("\n")
    
    # Initialize variables to find property names
    start_line = None
    end_line = None
    target_lines = []
    
    # Find the starting and ending line indexes of the target section for property names
    for i, line in enumerate(lines):
        if "# First column is atom index #" in line:
            start_line = i
        elif "i_rdk_index" in line:
            end_line = i
    
    # Extract the target lines containing property names
    if start_line is not None and end_line is not None:
        target_lines = lines[start_line:end_line + 1]
    
    # Collect the extracted lines as a list of atom property names
    headers = []
    desired_indices = []
    for line in target_lines:
        headers.append(line)
        if line in desired_properties:
            desired_indices.append(target_lines.index(line))
        else:
            continue
    
    # Initialize an empty list to store the values
    values = []
    
    # gets which nitrogen atom we're interested in
    nitrogen_atom_num = "N" + str(nitrogen_atom_nums[molecules.index(molecule)])
    
    # Iterate through the lines to fine desired data
    # this is currently treating 0s as None, messing up output data ***FIX***
    for line in lines:
        if nitrogen_atom_num in line:
            elements = line.strip().split()
            values = elements[0:]
            break
    
    # Collects desired values based on their indices
    desired_values = [molecule, nitrogen_atom_num]
    for index in desired_indices:
        if index < len(values):
            desired_values.append(values[index])
    output_data.append(desired_values)
    
    # gets which carbon atom we're interested in
    carbon_atom_num = "C" + str(carbon_atom_nums[molecules.index(molecule)])
    
    # Iterate through the lines to fine desired data
    for line in lines:
        if carbon_atom_num in line:
            elements = line.strip().split()
            values = elements[0:]
            break
    
    # Collects desired values based on their indices
    desired_values = [molecule, carbon_atom_num]
    for index in desired_indices:
        if index < len(values):
            desired_values.append(values[index])
    output_data.append(desired_values)

# changes the working directory to the directory this script is located in
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Combines desired values with headers and makes dataframe
desired_properties.insert(0, 'molecule')
desired_properties.insert(1, 'atom')
output_data.insert(0, desired_properties)
output_df = pd.DataFrame(output_data)

# Specify the file path for saving the Excel spreadsheet
file_path = "output_atom_data.xlsx"

# Export the DataFrame to an Excel spreadsheet using pandas
output_df.to_excel(file_path, index=False, header=False)

print("\n")
print("Data exported as an Excel spreadsheet successfully.")

