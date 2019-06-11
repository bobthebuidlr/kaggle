import pandas as pd
import numpy as np
from tqdm import tqdm


def load_data(files=None):
    """ Provide a tuple with filenames to import. Otherwise all files will be imported """

    # The main directory in which the data is located
    DIR = '/Users/bobvanderhelm/datasets/champs-scalar-coupling/'

    # Initialize list to hold the loaded data
    dataframes = []

    # Loop over the provided files
    if files:
        for file in files:
            dataframes.append(pd.read_csv(DIR + file + '.csv'))

    # Loop over the complete file list
    else:
        files = ('dipole_moments', 'mulliken_charges', 'potential_energy', 'scalar_coupling_contributions', 'structures', 'test', 'train')
        for file in files:
            dataframes.append(pd.read_csv(DIR + file + '.csv'))

    # Return dataframes as tuple
    return tuple(dataframes)


def train_to_vec(train, structures):
    """ Convert training data to one-hot encoded matrix of shape=(N, 145) """

    unique_atoms = ('C', 'H', 'N', 'O', 'F')
    max_atoms_per_molecule = 29

    # Initialize matrix
    Xtrain = np.zeros((len(train), (len(unique_atoms) * max_atoms_per_molecule)))

    # List all unique molecule names
    molecules = structures['molecule_name'].unique()

    j = 0
    for molecule in tqdm(molecules):

        # Initialize 0 matrix with (maximum molecules x unique atoms)
        mat = np.zeros((max_atoms_per_molecule, len(unique_atoms)))
        i = 0

        # List every atom of the molecule
        for atom in structures[structures['molecule_name'] == molecule].atom:

            # One hot encode atom within matrix
            mat[i, unique_atoms.index(atom)] = 1
            i += 1

        # Stack matrix columns on top of each other to create 1 vector
        vec = mat.reshape((1, 145))

        # Append the vector to Xtrain
        Xtrain[j, :] = vec

        # Increase index for Xtrain
        j += 1

    return Xtrain
