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
    # TODO: Extend to also include atom pairs and their distance

    unique_atoms = ('C', 'H', 'N', 'O', 'F')
    max_atoms_per_molecule = 29

    # List all unique molecule names
    molecules = structures['molecule_name'].unique()

    # Initialize matrix
    Xtrain = np.zeros((len(molecules) - 1, (len(unique_atoms) * max_atoms_per_molecule)))

    structures = structures.values

    # Index for molecules
    j = 0

    # Index for row in structures
    i = -1

    # Index for atom in molecule
    k = -1

    # Inititalize matrix of 0's per molecule
    mat = np.zeros((max_atoms_per_molecule, len(unique_atoms)))

    for row in tqdm(structures):
        i += 1

        # Molecule name of current row
        molecule = molecules[j]

        # Atom name of current row
        atom = row[2]

        # If current molecule is molecule in scope
        if row[0] == molecule:
            k += 1
            mat[k, unique_atoms.index(atom)] = 1

        # If molecule does not match -> next molecule
        else:
            # Flatten molecule in scope and append to Xtrain
            vec = mat.reshape((1, 145))
            Xtrain[j, :] = vec

            # Start new molecule matrix
            mat = np.zeros((max_atoms_per_molecule, len(unique_atoms)))
            k = 0

            # And append current value to new matrix
            mat[k, unique_atoms.index(atom)] = 1

            j += 1

    return Xtrain
